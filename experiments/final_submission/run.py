"""Final H2 / P021 pipeline CLI.

Score mode evaluates one past origin against the official evaluator. Submit mode writes a
roster-format decision CSV and checks it with the official format validator.

Run from the project root, for example:
    python -m experiments.final_submission.run --mode score --origin 2026-09-30T16:00:00Z \
        --population proxy --data-dir <.../assignment2/data/db>
    python -m experiments.final_submission.run --mode submit --origin <S> --roster <csv> \
        --policy h2_p021 --out <csv> --data-dir <.../assignment2/data/db>
    python -m experiments.final_submission.run --mode score --origin 2026-09-30T16:00:00Z \
        --population proxy --data-dir <.../assignment2/data/db> --challenger
"""
import os

# Must run before numpy/pandas/sklearn are imported: native BLAS/OpenMP crashes on Windows otherwise.
for _name, _value in (
    ("OMP_NUM_THREADS", "4"), ("OPENBLAS_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
    ("MKL_ENABLE_INSTRUCTIONS", "SSE4_2"), ("MKL_THREADING_LAYER", "SEQUENTIAL"),
):
    os.environ.setdefault(_name, _value)

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import OfficialData, TABLES, TARGETS, load_official_evaluator, utc
from src.decision_policies import activity_flags, decision_time, feature_cutoff, population_anchor
from src.labels import LabelBuilder
from src.metrics import confusion_arrays, scores_from_counts
from src.phase_b_rules import blend_score, ew_scores, topk_at_persist
from experiments.neural_research_common.data import (
    HGB_CONFIG, TIMING, TARGET_ORDER, assert_feature_schema, construct_feature_builder,
    hgb_weight_vector, iso_utc, origin_population, report_train_origins, sha256_file, sha256_json,
)
from experiments.neural_research_common.train_controls import _predict
from experiments.pca_fisher_20261026.models import Preprocessor
from experiments.pca_followup_20261026.models import fit_weighted_lr, fixed_rank_scores

REFERENCE_CAMPAIGN = ROOT / "reports" / "neural_refresh_20261022T0033Z"
REFERENCE_F0_SCHEMA = REFERENCE_CAMPAIGN / "f0_schema.json"
REFERENCE_SOURCE_MANIFEST = REFERENCE_CAMPAIGN / "source_manifest.json"
FREEZE_PATH = ROOT / "reports" / "final_pipeline_20261023" / "FREEZE.json"
CHALLENGER_FREEZE_PATH = ROOT / "reports" / "final_pipeline_20261023" / "FREEZE_CHALLENGER_WPCA.json"
SEALED_START = utc("2026-10-01T00:00:00Z")
HORIZON = pd.Timedelta(days=7)
SCORE_LABEL_LAG = pd.Timedelta(days=12, hours=21)
LEAKAGE_LOOKBACK = pd.Timedelta(days=104)  # 90-day population window anchored 14 days earlier.
SCORE_POLICIES = ("H2_P021", "H2_P016", "P_persist7")
CHALLENGER_POLICY = "H2WPCA_P021"
SUBMIT_POLICY_KEYS = {"h2_p021": "H2_P021", "persist7": "P_persist7", "h2wpca_p021": CHALLENGER_POLICY}
CHALLENGER_SOURCE_FILES = ("experiments/pca_fisher_20261026/models.py",
                           "experiments/pca_followup_20261026/models.py")
EVALUATE_RELPATH = Path("evaluation") / "evaluate.py"
REUSED_SOURCE_FILES = (
    "src/data_loader.py", "src/decision_policies.py", "src/labels.py", "src/features.py",
    "src/baseline_models.py", "src/metrics.py", "src/phase_b_rules.py",
    "experiments/neural_research_common/data.py",
    "experiments/neural_research_common/train_controls.py",
)
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "MKL_ENABLE_INSTRUCTIONS", "MKL_THREADING_LAYER")
POLICY_SPEC = {
    "H2_P021": "0.5*rank01(p_H2)+0.5*rank01(EW7 tau7d lookback90d), top-k with k=P_persist7 "
               "positive count on the scored population, ties ascending UserID",
    "H2_P016": "top-k on raw p_H2 with k=P_persist7 positive count, ties ascending UserID",
    "P_persist7": "1 if the user has a visible row (seen_utc) in [C-7d, C) for the target, "
                  "C=feature_cutoff(S)=S-4h",
}
H2_WEIGHT_FORMULA = "0.5 ** (((S-s)/7d)/2), repeated over actual eligible rows and normalized to mean 1"
POPULATION_RULE = ("proxy: origin_population(source, S) = cohort users with >=5 visible rows "
                   "(seen_utc) in [anchor-90d, anchor), anchor=S-14d-4h; file:<csv> and roster: "
                   "the UserID column, validated as cohort users")
LABEL_RULE = ("training labels: LabelBuilder(s, 7d, label_as_of=A=S-3h); score labels: "
              "LabelBuilder(S, 7d, label_as_of=S+12d21h) unless --truth is given")
TRAINING_RULE = ("daily 16:00Z origins s with s >= S-63d and s+7d <= A=S-3h, ascending; "
                 "for on-grid S this equals report_train_origins(S) (56 origins); for off-grid S "
                 "the same permitted-origin rule is applied relative to the given S")
CHALLENGER_SPEC = ("0.25*rank01(p_H2)+0.25*rank01(p_WPCA)+0.5*rank01(EW7 tau7d lookback90d); p_WPCA = "
                   "weighted_pca99_c1: sign-log1p, training-only StandardScaler, PCA(0.99) on training rows, "
                   "six LogisticRegression(C=1.0) with H2 row weights; top-k with k=P_persist7 positive count, "
                   "ties ascending UserID")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _iso_or_none(value):
    return None if value is None or pd.isna(value) else iso_utc(value)


def _on_grid(S: pd.Timestamp) -> bool:
    return S.hour == 16 and S.minute == 0 and S.second == 0 and S.microsecond == 0 and S.nanosecond == 0


def _ids(values) -> np.ndarray:
    return np.unique(np.asarray(values, dtype=np.int64))


def permitted_training_origins(S) -> list[pd.Timestamp]:
    """Daily 16:00Z origins s with s >= S-63d and s+7d <= A, ascending (A = S-3h).

    Training origins stay on the 16:00Z grid regardless of S. For an on-grid S the result
    must equal report_train_origins(S), which also asserts the 56-origin schedule.
    """
    S = utc(S)
    A = decision_time(S, TIMING)
    first = S - pd.Timedelta(days=63)
    candidate = first.normalize() + pd.Timedelta(hours=16)
    if candidate < first:
        candidate += pd.Timedelta(days=1)
    origins = []
    while candidate + HORIZON <= A:
        origins.append(candidate)
        candidate += pd.Timedelta(days=1)
    if _on_grid(S) and origins != report_train_origins(S):
        raise AssertionError(f"Permitted training origins differ from report_train_origins at {iso_utc(S)}")
    return origins


def _load_f0_schema() -> list[str]:
    columns = json.loads(REFERENCE_F0_SCHEMA.read_text(encoding="utf-8"))["ordered_columns"]
    if len(columns) != 59:
        raise RuntimeError(f"Reference F0 schema has {len(columns)} columns, expected 59")
    return columns


def _check_frame(frame: pd.DataFrame, uids: np.ndarray, f0: list[str], label: str) -> None:
    if not np.array_equal(frame["UserID"].to_numpy(dtype=np.int64), uids):
        raise RuntimeError(f"{label}: feature UserIDs differ from the requested population")
    assert_feature_schema(frame, f0)


def _read_population_csv(path: Path, source: OfficialData) -> np.ndarray:
    frame = pd.read_csv(path)
    if "UserID" not in frame.columns:
        raise ValueError(f"{path} has no UserID column")
    raw = frame["UserID"]
    if raw.isna().any() or raw.duplicated().any():
        raise ValueError(f"{path}: UserID values must be unique and non-null")
    ids = raw.astype("int64")
    unknown = pd.Index(ids).difference(source.cohort_ids)
    if len(unknown):
        raise ValueError(f"{path}: {len(unknown)} UserIDs are not cohort users")
    return _ids(ids)


def _feature_leak(events, label: str, cutoff: pd.Timestamp, uids: np.ndarray) -> dict:
    """Rows feeding features, population, persistence and EW7 for one cutoff."""
    lower = cutoff - LEAKAGE_LOOKBACK
    index = pd.Index(uids, dtype="int64")
    seen_values, event_values, rows = [], [], 0
    for target in TARGET_ORDER:
        frame = events[target]
        sel = frame.loc[frame["seen_utc"].ge(lower) & frame["seen_utc"].lt(cutoff)]
        sel = sel.loc[sel["UserID"].isin(index)]
        rows += len(sel)
        if sel.empty:
            continue
        bad = sel.loc[sel["seen_utc"].ge(cutoff) | sel["event_utc"].ge(cutoff)]
        if len(bad):
            raise AssertionError(f"R2 feature leak at {label} target={target}, cutoff={iso_utc(cutoff)}:\n"
                                 f"{bad.head(20).to_string()}")
        seen_values.append(sel["seen_utc"].max())
        valid_events = sel["event_utc"].dropna()
        if len(valid_events):
            event_values.append(valid_events.max())
    return {
        "label": label, "cutoff_utc": iso_utc(cutoff), "population_users": int(len(index)),
        "rows_considered": int(rows),
        "max_seen_utc": _iso_or_none(max(seen_values) if seen_values else None),
        "max_event_utc": _iso_or_none(max(event_values) if event_values else None),
    }


def _train_label_leak(events, label: str, s: pd.Timestamp, uids: np.ndarray, A: pd.Timestamp) -> dict:
    """Rows making a training label positive: event in [s, s+7d), seen <= A."""
    end = s + HORIZON
    index = pd.Index(uids, dtype="int64")
    seen_values, event_values, rows = [], [], 0
    for target in TARGET_ORDER:
        frame = events[target]
        sel = frame.loc[frame["event_utc"].ge(s) & frame["event_utc"].lt(end) & frame["seen_utc"].le(A)]
        sel = sel.loc[sel["UserID"].isin(index)]
        rows += len(sel)
        if sel.empty:
            continue
        bad = sel.loc[sel["seen_utc"].ge(A) | sel["event_utc"].ge(A)]
        if len(bad):
            raise AssertionError(f"R2 training-label leak at {label} target={target}, A={iso_utc(A)}:\n"
                                 f"{bad.head(20).to_string()}")
        seen_values.append(sel["seen_utc"].max())
        event_values.append(sel["event_utc"].max())
    return {
        "label": label, "origin_utc": iso_utc(s), "label_as_of_utc": iso_utc(A),
        "population_users": int(len(index)), "positive_contributing_rows": int(rows),
        "max_seen_utc": _iso_or_none(max(seen_values) if seen_values else None),
        "max_event_utc": _iso_or_none(max(event_values) if event_values else None),
    }


def _score_label_record(events, S, uids: np.ndarray, as_of: pd.Timestamp) -> dict:
    """Record-only: positive-contributing rows for the score labels, seen <= as_of."""
    end = S + HORIZON
    index = pd.Index(uids, dtype="int64")
    seen_values, event_values, rows = [], [], 0
    for target in TARGET_ORDER:
        frame = events[target]
        sel = frame.loc[frame["event_utc"].ge(S) & frame["event_utc"].lt(end) & frame["seen_utc"].le(as_of)]
        sel = sel.loc[sel["UserID"].isin(index)]
        rows += len(sel)
        if sel.empty:
            continue
        seen_values.append(sel["seen_utc"].max())
        event_values.append(sel["event_utc"].max())
    return {
        "origin_utc": iso_utc(S), "label_as_of_utc": iso_utc(as_of),
        "positive_contributing_rows": int(rows),
        "max_seen_utc": _iso_or_none(max(seen_values) if seen_values else None),
        "max_event_utc": _iso_or_none(max(event_values) if event_values else None),
    }


def _decision_frame(uids: np.ndarray, matrix: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"UserID": uids.astype(np.int64)})
    for j, target in enumerate(TARGET_ORDER):
        frame[target] = np.asarray(matrix[:, j], dtype=np.int64)
    return frame


def _run_evaluate(python: str, evaluate_path: Path, args_list: list[str]) -> dict:
    proc = subprocess.run([python, str(evaluate_path), *args_list],
                          capture_output=True, text=True)
    return {"command": [python, str(evaluate_path), *args_list], "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


@contextmanager
def timed(timings: dict, name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - started, 3)


def _challenger_frozen() -> bool:
    try:
        record = json.loads(CHALLENGER_FREEZE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and "implementation" in record


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m experiments.final_submission.run",
                                     description="Final H2/P021 score or submit pipeline.")
    parser.add_argument("--mode", required=True, choices=["score", "submit"])
    parser.add_argument("--origin", required=True, help="Origin S, ISO UTC, e.g. 2026-09-30T16:00:00Z")
    parser.add_argument("--data-dir", required=True, help="Directory holding the eight *.parquet tables")
    parser.add_argument("--output-root", default="outputs/final_pipeline_20261023")
    parser.add_argument("--population", default="proxy", help="score mode: proxy or file:<csv>")
    parser.add_argument("--truth", help="score mode: official-format truth CSV")
    parser.add_argument("--challenger", action="store_true", help="score mode: also score H2WPCA_P021")
    parser.add_argument("--roster", help="submit mode: CSV with exactly one UserID column")
    parser.add_argument("--policy", choices=["h2_p021", "persist7", "h2wpca_p021"], help="submit mode")
    parser.add_argument("--out", help="submit mode: output CSV; must not already exist")
    args = parser.parse_args(argv)
    if args.mode == "score":
        if args.roster or args.policy or args.out:
            parser.error("score mode does not take --roster, --policy or --out")
        if not (args.population == "proxy" or args.population.startswith("file:")):
            parser.error("--population must be 'proxy' or 'file:<csv>'")
    else:
        if args.truth or args.population != "proxy":
            parser.error("submit mode does not take --truth or --population")
        if args.challenger:
            parser.error("submit mode does not take --challenger; use --policy h2wpca_p021")
        missing = [flag for flag, value in (("--roster", args.roster), ("--policy", args.policy),
                                            ("--out", args.out)) if not value]
        if missing:
            parser.error("submit mode requires " + ", ".join(missing))
    return args


def main(argv=None) -> int:
    wall_started = time.perf_counter()
    args = parse_args(argv)
    mode = args.mode
    challenger = args.challenger or (mode == "submit" and args.policy == "h2wpca_p021")
    S = utc(args.origin)
    C = feature_cutoff(S, TIMING)
    A = decision_time(S, TIMING)
    label_as_of = S + SCORE_LABEL_LAG

    if mode == "score" and S >= SEALED_START and not FREEZE_PATH.exists():
        print("SEALED: October scoring requires FREEZE.json", file=sys.stderr)
        return 4
    if challenger and mode == "score" and S >= SEALED_START and not _challenger_frozen():
        print("SEALED: October challenger scoring requires stage-2 FREEZE_CHALLENGER_WPCA.json", file=sys.stderr)
        return 4
    out_path = None
    if mode == "submit":
        out_path = Path(args.out).resolve()
        if out_path.exists():
            print(f"REFUSED: --out already exists: {out_path}", file=sys.stderr)
            return 2
    truth_path = Path(args.truth).resolve() if args.truth else None
    needs_model = not (mode == "submit" and args.policy == "persist7")

    timings: dict[str, float] = {}
    with timed(timings, "load_source"):
        source = OfficialData(args.data_dir, expected_cohort=9000)
        events = {target: source.events(target) for target in TARGET_ORDER}
        latest_observed = source.latest_observed()
        table_sha256 = {table: sha256_file(source.data_dir / f"{table}.parquet") for table in TABLES}
    evaluator = load_official_evaluator(source.official_repo)
    official_evaluate = source.official_repo / EVALUATE_RELPATH

    with timed(timings, "population"):
        if mode == "submit":
            roster_path = Path(args.roster).resolve()
            uids = _ids(source.official_test_user_ids(roster_path).to_numpy(dtype=np.int64))
            poptag = f"roster-{roster_path.stem}"
        elif args.population == "proxy":
            uids = _ids(origin_population(source, S).to_numpy(dtype=np.int64))
            poptag = "proxy"
        else:
            pop_path = Path(args.population[len("file:"):]).resolve()
            uids = _read_population_csv(pop_path, source)
            poptag = f"file-{pop_path.stem}"
    if len(uids) == 0:
        raise RuntimeError("Scored population is empty")

    now = datetime.now(timezone.utc)
    run_id = f"{mode}_{S.strftime('%Y%m%dT%H%MZ')}_{poptag}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(args.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN_DIR {run_dir}", flush=True)

    reused_hashes = {rel: sha256_file(ROOT / rel) for rel in REUSED_SOURCE_FILES}
    reused_hashes["official:evaluation/evaluate.py"] = sha256_file(official_evaluate)
    challenger_hashes = {rel: sha256_file(ROOT / rel) for rel in CHALLENGER_SOURCE_FILES} if challenger else {}
    challenger_config = {"challenger": {
        "policy": CHALLENGER_POLICY, "spec": CHALLENGER_SPEC,
        "freeze_record": CHALLENGER_FREEZE_PATH.relative_to(ROOT).as_posix(),
    }} if challenger else {}
    reference_tables = None
    if REFERENCE_SOURCE_MANIFEST.is_file():
        reference_tables = json.loads(REFERENCE_SOURCE_MANIFEST.read_text(encoding="utf-8"))["table_sha256"]
    _write_json(run_dir / "config.json", {
        "argv": sys.argv, "run_id": run_id, "mode": mode,
        "origin_utc": iso_utc(S), "feature_cutoff_utc": iso_utc(C), "decision_time_utc": iso_utc(A),
        "population_tag": poptag, "policy_spec": POLICY_SPEC,
        "submit_policy": args.policy if mode == "submit" else None,
        "hgb_config": HGB_CONFIG,
        "hgb_resolved_estimator_params": HistGradientBoostingClassifier(**HGB_CONFIG).get_params(deep=True),
        "h2_weight_formula": H2_WEIGHT_FORMULA,
        "p021_spec": POLICY_SPEC["H2_P021"],
        "population_rule": POPULATION_RULE,
        "population_anchor_utc": iso_utc(population_anchor(S, TIMING)),
        "training_rule": TRAINING_RULE,
        "label_as_of_rule": LABEL_RULE,
        "score_label_as_of_utc": iso_utc(label_as_of),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "joblib": joblib.__version__},
        "env_thread_vars": {name: os.environ.get(name) for name in THREAD_ENV},
        "sha256": {"run.py": sha256_file(Path(__file__)), **reused_hashes, **challenger_hashes},
        **challenger_config,
    })
    _write_json(run_dir / "source.json", {
        "snapshot_sha": source.snapshot_sha, "data_dir": str(source.data_dir),
        "official_repo": str(source.official_repo), "table_sha256": table_sha256,
        "matches_reference_campaign_manifest": (None if reference_tables is None else
                                               table_sha256 == reference_tables),
        "latest_observed_utc": iso_utc(latest_observed), "cohort_users": int(len(source.cohort)),
        "scored_population_users": int(len(uids)),
    })
    if mode == "score" and truth_path is None and latest_observed < label_as_of:
        _write_json(run_dir / "not_mature.json", {
            "origin_utc": iso_utc(S), "required_label_as_of_utc": iso_utc(label_as_of),
            "latest_observed_utc": iso_utc(latest_observed),
        })
        print(f"NOT_MATURE origin={iso_utc(S)} required_as_of={iso_utc(label_as_of)} "
              f"latest_observed={iso_utc(latest_observed)} run_dir={run_dir}", flush=True)
        _write_json(run_dir / "timing.json", {"stages_seconds": timings,
                                             "total_wall_seconds": round(time.perf_counter() - wall_started, 3)})
        return 3

    f0 = _load_f0_schema()
    builder = construct_feature_builder(source)
    feature_items = [("scored", C, uids)]
    origins: list[pd.Timestamp] = []
    x_blocks, y_blocks, row_counts, train_pops, key_rows = [], [], [], [], []

    if needs_model:
        origins = permitted_training_origins(S)
        if len(origins) == 0:
            raise RuntimeError("No permitted training origins")
        with timed(timings, "training_inputs"):
            for s in origins:
                pop_s = _ids(origin_population(source, s).to_numpy(dtype=np.int64))
                pop_index = pd.Index(pop_s, name="UserID")
                frame = builder.build(feature_cutoff(s, TIMING), user_ids=pop_index)
                _check_frame(frame, pop_s, f0, f"training features {iso_utc(s)}")
                labels = LabelBuilder(source).build(s, horizon_days=7, label_as_of=A, user_ids=pop_index)
                if not np.array_equal(labels["UserID"].to_numpy(dtype=np.int64), pop_s):
                    raise RuntimeError(f"Training labels at {iso_utc(s)} disagree with the population")
                x = frame.loc[:, f0].to_numpy(dtype=np.float64, copy=True)
                y = labels.loc[:, list(TARGET_ORDER)].to_numpy(dtype=np.int8, copy=True)
                if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                    raise RuntimeError(f"Nonfinite features or nonbinary labels at {iso_utc(s)}")
                x_blocks.append(x)
                y_blocks.append(y)
                row_counts.append(int(len(pop_s)))
                train_pops.append(pop_s)
                key_rows.extend([iso_utc(s), str(int(u))] for u in pop_s)
                feature_items.append((f"train:{iso_utc(s)}", feature_cutoff(s, TIMING), pop_s))
        X = np.concatenate(x_blocks)
        Y = np.concatenate(y_blocks)
        weights = hgb_weight_vector(S, origins, row_counts)
        training_key_sha256 = sha256_json(key_rows)
        del x_blocks, y_blocks

    with timed(timings, "leakage_check"):
        feature_records = [_feature_leak(events, label, cutoff, ids)
                           for label, cutoff, ids in feature_items]
        train_records = []
        if needs_model:
            train_records = [_train_label_leak(events, f"train:{iso_utc(s)}", s, pop_s, A)
                             for s, pop_s in zip(origins, train_pops)]
        score_record = None
        if mode == "score" and truth_path is None:
            score_record = _score_label_record(events, S, uids, label_as_of)
    max_feature_seen = max(r["max_seen_utc"] for r in feature_records if r["max_seen_utc"])
    scored_c = iso_utc(C)
    summary = (f"R2 PASS features max_seen={max_feature_seen} (< C={scored_c}; checked "
               f"{len(feature_records)} feature cutoffs, each max_seen and max_event < its cutoff) ")
    if needs_model:
        max_train_seen = max(r["max_seen_utc"] for r in train_records if r["max_seen_utc"])
        summary += f"train_labels max_seen={max_train_seen} < A={iso_utc(A)}"
    else:
        summary += "train_labels not applicable (persist7 has no training)"
    print(summary, flush=True)
    _write_json(run_dir / "leakage.json", {
        "result": "PASS", "summary": summary, "feature_cutoffs": feature_records,
        "training_labels": train_records, "score_labels_record_only": score_record,
    })

    p = None
    models = {}
    if needs_model:
        with timed(timings, "fit_models"):
            models_dir = run_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            fit_info = {}
            for j, target in enumerate(TARGET_ORDER):
                y_j = Y[:, j]
                present = np.unique(y_j)
                started = time.perf_counter()
                if len(present) == 1:
                    model = DummyClassifier(strategy="constant", constant=int(present[0])).fit(X, y_j)
                    estimator = "DummyClassifier(constant)"
                else:
                    model = HistGradientBoostingClassifier(**HGB_CONFIG).fit(X, y_j, sample_weight=weights)
                    estimator = "HistGradientBoostingClassifier"
                joblib.dump(model, models_dir / f"{target}.joblib", compress=3)
                models[target] = model
                fit_info[target] = {
                    "estimator": estimator, "positive_count": int(y_j.sum()), "row_count": int(len(y_j)),
                    "fit_seconds": round(time.perf_counter() - started, 3),
                    "iterations": int(getattr(model, "n_iter_", 0)),
                    "estimator_params": model.get_params(deep=True),
                }
                print(f"fit {target} {estimator} positives={int(y_j.sum())} "
                      f"seconds={fit_info[target]['fit_seconds']}", flush=True)
        if challenger:
            with timed(timings, "fit_challenger"):
                prep = Preprocessor(f0, 0.99).fit(X)
                bundle, wpca_summary = fit_weighted_lr("weighted_pca99_c1", 1.0, prep, X, Y, weights, f0)
                joblib.dump(bundle, models_dir / "wpca99_c1.joblib", compress=3)
                _write_json(run_dir / "challenger_fit.json",
                            {"fit": wpca_summary, "preprocessing": prep.summary()})
            print(f"fit challenger wpca99_c1 dims={wpca_summary['dimensions']} "
                  f"seconds={timings['fit_challenger']}", flush=True)
        _write_json(run_dir / "training_origins.json", {
            "origins_utc": [iso_utc(s) for s in origins], "origin_count": len(origins),
            "on_grid_S": bool(_on_grid(S)),
            "row_counts_by_origin": {iso_utc(s): n for s, n in zip(origins, row_counts)},
            "total_rows": int(sum(row_counts)), "label_as_of_utc": iso_utc(A),
            "train_positive_counts": {t: int(Y[:, j].sum()) for j, t in enumerate(TARGET_ORDER)},
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()), "training_key_sha256": training_key_sha256,
            "fits": fit_info,
        })

    with timed(timings, "score_predict"):
        F0 = builder.build(C, user_ids=pd.Index(uids, name="UserID"))
        _check_frame(F0, uids, f0, "scored features")
        persist = activity_flags({t: events[t] for t in TARGET_ORDER}, C, uids, 7).astype(np.int8)
        F0_active = F0[[f"{t}_active_prior_7d" for t in TARGET_ORDER]].to_numpy(dtype=np.int8)
        if not np.array_equal(persist, F0_active):
            raise RuntimeError("P_persist7 does not match F0 active_prior_7d columns")
        decisions = {"P_persist7": persist.copy()}
        ew = None
        if needs_model:
            x_report = F0.loc[:, f0].to_numpy(dtype=np.float64, copy=True)
            p = np.column_stack([_predict(models[t], x_report, j) for j, t in enumerate(TARGET_ORDER)])
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise RuntimeError("Invalid H2 probabilities")
            ew = ew_scores({t: events[t] for t in TARGET_ORDER}, C, uids, tau_days=7, window_days=90)
            decisions["H2_P021"] = topk_at_persist(blend_score(p, ew, 0.5), persist, uids).astype(np.int8)
            decisions["H2_P016"] = topk_at_persist(p, persist, uids).astype(np.int8)
            if challenger:
                p_w = bundle.predict(x_report, columns=f0)
                _, score_w = fixed_rank_scores({"H2": p, "WPCA": p_w}, ew, {"H2": 0.5, "WPCA": 0.5})
                decisions[CHALLENGER_POLICY] = topk_at_persist(score_w, persist, uids).astype(np.int8)

    truth_mat = None
    truth_frame = None
    if mode == "score":
        with timed(timings, "labels_and_scores"):
            if truth_path is not None:
                truth_frame = evaluator.read_predictions(str(truth_path))
                truth_frame = truth_frame.sort_values("UserID").reset_index(drop=True)
                if not np.array_equal(truth_frame["UserID"].to_numpy(dtype=np.int64), uids):
                    raise RuntimeError("Truth UserIDs differ from the scored population")
                truth_frame = truth_frame[["UserID", *TARGET_ORDER]].astype("int64")
                truth_source = "official_truth_csv"
            else:
                built = LabelBuilder(source).build(S, horizon_days=7, label_as_of=label_as_of,
                                                   user_ids=pd.Index(uids, name="UserID"))
                truth_frame = built[["UserID", *TARGET_ORDER]].copy()
                truth_frame["UserID"] = truth_frame["UserID"].astype("int64")
                for target in TARGET_ORDER:
                    truth_frame[target] = truth_frame[target].astype("int64")
                truth_source = "built_labels"
            truth_mat = truth_frame[list(TARGET_ORDER)].to_numpy(dtype=np.int8)

            per_policy = {}
            confusion_rows = []
            cli_results = {}
            truth_csv = run_dir / "truth.csv"
            truth_frame.to_csv(truth_csv, index=False)
            score_policies = (*SCORE_POLICIES, CHALLENGER_POLICY) if challenger else SCORE_POLICIES
            for policy in score_policies:
                pred_frame = _decision_frame(uids, decisions[policy])
                pred_matrix = decisions[policy].astype(np.int8)
                counts = confusion_arrays(pred_matrix, truth_mat)
                ours = scores_from_counts(counts)
                official = evaluator.scores(pred_frame, truth_frame)
                for key in ("macro_f1", "micro_f1", "score"):
                    if abs(float(ours[key]) - float(official[key])) > 1e-10:
                        raise AssertionError(f"{policy} {key}: ours {ours[key]} vs official {official[key]}")
                for j, target in enumerate(TARGET_ORDER):
                    tp, fp, fn, tn = (int(counts[k][j]) for k in ("tp", "fp", "fn", "tn"))
                    denom = 2 * tp + fp + fn
                    confusion_rows.append({
                        "policy": policy, "target": target, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "f1": (2 * tp / denom) if denom else 1.0,
                        "support": tp + fn, "predicted_positive": tp + fp,
                    })
                sub_csv = run_dir / f"submission_{policy}.csv"
                pred_frame.to_csv(sub_csv, index=False)
                cli = _run_evaluate(sys.executable, official_evaluate,
                                    ["--pred", str(sub_csv), "--truth", str(truth_csv)])
                cli_results[policy] = cli
                per_policy[policy] = {
                    "macro_f1": float(ours["macro_f1"]), "micro_f1": float(ours["micro_f1"]),
                    "score": float(ours["score"]),
                    "official_module": {k: float(official[k]) for k in ("macro_f1", "micro_f1", "score")},
                    "official_cli_returncode": cli["returncode"], "official_cli_stdout": cli["stdout"],
                }
                print(f"{policy} macro_f1={ours['macro_f1']:.6f} micro_f1={ours['micro_f1']:.6f} "
                      f"score={ours['score']:.6f} official_cli_rc={cli['returncode']}", flush=True)
            pd.DataFrame(confusion_rows).to_csv(run_dir / "confusion.csv", index=False)
            _write_json(run_dir / "scores.json", {
                "mode": "score", "origin_utc": iso_utc(S), "truth_source": truth_source,
                "label_as_of_utc": None if truth_path is not None else iso_utc(label_as_of),
                "n_users": int(len(uids)), "policies": per_policy,
                "official_cli": {k: {"command": v["command"], "returncode": v["returncode"],
                                     "stdout": v["stdout"], "stderr": v["stderr"]}
                                 for k, v in cli_results.items()},
            })
            scores_failed = any(v["returncode"] != 0 for v in cli_results.values())

    if mode == "submit":
        final_key = SUBMIT_POLICY_KEYS[args.policy]
        submission = _decision_frame(uids, decisions[final_key])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_path, index=False)
        submission.to_csv(run_dir / "submission.csv", index=False)
        with timed(timings, "official_format_check"):
            cli = _run_evaluate(sys.executable, official_evaluate,
                                ["--pred", str(out_path), "--input", str(roster_path)])
        _write_json(run_dir / "evaluate_submit.json", cli)
        print(cli["stdout"], end="", flush=True)
        if cli["stderr"]:
            print(cli["stderr"], end="", file=sys.stderr)
        submit_failed = cli["returncode"] != 0

    cols = {"UserID": uids}
    for j, target in enumerate(TARGET_ORDER):
        if p is not None:
            cols[f"p_H2_{target}"] = p[:, j]
            cols[f"ew7_{target}"] = ew[:, j]
        if challenger:
            cols[f"p_WPCA_{target}"] = p_w[:, j]
        cols[f"persist_{target}"] = persist[:, j]
        if p is not None:
            cols[f"dec_H2_P021_{target}"] = decisions["H2_P021"][:, j]
            cols[f"dec_H2_P016_{target}"] = decisions["H2_P016"][:, j]
        if challenger:
            cols[f"dec_{CHALLENGER_POLICY}_{target}"] = decisions[CHALLENGER_POLICY][:, j]
        cols[f"dec_P_persist7_{target}"] = decisions["P_persist7"][:, j]
        if truth_mat is not None:
            cols[f"y_{target}"] = truth_mat[:, j]
    pd.DataFrame(cols).to_parquet(run_dir / "predictions.parquet", index=False)

    total = round(time.perf_counter() - wall_started, 3)
    _write_json(run_dir / "timing.json", {"stages_seconds": timings, "total_wall_seconds": total})
    print(f"RUN_DIR {run_dir}", flush=True)
    print(f"WALL_SECONDS {total}", flush=True)
    if mode == "score":
        print(f"SCORES written to {run_dir / 'scores.json'}", flush=True)
        if scores_failed:
            return 1
    elif submit_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
