"""Run the five preregistered development recipes; never score October outcomes.

python -B -m experiments.pca_fisher_20261026.run
"""
from __future__ import annotations

import os

# Set before importing numpy/sklearn: the existing Windows runtime needs these limits.
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                     "MKL_NUM_THREADS": "4", "MKL_ENABLE_INSTRUCTIONS": "SSE4_2",
                     "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from src.data_loader import OfficialData
from src.decision_policies import activity_flags, decision_time, feature_cutoff
from src.labels import LabelBuilder
from src.metrics import confusion_arrays, scores_from_counts
from src.phase_b_rules import blend_score, ew_scores, topk_at_persist
from experiments.neural_research_common.data import (
    CAMPAIGN_ID, REPORT_ORIGINS, SOURCE_SHA, TARGET_ORDER, TIMING,
    construct_feature_builder, iso_utc, origin_population, sha256_file, sha256_json, validate_source,
)
from experiments.neural_research_common.train_controls import _load_inputs, _runtime_info, _training_rows
from .models import RECIPES, LR_PARAMS, LDA_PARAMS, fit_recipe
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[2]
DATA_RUN = ROOT / "outputs" / CAMPAIGN_ID
SOURCE_DB = (ROOT / "data/legacy/snapshots" / f"{SOURCE_SHA}_20261023"
             / "assignment2_moved_aside" / "data" / "db")
PLAN = ROOT / "docs/exp/PCA_FISHER_PLAN_20261026.md"
EXPECTED_H2 = 0.5476344735753119
REPORT_LAG = pd.Timedelta(days=12, hours=21)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def matrix(frame, prefix):
    return frame[[f"{prefix}{t}" for t in TARGET_ORDER]].to_numpy()


def preflight():
    source = OfficialData(SOURCE_DB, expected_cohort=9000)
    hashes = validate_source(source, ROOT / "reports" / CAMPAIGN_ID / "source_manifest.json")
    fm, lm, f0, feature_paths, label_paths = _load_inputs(DATA_RUN)
    if fm["table_sha256"] != hashes:
        raise RuntimeError("Authenticated caches and preserved source tables differ")
    control_path = DATA_RUN / "controls/predictions.parquet"
    control_manifest = DATA_RUN / "controls/predictions_manifest.json"
    cm = json.loads(control_manifest.read_text(encoding="utf-8"))
    control_config_path = DATA_RUN / "controls/config_resolved.json"
    control_config = json.loads(control_config_path.read_text(encoding="utf-8"))
    report_as_of = pd.Timestamp(control_config["report_label_as_of_utc"])
    if cm["source_sha"] != SOURCE_SHA or sha256_file(control_path) != cm["file_sha256"]:
        raise RuntimeError("Control source/content authentication failed")
    controls = pd.read_parquet(control_path)
    if list(controls.columns) != cm["ordered_columns"] or len(controls) != cm["rows"]:
        raise RuntimeError("Control schema/row-count authentication failed")
    expected_origins = [iso_utc(s) for s in REPORT_ORIGINS]
    if sorted(controls.origin.unique()) != expected_origins or controls.duplicated(["origin", "UserID"]).any():
        raise RuntimeError("Control report origins or keys differ from the registration")
    base = scores_from_counts(confusion_arrays(matrix(controls, "dec_H2_P021_"), matrix(controls, "y_")))["score"]
    if abs(base - EXPECTED_H2) > 1e-12:
        raise RuntimeError(f"Saved H2 score does not reproduce: {base}")
    builder = construct_feature_builder(source)
    events = {t: source.events(t) for t in TARGET_ORDER}
    report_features, checks = {}, []
    if report_as_of < max(REPORT_ORIGINS) + REPORT_LAG or source.latest_observed() < report_as_of:
        raise RuntimeError("Preserved source does not meet the fixed report-label maturity rule")
    for S in REPORT_ORIGINS:
        key = iso_utc(S)
        saved = controls.loc[controls.origin == key].sort_values("UserID").reset_index(drop=True)
        ids = origin_population(source, S).to_numpy(dtype=np.int64)
        if not np.array_equal(ids, saved.UserID.to_numpy()):
            raise RuntimeError(f"{key}: source and saved report populations differ")
        cutoff = feature_cutoff(S, TIMING)
        fresh = builder.build(cutoff, user_ids=pd.Index(ids, name="UserID"))
        cached = pd.read_parquet(feature_paths[key])
        if not np.array_equal(cached.UserID, ids) or not np.array_equal(cached[f0].to_numpy(), fresh[f0].to_numpy()):
            raise RuntimeError(f"{key}: report features do not reproduce from the preserved source")
        early_truth = LabelBuilder(source).build(S, horizon_days=7, label_as_of=S + REPORT_LAG,
                                                user_ids=pd.Index(ids, name="UserID"))
        truth = LabelBuilder(source).build(S, horizon_days=7, label_as_of=report_as_of,
                                          user_ids=pd.Index(ids, name="UserID"))
        if not np.array_equal(truth.UserID, ids) or not np.array_equal(truth[list(TARGET_ORDER)].to_numpy(), matrix(saved, "y_")):
            raise RuntimeError(f"{key}: mature report truth differs from saved control truth")
        label_cache = pd.read_parquet(label_paths[(key, iso_utc(report_as_of))])
        if not np.array_equal(label_cache.UserID, ids) or not np.array_equal(label_cache[list(TARGET_ORDER)], truth[list(TARGET_ORDER)]):
            raise RuntimeError(f"{key}: authenticated report-label cache differs from rebuilt truth")
        persist = activity_flags(events, cutoff, ids, 7)
        ew = ew_scores(events, cutoff, ids, tau_days=7, window_days=90)
        if not np.array_equal(persist, matrix(saved, "persist_")) or not np.allclose(ew, matrix(saved, "ew7_"), rtol=0, atol=1e-12):
            raise RuntimeError(f"{key}: persistence/EW7 controls do not reproduce")
        h2_dec = topk_at_persist(blend_score(matrix(saved, "p_H2_"), ew, 0.5), persist, ids)
        if not np.array_equal(h2_dec, matrix(saved, "dec_H2_P021_")):
            raise RuntimeError(f"{key}: H2/P021 decisions do not reproduce")
        report_features[key] = cached
        checks.append({"origin": key, "users": len(ids), "feature_cutoff": iso_utc(cutoff),
                       "minimum_mature_label_as_of": iso_utc(S + REPORT_LAG),
                       "report_label_as_of": iso_utc(report_as_of), "source_rebuild_parity": True,
                       "minimum_lag_vs_control_truth_mismatches_by_target": dict(zip(TARGET_ORDER,
                           (early_truth[list(TARGET_ORDER)].to_numpy() != matrix(saved, "y_")).sum(axis=0).tolist()))})
    input_paths = [DATA_RUN / "data_build/feature_inputs_manifest.json",
                   DATA_RUN / "data_build/label_inputs_manifest.json",
                   DATA_RUN / "data_build/data_build_manifest.json", control_path, control_manifest,
                   ROOT / "reports" / CAMPAIGN_ID / "f0_schema.json",
                   ROOT / "reports" / CAMPAIGN_ID / "source_manifest.json", control_config_path]
    info = {"source_sha": SOURCE_SHA, "source_db": str(SOURCE_DB), "source_table_sha256": hashes,
            "inputs_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in input_paths},
            "control_path": str(control_path), "control_rows": len(controls), "h2_p021_combined_f1": base,
            "ordered_f0": f0, "target_order": list(TARGET_ORDER), "report_checks": checks,
            "feature_builder_sha256": fm["builder_sha256"],
            "runtime": _runtime_info(), "no_october_report_outcomes": True}
    return info, controls, f0, feature_paths, label_paths, report_features


def freeze(run_dir, preflight_info):
    paths = [PLAN, *Path(__file__).parent.glob("*.py"),
             ROOT / "tests/test_pca_fisher.py", ROOT / "experiments/neural_research_common/train_controls.py",
             ROOT / "src/metrics.py"]
    paths += [ROOT / p for p in preflight_info["feature_builder_sha256"] if p != "experiments/neural_research_common/build_data.py"]
    implementation = {}
    for path in sorted(set(paths)):
        rel = path.relative_to(ROOT)
        dest = run_dir / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        implementation[str(rel)] = sha256_file(dest)
    recipe_config = {"recipes": RECIPES, "lr": LR_PARAMS, "lda": LDA_PARAMS,
                     "pca": {"svd_solver": "full", "whiten": False},
                     "preprocessing": "continuous signed log1p, then training-only StandardScaler on all columns",
                     "weights": "uniform", "training_origins": 56,
                     "primary_policy": "P021", "diagnostic_policy": "P016",
                     "report_origins": [iso_utc(s) for s in REPORT_ORIGINS],
                     "bootstrap": {"draws": 1000, "seed": 42, "cluster": "stable UserID"},
                     "followup_screen": {"pooled_gain_min": 0.001, "positive_weeks_min": 3, "weekly_delta_min": -0.010}}
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), **preflight_info,
              "config": recipe_config, "source_files_sha256": implementation,
              "resolved_lr_defaults": LogisticRegression(**LR_PARAMS).get_params(),
              "resolved_lda_defaults": LinearDiscriminantAnalysis(**LDA_PARAMS).get_params(),
              "interpretation": "Exploratory: previously inspected September weeks; no automatic promotion"}
    write_json(run_dir / "preregistration.json", config)
    return config


def run(run_dir):
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN {run_dir}", flush=True)
    print("PREFLIGHT authenticating source/caches and rebuilding September controls", flush=True)
    info, controls, f0, feature_paths, label_paths, report_features = preflight()
    freeze(run_dir, info)
    print(f"PREFLIGHT PASS: {len(controls)} rows, H2/P021={EXPECTED_H2:.12f}; registration frozen", flush=True)
    controls.to_parquet(run_dir / "controls.parquet", index=False)
    evaluator_source = SOURCE_DB.parents[1] / "evaluation/evaluate.py"
    shutil.copy2(evaluator_source, run_dir / "official_evaluate.py")
    frames, fit_summaries = [], []
    for S in REPORT_ORIGINS:
        key = iso_utc(S)
        A = decision_time(S, TIMING)
        origins, x, y, row_counts, training_keys = _training_rows(S, f0, feature_paths, label_paths, A)
        if len(origins) != 56 or any(s + pd.Timedelta(days=7) > A for s in origins):
            raise RuntimeError("Training origins violate the permitted label clock")
        report = report_features[key]
        saved = controls.loc[controls.origin == key].sort_values("UserID").reset_index(drop=True)
        ids = saved.UserID.to_numpy(dtype=np.int64)
        if not np.array_equal(report.UserID, ids):
            raise RuntimeError("Report row ordering mismatch")
        xe = report[f0].to_numpy(dtype=np.float64)
        persist, ew = matrix(saved, "persist_"), matrix(saved, "ew7_")
        identity = {"origin": key, "decision_time": iso_utc(A), "training_origins": [iso_utc(s) for s in origins],
                    "row_counts": row_counts.tolist(), "training_keys_sha256": sha256_json(training_keys),
                    "x_sha256": array_sha(x), "y_sha256": array_sha(y), "report_x_sha256": array_sha(xe),
                    "report_ids_sha256": array_sha(ids)}
        input_path = run_dir / "report_features" / f"{S.strftime('%Y%m%d')}.parquet"
        input_path.parent.mkdir(exist_ok=True)
        report.to_parquet(input_path, index=False)
        for recipe in RECIPES:
            print(f"FIT {key} {recipe}: {len(x)} training rows", flush=True)
            bundle, summary = fit_recipe(recipe, x, y, f0)
            p = bundle.predict(xe, columns=f0)
            stem = run_dir / "models" / S.strftime("%Y%m%d") / recipe
            stem.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, stem.with_suffix(".joblib"), compress=3)
            replay = joblib.load(stem.with_suffix(".joblib")).predict(xe, columns=f0)
            if not np.array_equal(replay, p):
                raise RuntimeError("Reloaded model predictions do not replay exactly")
            summary.update(identity)
            summary.update({"model_sha256": sha256_file(stem.with_suffix(".joblib")),
                            "prediction_replay_max_abs_error": float(np.max(np.abs(replay - p)))})
            try:
                import psutil
                summary["rss_after_fit_bytes"] = psutil.Process().memory_info().rss
            except ImportError:
                summary["rss_after_fit_bytes"] = None
            write_json(stem.with_suffix(".json"), summary)
            fit_summaries.append(summary)
            frame = pd.DataFrame({"recipe": recipe, "origin": key, "UserID": ids})
            decisions = {"P021": topk_at_persist(blend_score(p, ew, 0.5), persist, ids),
                         "P016": topk_at_persist(p, persist, ids)}
            for j, target in enumerate(TARGET_ORDER):
                frame[f"p_{target}"] = p[:, j]
                frame[f"y_{target}"] = saved[f"y_{target}"].to_numpy()
                for policy, dec in decisions.items():
                    frame[f"dec_{policy}_{target}"] = dec[:, j].astype(np.int8)
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_parquet(run_dir / "predictions.parquet", index=False)
            write_json(run_dir / "progress.json", {"completed_origin_recipe_pairs": len(fit_summaries),
                       "target_classifiers": 6 * len(fit_summaries), "last_origin": key, "last_recipe": recipe})
            print(f"DONE {key} {recipe}: dimensions={summary['preprocessing']['output_dimensions']}, "
                  f"fit_seconds={summary['total_fit_seconds']:.1f}", flush=True)
        del x, y, training_keys
    write_json(run_dir / "fit_summaries.json", fit_summaries)
    artifact_hashes = {str(p.relative_to(run_dir)): sha256_file(p) for p in run_dir.rglob("*")
                       if p.is_file() and "source_snapshot" not in p.parts}
    write_json(run_dir / "artifacts_manifest.json", {"sha256": artifact_hashes})
    from .verify import verify
    verification = verify(run_dir)
    write_json(run_dir / "completion.json", {"status": "complete", "elapsed_seconds": time.perf_counter() - started,
               "verification": verification, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"COMPLETE {run_dir}; elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT / "outputs/pca_fisher_20261026" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run(destination.resolve())


if __name__ == "__main__":
    main()
