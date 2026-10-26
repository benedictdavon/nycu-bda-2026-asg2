"""One registered relational synthesis group in the fixed PCA ensemble."""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "MKL_ENABLE_INSTRUCTIONS": "SSE4_2", "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from src.data_loader import OfficialData
from src.decision_policies import decision_time, feature_cutoff
from src.phase_b_rules import blend_score, rank01, topk_at_persist
from experiments.neural_research_common.data import (
    CAMPAIGN_ID, REPORT_ORIGINS, TARGET_ORDER, TIMING, hgb_weight_vector, iso_utc, sha256_file, sha256_json,
)
from experiments.neural_research_common.train_controls import _load_inputs, _runtime_info, _training_rows
from experiments.pca_fisher_20261026.models import Preprocessor
from experiments.pca_fisher_20261026.run import array_sha, write_json
from experiments.pca_fisher_20261026.verify import matrix
from experiments.pca_followup_20261026.run import authenticate_parent
from experiments.pca_followup_20261026.models import fit_weighted_lr
from .features import COLUMNS, SynthesizedRepoFeatures, inventory

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "outputs/pca_followup_20261026/run_fixed_v1"
PLAN = ROOT / "docs/exp/DFS_PCA_PLAN_20261026.md"
DATA_RUN = ROOT / "outputs" / CAMPAIGN_ID


def authenticate_challenger():
    registration = json.loads((PARENT / "preregistration.json").read_text(encoding="utf-8"))
    base = Path(registration["parent_run"])
    baseline = authenticate_parent(base)
    for filename, field in (("preregistration.json", "parent_registration_sha256"),
                            ("artifacts_manifest.json", "parent_artifact_manifest_sha256")):
        if sha256_file(base / filename) != registration[field]:
            raise RuntimeError("Challenger parent identity changed")
    manifest = json.loads((PARENT / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["sha256"].items():
        if sha256_file(PARENT / rel) != expected:
            raise RuntimeError(f"Challenger artifact changed: {rel}")
    for rel, expected in registration["source_files_sha256"].items():
        if sha256_file(ROOT / rel) != expected or sha256_file(PARENT / "source_snapshot" / rel) != expected:
            raise RuntimeError(f"Challenger implementation changed: {rel}")
    if json.loads((PARENT / "completion.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError("Challenger run is incomplete")
    return baseline, base


def freeze(run_dir, baseline, base):
    paths = [PLAN, ROOT / "docs/FEATURE_ENGINEERING.md", ROOT / "src/relational_features.py",
             ROOT / "tests/test_dfs_pca.py", *Path(__file__).parent.glob("*.py")]
    files = {}
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        dest = run_dir / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files[str(rel)] = sha256_file(dest)
    registration = {"created_utc": datetime.now(timezone.utc).isoformat(), "parent_run": str(PARENT),
        "base_run": str(base), "source_db": baseline["source_db"], "source_sha": baseline["source_sha"],
        "source_table_sha256": baseline["source_table_sha256"], "runtime": _runtime_info(),
        "source_files_sha256": files, "ordered_f0": baseline["ordered_f0"],
        "ordered_augmented": [*baseline["ordered_f0"], *COLUMNS], "feature_inventory": inventory(),
        "parent_registration_sha256": sha256_file(PARENT / "preregistration.json"),
        "parent_artifact_manifest_sha256": sha256_file(PARENT / "artifacts_manifest.json"),
        "config": {"C": 1.0, "PCA_retained_variance": 0.99, "preprocessing_weights": "uniform",
            "classifier_weights": "normalized H2 two-week half-life", "primary_rank_weights": [0.25, 0.25, 0.50],
            "standalone_branch_rank_weights": [0.50, 0.50], "prediction_counts": "unchanged persistence positives",
            "report_origins": [iso_utc(s) for s in REPORT_ORIGINS], "report_label_as_of": "2026-10-21T11:59:00Z",
            "bootstrap_draws": 1000, "bootstrap_seed": 42,
            "verification_tolerances": {"scaler_atol": 1e-12, "PCA_covariance_atol": 1e-9, "PCA_orthogonality_atol": 1e-12},
            "followup_screen": {"gain_vs_existing_ensemble_min": 0.001, "weekly_wins_min": 3, "weekly_delta_min": -0.010}},
        "scope": "One 25-feature DFS-style group, 24 augmented target fits and six control-parity fits; exploratory September comparison"}
    write_json(run_dir / "preregistration.json", registration)
    return registration


def read_extra(path, expected_ids):
    frame = pd.read_parquet(path)
    if not np.array_equal(frame.UserID.to_numpy(), np.asarray(expected_ids)):
        raise RuntimeError("Synthesized feature keys differ from the F0 cache population")
    return frame[list(COLUMNS)].to_numpy(dtype=np.float64)


def training_extra(origins, feature_paths, generated):
    blocks = []
    for s in origins:
        key = iso_utc(s)
        ids = pd.read_parquet(feature_paths[key], columns=["UserID"]).UserID.to_numpy()
        blocks.append(read_extra(generated[key], ids))
    return np.concatenate(blocks)


def diagnostics(origin, x, extra, f0):
    hashes = {}
    for j, name in enumerate(f0):
        hashes.setdefault(array_sha(x[:, j]), []).append(name)
    result = []
    for j, name in enumerate(COLUMNS):
        values = extra[:, j]
        result.append({"origin": origin, "feature": name, "training_rows": len(values),
            "nonzero_fraction": float((values != 0).mean()), "mean": float(values.mean()),
            "std_ddof0": float(values.std()), "max": float(values.max()),
            "constant": bool(np.ptp(values) == 0),
            "exact_duplicate_f0_columns": ";".join(hashes.get(array_sha(values), []))})
    return result


def run(run_dir):
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN {run_dir}", flush=True)
    baseline, base = authenticate_challenger()
    fm, lm, f0, feature_paths, label_paths = _load_inputs(DATA_RUN)
    if f0 != baseline["ordered_f0"] or fm["table_sha256"] != baseline["source_table_sha256"]:
        raise RuntimeError("Source/cache identity differs from the parent")
    registration = freeze(run_dir, baseline, base)
    print("PREFLIGHT PASS; single-group configuration and expressions frozen", flush=True)
    for filename in ("controls.parquet", "official_evaluate.py"):
        shutil.copy2(PARENT / filename, run_dir / filename)
    source = OfficialData(Path(baseline["source_db"]), expected_cohort=9000)
    builder = SynthesizedRepoFeatures(source)
    write_json(run_dir / "source_audit.json", builder.source_summary)
    needed = set()
    for S in REPORT_ORIGINS:
        needed.add(iso_utc(S))
        needed.update(iso_utc(S - pd.Timedelta(days=i)) for i in range(63, 7, -1))
    generated, cache_rows = {}, []
    for index, key in enumerate(sorted(needed), start=1):
        original = pd.read_parquet(feature_paths[key], columns=["UserID"])
        cutoff = feature_cutoff(pd.Timestamp(key), TIMING)
        frame, audit = builder.build(cutoff, original.UserID.to_numpy())
        path = run_dir / "features" / f"{pd.Timestamp(key).strftime('%Y%m%d')}.parquet"
        path.parent.mkdir(exist_ok=True)
        frame.to_parquet(path, index=False)
        generated[key] = path
        cache_rows.append({"origin": key, "path": str(path.relative_to(run_dir)), "rows": len(frame),
            "sha256": sha256_file(path), "f0_cache_sha256": sha256_file(feature_paths[key]),
            "ordered_population_sha256": array_sha(original.UserID.to_numpy(dtype=np.int64)), "audit": audit})
        if index == 1 or index % 10 == 0 or index == len(needed):
            print(f"FEATURES {index}/{len(needed)} {key}: supported={audit['supported_users']}/{len(frame)}", flush=True)
    write_json(run_dir / "feature_cache_manifest.json", {"ordered_columns": list(COLUMNS), "entries": cache_rows})
    del builder, source
    gc.collect()
    controls = pd.read_parquet(run_dir / "controls.parquet")
    frames, summaries, feature_diagnostics = [], [], []
    for origin_index, S in enumerate(REPORT_ORIGINS):
        key, day = iso_utc(S), S.strftime("%Y%m%d")
        origins, x, y, row_counts, training_keys = _training_rows(S, f0, feature_paths, label_paths, decision_time(S, TIMING))
        identity = {"x_sha256": array_sha(x), "y_sha256": array_sha(y), "training_keys_sha256": sha256_json(training_keys)}
        parent_fit = json.loads((PARENT / "models" / day / "weighted_pca99_c1.json").read_text(encoding="utf-8"))
        if any(parent_fit[k] != v for k, v in identity.items()):
            raise RuntimeError("Training data differs from parent weighted PCA99")
        weights = hgb_weight_vector(S, origins, row_counts)
        if array_sha(weights) != parent_fit["weights_sha256"]:
            raise RuntimeError("Classifier weights differ from the parent")
        report = pd.read_parquet(feature_paths[key])
        xe = report[f0].to_numpy(dtype=np.float64)
        if origin_index == 0:
            print(f"CONTROL PARITY {key}: fresh F0 PCA99 and weighted LR", flush=True)
            prep = Preprocessor(f0, 0.99).fit(x)
            control, control_summary = fit_weighted_lr("F0_parity", 1.0, prep, x, y, weights, f0)
            expected = joblib.load(PARENT / "models" / day / "weighted_pca99_c1.joblib").predict(xe, columns=f0)
            actual = control.predict(xe, columns=f0)
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Fresh F0 control differs from parent: max error {np.max(np.abs(actual - expected))}")
            joblib.dump(control, run_dir / "control_parity.joblib", compress=3)
            write_json(run_dir / "control_parity.json", {**control_summary, "origin": key, "exact_probability_parity": True})
            del control, prep
        extra = training_extra(origins, feature_paths, generated)
        feature_diagnostics.extend(diagnostics(key, x, extra, f0))
        x_aug = np.column_stack([x, extra])
        xe_aug = np.column_stack([xe, read_extra(generated[key], report.UserID.to_numpy())])
        columns = registration["ordered_augmented"]
        print(f"FIT {key}: {len(x_aug)} rows, {len(columns)} inputs", flush=True)
        prep_started = time.perf_counter()
        prep = Preprocessor(columns, 0.99).fit(x_aug)
        prep_seconds = time.perf_counter() - prep_started
        bundle, summary = fit_weighted_lr("augmented_pca99_c1", 1.0, prep, x_aug, y, weights, columns)
        p = bundle.predict(xe_aug, columns=columns)
        stem = run_dir / "models" / day
        stem.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, stem.with_suffix(".joblib"), compress=3)
        np.testing.assert_array_equal(joblib.load(stem.with_suffix(".joblib")).predict(xe_aug, columns=columns), p)
        summary.update({**identity, "origin": key, "training_origins": [iso_utc(s) for s in origins],
            "row_counts": row_counts.tolist(), "augmented_x_sha256": array_sha(x_aug),
            "weights_sha256": array_sha(weights), "model_sha256": sha256_file(stem.with_suffix(".joblib")),
            "preprocessing": prep.summary(), "preprocessing_seconds": prep_seconds, "replay_max_abs_error": 0.0,
            "preprocessing_reused": False, "preprocessing_source": "fresh training-only 84-column representation"})
        write_json(stem.with_suffix(".json"), summary)
        summaries.append(summary)
        saved = controls.loc[controls.origin == key].sort_values("UserID").reset_index(drop=True)
        if not np.array_equal(saved.UserID.to_numpy(), report.UserID.to_numpy()):
            raise RuntimeError("Scored user keys differ from controls")
        ids = saved.UserID.to_numpy(dtype=np.int64)
        persist, ew = matrix(saved, "persist_"), matrix(saved, "ew7_")
        ensemble_score = 0.5 * (0.5 * rank01(matrix(saved, "p_H2_")) + 0.5 * rank01(p)) + 0.5 * rank01(ew)
        scores = {"augmented_branch": blend_score(p, ew, 0.5), "augmented_ensemble": ensemble_score}
        frame = pd.DataFrame({"origin": key, "UserID": ids})
        for j, target in enumerate(TARGET_ORDER):
            frame[f"p_{target}"] = p[:, j]
            frame[f"y_{target}"] = saved[f"y_{target}"].to_numpy()
        for policy, score in scores.items():
            dec = topk_at_persist(score, persist, ids)
            for j, target in enumerate(TARGET_ORDER):
                frame[f"score_{policy}_{target}"] = score[:, j]
                frame[f"dec_{policy}_{target}"] = dec[:, j].astype(np.int8)
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_parquet(run_dir / "predictions.parquet", index=False)
        write_json(run_dir / "progress.json", {"completed_origins": len(frames), "candidate_target_fits": len(frames) * 6})
        print(f"DONE {key}: {prep.pca.n_components_} PCs; preprocessing {prep_seconds:.1f}s, classifier {summary['total_fit_seconds']:.1f}s", flush=True)
        del x, y, x_aug, extra, prep, bundle, training_keys
        gc.collect()
    write_json(run_dir / "fit_summaries.json", summaries)
    pd.DataFrame(feature_diagnostics).to_csv(run_dir / "training_feature_diagnostics.csv", index=False)
    hashes = {str(p.relative_to(run_dir)): sha256_file(p) for p in run_dir.rglob("*")
              if p.is_file() and "source_snapshot" not in p.parts}
    write_json(run_dir / "artifacts_manifest.json", {"sha256": hashes})
    from .verify import verify_run
    verification = verify_run(run_dir)
    write_json(run_dir / "completion.json", {"status": "complete", "elapsed_seconds": time.perf_counter() - started,
        "verification": verification, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"COMPLETE {run_dir}: {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.output or ROOT / "outputs/dfs_pca_20261026" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run(path.resolve())


if __name__ == "__main__":
    main()
