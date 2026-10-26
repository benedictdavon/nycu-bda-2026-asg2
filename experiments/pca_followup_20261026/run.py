"""Run the fixed 12-candidate PCA follow-up without reading October report outcomes."""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "MKL_ENABLE_INSTRUCTIONS": "SSE4_2", "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from experiments.neural_research_common.data import (
    CAMPAIGN_ID, REPORT_ORIGINS, TIMING, hgb_weight_vector, iso_utc,
    sha256_file, sha256_json,
)
from experiments.neural_research_common.train_controls import _load_inputs, _runtime_info, _training_rows
from experiments.pca_fisher_20261026.run import array_sha, write_json
from src.decision_policies import decision_time
from .models import BASE_ENSEMBLES, WEIGHTED_RECIPES, all_recipes, fit_weighted_lr

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "outputs/pca_fisher_20261026/run_fixed_v2"
DATA_RUN = ROOT / "outputs" / CAMPAIGN_ID
PLAN = ROOT / "docs/exp/PCA_FOLLOWUP_PLAN_20261026.md"


def authenticate_parent(parent):
    """Read-only authentication of the previously verified run and controlling inputs."""
    registration_path = parent / "preregistration.json"
    manifest_path = parent / "artifacts_manifest.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if json.loads((parent / "completion.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise RuntimeError("Parent experiment is incomplete")
    for rel, expected in manifest["sha256"].items():
        if sha256_file(parent / rel) != expected:
            raise RuntimeError(f"Parent artifact changed: {rel}")
    for rel, expected in registration["source_files_sha256"].items():
        if sha256_file(parent / "source_snapshot" / rel) != expected or sha256_file(ROOT / rel) != expected:
            raise RuntimeError(f"Parent implementation changed: {rel}")
    for rel, expected in registration["inputs_sha256"].items():
        if sha256_file(ROOT / rel) != expected:
            raise RuntimeError(f"Controlling parent input changed: {rel}")
    for table, expected in registration["source_table_sha256"].items():
        if sha256_file(Path(registration["source_db"]) / f"{table}.parquet") != expected:
            raise RuntimeError(f"Preserved source changed: {table}")
    return registration


def freeze(run_dir, parent_registration):
    paths = [PLAN, *Path(__file__).parent.glob("*.py"), ROOT / "tests/test_pca_followup.py"]
    files = {}
    for path in sorted(paths):
        rel = path.relative_to(ROOT)
        target = run_dir / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(rel)] = sha256_file(target)
    registration = {"created_utc": datetime.now(timezone.utc).isoformat(), "parent_run": str(PARENT),
        "parent_registration_sha256": sha256_file(PARENT / "preregistration.json"),
        "parent_artifact_manifest_sha256": sha256_file(PARENT / "artifacts_manifest.json"),
        "parent_source_sha": parent_registration["source_sha"],
        "source_files_sha256": files, "runtime": _runtime_info(),
        "config": {"base_ensembles": BASE_ENSEMBLES, "weighted_recipes": WEIGHTED_RECIPES,
                   "all_candidates": all_recipes(), "EW7_share": 0.5, "rank_intermediate_mixture_again": False,
                   "preprocessing_weights": "uniform; reuse exact parent PCA99 transform",
                   "classifier_weights": "H2 two-week half-life, normalized to mean one",
                   "prediction_count": "unchanged persistence-positive count per target and report origin",
                   "report_origins": [iso_utc(s) for s in REPORT_ORIGINS],
                   "report_label_as_of": "2026-10-21T11:59:00Z", "bootstrap_draws": 1000, "bootstrap_seed": 42,
                   "followup_screen": {"pooled_gain_min": 0.001, "positive_weeks_min": 3, "worst_week_delta_min": -0.010}},
        "interpretation": "Exploratory; previously inspected September weeks, 12 candidates, no automatic promotion"}
    write_json(run_dir / "preregistration.json", registration)
    return registration


def run(run_dir):
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"RUN {run_dir}", flush=True)
    parent_registration = authenticate_parent(PARENT)
    fm, lm, f0, feature_paths, label_paths = _load_inputs(DATA_RUN)
    if f0 != parent_registration["ordered_f0"] or fm["table_sha256"] != parent_registration["source_table_sha256"]:
        raise RuntimeError("Follow-up caches differ from parent inputs")
    freeze(run_dir, parent_registration)
    print("PREFLIGHT PASS; fixed 12-candidate registration frozen", flush=True)
    for filename in ("controls.parquet", "official_evaluate.py"):
        shutil.copy2(PARENT / filename, run_dir / filename)
    shutil.copy2(PARENT / "predictions.parquet", run_dir / "parent_predictions.parquet")
    weighted_frames, fit_summaries = [], []
    for S in REPORT_ORIGINS:
        key, day = iso_utc(S), S.strftime("%Y%m%d")
        origins, x, y, row_counts, training_keys = _training_rows(S, f0, feature_paths, label_paths, decision_time(S, TIMING))
        if len(origins) != 56:
            raise RuntimeError("Unexpected training-origin count")
        parent_stem = PARENT / "models" / day / "pca99_lr"
        parent_fit = json.loads(parent_stem.with_suffix(".json").read_text(encoding="utf-8"))
        input_identity = {"x_sha256": array_sha(x), "y_sha256": array_sha(y),
                          "training_keys_sha256": sha256_json(training_keys)}
        if any(parent_fit[k] != v for k, v in input_identity.items()):
            raise RuntimeError("Weighted classifier inputs differ from parent PCA99 training inputs")
        preprocessor = joblib.load(parent_stem.with_suffix(".joblib")).preprocessor
        weights = hgb_weight_vector(S, origins, row_counts)
        report = pd.read_parquet(PARENT / "report_features" / f"{day}.parquet")
        xe = report[f0].to_numpy(dtype=np.float64)
        if array_sha(xe) != parent_fit["report_x_sha256"]:
            raise RuntimeError("Report features differ from parent fit")
        for name, C in WEIGHTED_RECIPES.items():
            print(f"FIT {key} {name}: {len(x)} rows, {preprocessor.pca.n_components_} PCs", flush=True)
            bundle, summary = fit_weighted_lr(name, C, preprocessor, x, y, weights, f0)
            p = bundle.predict(xe, columns=f0)
            stem = run_dir / "models" / day / name
            stem.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, stem.with_suffix(".joblib"), compress=3)
            np.testing.assert_array_equal(joblib.load(stem.with_suffix(".joblib")).predict(xe, columns=f0), p)
            summary.update({**input_identity, "origin": key, "ordered_f0": f0,
                "training_origins": [iso_utc(s) for s in origins], "row_counts": row_counts.tolist(),
                "weights_sha256": array_sha(weights), "model_sha256": sha256_file(stem.with_suffix(".joblib")),
                "parent_pca99_model_sha256": parent_fit["model_sha256"], "replay_max_abs_error": 0.0})
            write_json(stem.with_suffix(".json"), summary)
            fit_summaries.append(summary)
            frame = pd.DataFrame({"recipe": name, "origin": key, "UserID": report.UserID.to_numpy()})
            for j, t in enumerate(parent_registration["target_order"]):
                frame[f"p_{t}"] = p[:, j]
            weighted_frames.append(frame)
            pd.concat(weighted_frames, ignore_index=True).to_parquet(run_dir / "weighted_probabilities.parquet", index=False)
            write_json(run_dir / "progress.json", {"completed_origin_recipe_pairs": len(fit_summaries),
                "target_fits": len(fit_summaries) * 6, "last_origin": key, "last_recipe": name})
            print(f"DONE {key} {name}: {summary['total_fit_seconds']:.1f}s", flush=True)
        del x, y, training_keys
    write_json(run_dir / "fit_summaries.json", fit_summaries)
    from .verify import construct_candidates, verify_run
    predictions = construct_candidates(pd.read_parquet(run_dir / "controls.parquet"),
        pd.read_parquet(run_dir / "parent_predictions.parquet"), pd.read_parquet(run_dir / "weighted_probabilities.parquet"))
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    files = {str(p.relative_to(run_dir)): sha256_file(p) for p in run_dir.rglob("*")
             if p.is_file() and "source_snapshot" not in p.parts}
    write_json(run_dir / "artifacts_manifest.json", {"sha256": files})
    verification = verify_run(run_dir)
    write_json(run_dir / "completion.json", {"status": "complete", "elapsed_seconds": time.perf_counter() - started,
               "verification": verification, "completed_utc": datetime.now(timezone.utc).isoformat()})
    print(f"COMPLETE {run_dir}: {time.perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT / "outputs/pca_followup_20261026" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run(destination.resolve())


if __name__ == "__main__":
    main()
