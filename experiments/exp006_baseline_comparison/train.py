"""EXP-006 reproducible baseline training against EXP-005 datasets.

Run from your personal Assignment 2 project root:
  python -m experiments.exp006_baseline_comparison.train --models zero,persistence,logistic
  python -m experiments.exp006_baseline_comparison.train --models zero,persistence,logistic,hgb

Labels from the instructor's future November window are NEVER read, and the
EXP-005 historical holdout stays sealed. Calibration/report periods are temporal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.data_loader import TARGETS, load_official_evaluator
from src.dataset_io import find_run, load_dataset_run, split_xy, KEYS
from .protocol import hash_config, load_config, select_time_slices
from src.baseline_models import MODELS, fit_bundle, probabilities
from src.metrics import (
    binary_with_thresholds, detailed_metrics, official_week_check, tune_thresholds,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(2 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_predictions(path: Path, frame: pd.DataFrame, pred: np.ndarray) -> None:
    d = frame[KEYS].copy()
    for col, vals in zip(TARGETS, pred.T):
        d[col] = vals.astype(np.int8)
    d.to_csv(path, index=False)


def _evaluate_slice(official, frame: pd.DataFrame, pred: np.ndarray, *,
                    segment: str, method: str, model: str,
                    verify_official: bool = True) -> tuple[list[dict], list[dict], dict]:
    metrics = []
    per_target = []
    checks = {}
    for name, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("eligible_proxy", frame.eligible_proxy.to_numpy() == 1),
    ):
        subset = frame.loc[mask].reset_index(drop=True)
        binary = pred[mask]
        if subset.empty:
            raise ValueError(f"No {name} users in {segment}")
        summary, table = detailed_metrics(binary, subset[list(TARGETS)].to_numpy())
        check = official_week_check(official, subset, binary) if verify_official else {}
        checks[name] = check
        metrics.append({"model": model, "segment": segment, "threshold_mode": method,
                        "population": name, "weeks": int(subset.cutoff_utc.nunique()), **summary})
        for line in table.to_dict("records"):
            per_target.append({"model": model, "segment": segment, "threshold_mode": method,
                               "population": name, **line})
    return metrics, per_target, checks


def train_run(*, exp005_run: Path, cfg: dict, model_names: list[str],
              output_dir: Path, overwrite: bool = False,
              dry_run: bool = False) -> dict:
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"EXP-006 output exists: {output_dir}. Use --overwrite intentionally.")
    if len(model_names) != len(set(model_names)) or any(m not in MODELS for m in model_names):
        raise ValueError(f"Model names must be distinct and among {MODELS}")
    exp005_manifest, dataset, audit = load_dataset_run(exp005_run)
    split_frames, split_details = select_time_slices(dataset, audit, cfg)
    feature_columns = [c for c in split_frames["train"] if c not in ("UserID", "cutoff_utc", *TARGETS)]
    # Data loading and the historical holdout protection are checked before model fit.
    overview = {
        "source_exp005_dir": str(exp005_run),
        "source_official_sha": exp005_manifest["source"].get("official_sha"),
        "rows_per_segment": {name: len(frame) for name, frame in split_frames.items()},
        "selected_cutoffs": split_details, "n_features": len(feature_columns),
        "models": model_names,
        "tuning_population": cfg["tuning_population"],
        "warning": "EXP-004 exact label reconstruction is not a predictive F1 score; holdout remains sealed",
    }
    if dry_run:
        return overview
    root = exp005_run.resolve().parents[3]  # personal project: .../experiments/exp005_temporal_dataset/runs/build-ID
    official_repo = root / "assignment2"
    official = load_official_evaluator(official_repo)
    if output_dir.is_file():
        raise ValueError(f"Output path is a file: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = split_xy(split_frames["train"])
    cal_x, cal_y = split_xy(split_frames["calibration"])
    report_x, report_y = split_xy(split_frames["primary"])
    baseline_th = np.full(len(TARGETS), 0.5, dtype=float)
    all_metrics: list[dict] = []
    all_targets: list[dict] = []
    timing = {}
    for kind in model_names:
        started = time.perf_counter()
        model_dir = output_dir / kind
        model_dir.mkdir(exist_ok=True)
        print(f"[EXP-006] Fitting {kind} on {len(train_x):,} user-weeks...", flush=True)
        bundle = fit_bundle(kind, train_x, train_y, cfg)
        fit_seconds = time.perf_counter() - started
        cal_prob = probabilities(bundle, cal_x)
        report_prob = probabilities(bundle, report_x)
        tune_mask = (cal_x.eligible_proxy.to_numpy() == 1 if cfg["tuning_population"] == "eligible_proxy"
                     else np.ones(len(cal_x), dtype=bool))
        if not np.count_nonzero(tune_mask):
            raise ValueError("Zero users in calibration tuning_population")
        if kind in ("logistic", "hgb"):
            tuned_th, tune_diag = tune_thresholds(
                cal_prob[tune_mask], cal_y.to_numpy()[tune_mask],
                grid=cfg["threshold_grid"], passes=int(cfg["threshold_passes"]),
            )
        else:
            tuned_th = baseline_th.copy()
            tune_diag = {"objective": "Fixed rule; thresholds are not tuned", "trace": []}
        bundle.thresholds = {col: float(th) for col, th in zip(TARGETS, tuned_th)}
        evaluation_checks = {}
        for stage, frame, prob in (
            ("calibration", split_frames["calibration"], cal_prob),
            ("primary_validation", split_frames["primary"], report_prob),
        ):
            for mode, th in (("default_0.5", baseline_th), ("tuned_calibration", tuned_th)):
                result = binary_with_thresholds(prob, th)
                metrics, target_stats, checks = _evaluate_slice(
                    official, frame, result, segment=stage, method=mode, model=kind,
                )
                all_metrics.extend(metrics)
                all_targets.extend(target_stats)
                evaluation_checks[f"{stage}/{mode}"] = checks
                if stage == "primary_validation" and mode == "tuned_calibration":
                    _output_predictions(model_dir / "primary_predictions.csv", frame, result)
        if kind not in ("zero", "persistence"):
            joblib.dump(bundle, model_dir / "model_bundle.joblib", compress=3)
        else:
            # Save trivial rules too so later inference uses a single uniform interface.
            joblib.dump(bundle, model_dir / "model_bundle.joblib", compress=3)
        (model_dir / "thresholds.json").write_text(
            json.dumps({"thresholds": bundle.thresholds, "calibration_details": tune_diag}, indent=2),
            encoding="utf-8",
        )
        total_seconds = time.perf_counter() - started
        timing[kind] = {"fit_seconds": round(fit_seconds, 2), "end_to_end_seconds": round(total_seconds, 2),
                        "constant_targets": bundle.constant_labels}
        scored = next(row for row in all_metrics if row["model"] == kind and
                      row["segment"] == "primary_validation" and
                      row["threshold_mode"] == "tuned_calibration" and
                      row["population"] == cfg["tuning_population"])
        print(f"[EXP-006] {kind}: primary {cfg['tuning_population']} "
              f"macro={scored['macro_f1']:.4f} micro={scored['micro_f1']:.4f} "
              f"combined={scored['score']:.4f}, fit={fit_seconds:.1f}s", flush=True)
        (model_dir / "official_week_checks.json").write_text(
            json.dumps(evaluation_checks, indent=2), encoding="utf-8"
        )
    summary = pd.DataFrame(all_metrics)
    details = pd.DataFrame(all_targets)
    summary.to_csv(output_dir / "summary.csv", index=False)
    details.to_csv(output_dir / "per_target_metrics.csv", index=False)
    primary = summary[(summary.segment == "primary_validation") &
                      (summary.threshold_mode == "tuned_calibration") &
                      (summary.population == cfg["tuning_population"])].copy()
    primary = primary.sort_values(["score", "model"], ascending=[False, True])
    primary.to_csv(output_dir / "primary_leaderboard.csv", index=False)
    files_sha = {file: _sha256(exp005_run / file) for file in
                 ("train_features.parquet", "train_labels.parquet",
                  "validation_features.parquet", "validation_labels.parquet", "manifest.json")}
    manifest = {
        "experiment": "EXP-006", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_run": str(exp005_run.resolve()),
        "input_official_sha": exp005_manifest["source"].get("official_sha"),
        "input_exp005_config": exp005_manifest["config"],
        "input_file_sha256": files_sha,
        "exp006_config": cfg, "models": model_names,
        "feature_columns": feature_columns,
        "split_protocol": split_details,
        "tuning_population": cfg["tuning_population"],
        "selected_model": primary.iloc[0].model if len(primary) else None,
        "metrics": primary.to_dict("records"), "timing_seconds": timing,
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__,
                    "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                    "platform": platform.platform()},
        "warnings": [
            "Primary validation is chronological, but repeated model selection on it makes it development validation, NOT untouched final holdout",
            "Sep21 and Sep28 windows overlap the Sep27 example that calibrated label semantics; excluded from primary validation",
            "Validation Aug24 is left as a maturity gap: calibration labels finish maturing before primary validation starts",
            "Eligible-proxy population is NOT the official Oct28 test roster",
            "Historical labels use a configurable seven-day maturation assumption, not guaranteed complete event discovery",
            "Classifier training uses only 21 historical training weeks; thresholds fit on earlier validation calibration weeks",
            "The official evaluator is used for unique-UserID weekly verification. Pooled scores sum TP/FP/FN before calculating F1",
            "HISTORICAL HOLDOUT LABELS ARE NEVER LOADED OR UNSEALED",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"output_dir": str(output_dir), "rows": overview["rows_per_segment"],
            "models": model_names, "selected_model_by_primary_eligible_score": manifest["selected_model"],
            "primary_leaderboard": primary[["model", "macro_f1", "micro_f1", "score", "rows", "weeks"]].to_dict("records")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp005-run", help="Exact EXP-005 build directory; otherwise latest valid build is used")
    parser.add_argument("--exp005-base", default="experiments/exp005_temporal_dataset/runs")
    parser.add_argument("--config", default="configs/baseline_comparison.json")
    parser.add_argument("--models", default="zero,persistence,logistic", help="Comma-separated: zero,persistence,logistic,hgb")
    parser.add_argument("--output-base", default="experiments/exp006_baseline_comparison/runs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate split/provenance without fitting models")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    names = [value.strip() for value in args.models.split(",") if value.strip()]
    input_path = find_run(args.exp005_base, explicit=args.exp005_run)
    out = Path(args.output_base) / ("run-" + input_path.name + "-" + hash_config(config, names))
    result = train_run(exp005_run=input_path, cfg=config, model_names=names,
                       output_dir=out, overwrite=args.overwrite, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, AssertionError) as exc:
        print(f"EXP-006 error: {exc}", file=sys.stderr)
        sys.exit(1)
