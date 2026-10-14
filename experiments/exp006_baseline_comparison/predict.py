"""Produce FINAL-SCHEMA six-label CSV from an EXP-006 model and EXP-005 inference features.

Only run when the actual official test_input.csv and final snapshot are available.
This module NEVER creates or guesses target labels from November ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from src.data_loader import TARGETS, load_official_evaluator
from src.baseline_models import probabilities
from src.metrics import binary_with_thresholds
from src.paths import resolve_run_path
from src.model_io import load_model_bundle


def predict(*, model_run: Path, inference_run: Path, model: str, test_input: Path,
            output: Path, allow_new_snapshot: bool = False,
            overwrite: bool = False) -> dict:
    model_run, inference_run = resolve_run_path(model_run), resolve_run_path(inference_run)
    trained = json.loads((model_run / "manifest.json").read_text(encoding="utf-8"))
    inference = json.loads((inference_run / "manifest.json").read_text(encoding="utf-8"))
    if trained.get("experiment") != "EXP-006" or inference.get("experiment") != "EXP-005" or inference.get("mode") != "inference":
        raise ValueError("Expected EXP-006 trained model manifest and EXP-005 inference manifest")
    old_sha = trained["input_official_sha"]
    new_sha = inference["source"].get("official_sha")
    if old_sha != new_sha and not allow_new_snapshot:
        raise ValueError(
            f"Training official SHA {old_sha} differs from inference SHA {new_sha}; "
            "pass --allow-new-snapshot if using a later authorized instructor release"
        )
    bundle_path = model_run / model / "model_bundle.joblib"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction file: {output}")
    # WARNING: joblib can execute pickle code. Load only YOUR OWN trusted models.
    bundle = load_model_bundle(bundle_path)
    feats = pd.read_parquet(inference_run / "inference_features.parquet")
    if any(t in feats for t in TARGETS):
        raise ValueError("Inference input unexpectedly contains target label columns")
    if not feats.UserID.is_unique or feats.UserID.isna().any():
        raise ValueError("Duplicate/null inference UserID")
    if feats.cutoff_utc.nunique() != 1:
        raise ValueError("Inference features must contain exactly one prediction cutoff")
    expected = pd.read_csv(test_input)
    if list(expected.columns) != ["UserID"] or expected.UserID.isna().any() or expected.UserID.duplicated().any():
        raise ValueError("Official test_input.csv must contain unique nonnull UserID values")
    if len(feats) != len(expected) or set(feats.UserID) != set(expected.UserID):
        raise ValueError("Inference features do not match EXACT official test user IDs")
    expected_columns = set(trained["feature_columns"])
    available_columns = set(feats.columns) - {"UserID", "cutoff_utc"}
    if expected_columns != available_columns:
        raise ValueError(f"Feature schema drift: added={sorted(available_columns-expected_columns)}, "
                         f"missing={sorted(expected_columns-available_columns)}")
    # Exact assignment: authoritatively preserve the official CSV user ordering.
    feats = feats.set_index("UserID").loc[expected.UserID].reset_index()
    probability = probabilities(bundle, feats)
    thresholds = [bundle.thresholds[t] for t in TARGETS]
    predicted = binary_with_thresholds(probability, thresholds)
    result = pd.DataFrame(predicted, columns=TARGETS)
    result.insert(0, "UserID", expected.UserID.to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    # Use instructor's reader for schema validation when available.
    official_eval = test_input.resolve().parent / "evaluate.py"
    if official_eval.is_file():
        official = load_official_evaluator(test_input.resolve().parent.parent)
        check = official.read_predictions(str(output))
        if len(check) != len(expected) or not check.UserID.equals(expected.UserID):
            raise AssertionError("Final prediction CSV does not match the test roster")
    return {"output": str(output.resolve()), "model": model, "users": len(result),
            "target_columns": list(TARGETS), "trained_sha": old_sha, "inference_sha": new_sha,
            "cutoff_utc": str(pd.to_datetime(feats.cutoff_utc.iloc[0], utc=True)),
            "snapshot_changed": old_sha != new_sha}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-run", required=True)
    p.add_argument("--inference-run", required=True)
    p.add_argument("--model", default="logistic")
    p.add_argument("--test-input", default="assignment2/evaluation/test_input.csv")
    p.add_argument("--output", default="outputs/baseline_predictions.csv")
    p.add_argument("--allow-new-snapshot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    output = predict(model_run=Path(a.model_run), inference_run=Path(a.inference_run),
                     model=a.model, test_input=Path(a.test_input), output=Path(a.output),
                     allow_new_snapshot=a.allow_new_snapshot, overwrite=a.overwrite)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, AssertionError) as exc:
        print(f"EXP-006 inference error: {exc}", file=sys.stderr)
        sys.exit(1)
