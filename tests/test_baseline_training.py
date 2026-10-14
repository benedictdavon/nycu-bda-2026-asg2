import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import TARGETS
from experiments.exp006_baseline_comparison.protocol import load_config, select_time_slices
from src.dataset_io import load_dataset_run
from src.baseline_models import fit_bundle, probabilities, feature_names
from experiments.exp006_baseline_comparison.train import train_run


def utc_tw(s):
    return pd.Timestamp(s, tz="Asia/Taipei").tz_convert("UTC")


@pytest.fixture
def miniature_run(synthetic):
    """Offline stand-in for real EXP-005 Parquet outputs. pytest shim uses pickle."""
    root = synthetic["root"]
    run = root / "experiments/exp005_temporal_dataset/runs/build-synthetic"
    run.mkdir(parents=True)
    train_dates = ["2026-07-06", "2026-07-13", "2026-07-20"]
    val_dates = ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24",
                 "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28"]
    dates = [(day, "train") for day in train_dates] + [(day, "validation") for day in val_dates]
    features, labels, audit = {}, {}, []
    for split in ("train", "validation"):
        features[split], labels[split] = [], []
    for day, split in dates:
        cutoff = utc_tw(day)
        for i, user in enumerate([101, 202, 303]):
            feature = {"UserID": user, "cutoff_utc": cutoff,
                       "eligible_proxy": int(i != 2),
                       "historical_observation_count_90d_proxy": int(i + 5)}
            label = {"UserID": user, "cutoff_utc": cutoff}
            week = len(audit)
            for j, name in enumerate(TARGETS):
                flag = int((i + j + week) % 3 == 0)
                feature[f"{name}_active_prior_7d"] = flag
                feature[f"{name}_count_7d"] = (i + week + j) % 10
                label[name] = int((i + j + week) % 2 == 0)
            features[split].append(feature)
            labels[split].append(label)
        audit.append({"split": split, "cutoff_utc": cutoff,
                      "label_end_utc": cutoff + pd.Timedelta(days=7),
                      "label_as_of_utc": cutoff + pd.Timedelta(days=14)})
    for split in ("train", "validation"):
        pd.DataFrame(features[split]).to_parquet(run / f"{split}_features.parquet", index=False)
        pd.DataFrame(labels[split]).to_parquet(run / f"{split}_labels.parquet", index=False)
    pd.DataFrame(audit + [{"split": "holdout", "cutoff_utc": utc_tw("2026-10-12"),
                          "label_end_utc": utc_tw("2026-10-19"),
                          "label_as_of_utc": utc_tw("2026-10-26")}]).to_csv(run / "label_audit.csv", index=False)
    manifest = {"experiment": "EXP-005", "mode": "build",
                "source": {"official_sha": "synthetic-snapshot"},
                "config": {"horizon_days": 7},
                "details": {"holdout_unsealed": False},
                "created_utc": "2026-10-14T01:00:00+00:00"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_safe_validation_periods_and_example_exclusion(miniature_run):
    cfg = load_config("configs/baseline_comparison.json")
    _, frames, audit = load_dataset_run(miniature_run)
    selected, info = select_time_slices(frames, audit, cfg)
    assert len(selected["train"]) == 9
    assert len(selected["calibration"]) == 9  # Aug3/10/17, 3 users
    assert len(selected["primary"]) == 9  # Aug31/Sep7/Sep14, 3 users
    assert len(info["excluded_validation_weeks"]) == 3  # Aug24 / Sep21 / Sep28
    assert pd.Timestamp(info["primary_first_cutoff"]).tz_convert("Asia/Taipei") == utc_tw("2026-08-31").tz_convert("Asia/Taipei")
    cfg["report_end_exclusive_taiwan"] = "2026-10-05"
    with pytest.raises(ValueError, match="overlaps Sep27"):
        select_time_slices(frames, audit, cfg)


def test_reject_label_maturity_leak(miniature_run):
    cfg = load_config("configs/baseline_comparison.json")
    _, frames, audit = load_dataset_run(miniature_run)
    cfg["report_start_taiwan"] = "2026-08-24"  # Aug17 labels mature Aug31
    with pytest.raises(ValueError, match="Time leakage"):
        select_time_slices(frames, audit, cfg)


def test_end_to_end_training_scoring_and_sealed_holdout(miniature_run, tmp_path):
    cfg = load_config("configs/baseline_comparison.json")
    out = tmp_path / "exp006-models"
    dry = train_run(exp005_run=miniature_run, cfg=cfg, model_names=["zero", "persistence", "logistic"],
                    output_dir=out, dry_run=True)
    assert dry["rows_per_segment"] == {"train": 9, "calibration": 9, "primary": 9}
    assert not out.exists()
    result = train_run(exp005_run=miniature_run, cfg=cfg,
                       model_names=["zero", "persistence", "logistic"], output_dir=out)
    assert set(result["models"]) == {"zero", "persistence", "logistic"}
    assert (out / "summary.csv").is_file()
    assert (out / "primary_leaderboard.csv").is_file()
    assert (out / "logistic/model_bundle.joblib").is_file()
    assert (out / "persistence/primary_predictions.csv").is_file()
    assert not (miniature_run / "holdout_labels.parquet").exists()
    recorded = pd.read_csv(out / "summary.csv")
    assert len(recorded) == 3 * 2 * 2 * 2
    assert recorded.score.between(0, 1).all()
    with pytest.raises(FileExistsError):
        train_run(exp005_run=miniature_run, cfg=cfg,
                  model_names=["zero", "persistence", "logistic"], output_dir=out)


def test_label_in_features_detected(miniature_run):
    _, frames, audit = load_dataset_run(miniature_run)
    train = frames["train"].copy()
    with pytest.raises(ValueError):
        feature_names(train)  # joined train frame contains six label fields


def test_holdout_unseal_rejected(miniature_run):
    (miniature_run / "holdout_labels.parquet").write_bytes(b"irrelevant")
    with pytest.raises(ValueError, match="unsealed"):
        load_dataset_run(miniature_run)


def test_optional_hgb_pipeline(miniature_run, tmp_path):
    cfg = load_config("configs/baseline_comparison.json")
    cfg["hgb_max_iter"] = 5
    cfg["hgb_min_samples_leaf"] = 2
    output = tmp_path / "hgb-test"
    result = train_run(exp005_run=miniature_run, cfg=cfg, model_names=["hgb"], output_dir=output)
    assert result["models"] == ["hgb"]
    assert (output / "hgb/model_bundle.joblib").is_file()
    assert pd.read_csv(output / "primary_leaderboard.csv").score.between(0, 1).all()


def test_final_schema_inference_and_snapshot_guard(miniature_run, synthetic, tmp_path):
    from experiments.exp006_baseline_comparison.predict import predict
    cfg = load_config("configs/baseline_comparison.json")
    model_output = tmp_path / "trained"
    train_run(exp005_run=miniature_run, cfg=cfg, model_names=["logistic"], output_dir=model_output)
    infer_run = tmp_path / "inference"
    infer_run.mkdir()
    _, frames, _ = load_dataset_run(miniature_run)
    feature_cols = [x for x in frames["train"] if x not in TARGETS]
    feats = frames["train"].loc[frames["train"].cutoff_utc == frames["train"].cutoff_utc.min(), feature_cols]
    feats = feats.set_index("UserID").loc[[202, 101]].reset_index()
    feats["cutoff_utc"] = pd.Timestamp("2026-11-11T16:00:00Z")
    feats.to_parquet(infer_run / "inference_features.parquet", index=False)
    (infer_run / "manifest.json").write_text(json.dumps({"experiment": "EXP-005", "mode": "inference",
        "source": {"official_sha": "synthetic-snapshot"}}))
    test_input = synthetic["root"] / "assignment2/evaluation/test_input.csv"
    pd.DataFrame({"UserID": [202, 101]}).to_csv(test_input, index=False)
    output = tmp_path / "predict.csv"
    result = predict(model_run=model_output, inference_run=infer_run, model="logistic",
                     test_input=test_input, output=output)
    csv = pd.read_csv(output)
    assert csv.columns.tolist() == ["UserID", *TARGETS]
    assert csv.UserID.tolist() == [202, 101]
    assert csv[list(TARGETS)].isin([0, 1]).all().all()
    assert result["users"] == 2
    (infer_run / "manifest.json").write_text(json.dumps({"experiment": "EXP-005", "mode": "inference",
        "source": {"official_sha": "later-snapshot"}}))
    with pytest.raises(ValueError, match="differs"):
        predict(model_run=model_output, inference_run=infer_run, model="logistic",
                test_input=test_input, output=output, overwrite=True)
