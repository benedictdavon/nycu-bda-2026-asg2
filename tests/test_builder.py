import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset_builder import DEFAULTS, build_historical, build_inference, load_config


def test_full_pipeline_smoke_and_holdout_safety(synthetic, tmp_path):
    src = synthetic["source"]
    cfg = dict(DEFAULTS)
    cfg.update({
        "first_cutoff_taiwan": "2026-09-07",
        "validation_start_taiwan": "2026-09-14",
        "holdout_start_taiwan": "2026-09-21",
        "label_maturation_days": 0,
    })
    # Snapshot's latest recorded observed row is Oct 10; weekly cutoffs >= Oct 12 unavailable.
    out = tmp_path / "derived"
    stats = build_historical(src, cfg, output_dir=out)
    assert (out / "manifest.json").is_file()
    assert (out / "train_features.parquet").is_file()
    assert (out / "validation_labels.parquet").is_file()
    assert (out / "holdout_features.parquet").is_file()
    assert not (out / "holdout_labels.parquet").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["details"]["holdout_unsealed"] is False
    tr = pd.read_pickle(out / "train_features.parquet")
    y = pd.read_pickle(out / "train_labels.parquet")
    assert tr[["UserID", "cutoff_utc"]].equals(y[["UserID", "cutoff_utc"]])
    assert not any(col in tr.columns for col in ("open_pr", "open_issue", "push"))
    with pytest.raises(FileExistsError):
        build_historical(src, cfg, output_dir=out)
    stats2 = build_historical(src, cfg, output_dir=out, overwrite=True)
    tr2 = pd.read_pickle(out / "train_features.parquet")
    pd.testing.assert_frame_equal(tr, tr2)
    assert stats2["weeks"] == stats["weeks"]


def test_inference_uses_exact_test_ids_and_no_future_labels(synthetic, tmp_path):
    src = synthetic["source"]
    ids = src.official_repo / "evaluation/test_input.csv"
    pd.DataFrame({"UserID": [202, 101]}).to_csv(ids, index=False)
    out = tmp_path / "inference"
    result = build_inference(src, DEFAULTS, output_dir=out,
                             cutoff="2026-11-11T16:00:00Z")
    feat = pd.read_pickle(out / "inference_features.parquet")
    assert feat.UserID.tolist() == [202, 101]
    assert result["users"] == 2
    assert not (out / "inference_labels.parquet").exists()
    assert feat.filter(regex="^y_").empty


def test_config_rejects_unknown(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"bad_setting":true}')
    with pytest.raises(ValueError):
        load_config(path)
