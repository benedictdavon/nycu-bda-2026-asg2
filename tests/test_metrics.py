import numpy as np
import pandas as pd
import pytest

from src.data_loader import TARGETS
from src.metrics import (
    binary_with_thresholds, confusion_arrays, detailed_metrics,
    official_week_check, scores_from_counts, tune_thresholds,
)


def example(n=12):
    y = np.zeros((n, 6), dtype=np.int8)
    y[::3, 0] = 1
    y[::2, 5] = 1
    p = np.full((n, 6), 0.1)
    p[:, 0] = np.where(y[:, 0], 0.8, 0.2)
    p[:, 5] = np.where(y[:, 5], 0.9, 0.1)
    return p, y


def test_empty_positive_official_convention():
    pred = np.zeros((10, 6), dtype=np.int8)
    truth = np.zeros((10, 6), dtype=np.int8)
    result, details = detailed_metrics(pred, truth)
    assert result["score"] == 1.0
    assert details.f1.eq(1).all()
    truth[0, 0] = 1
    result, details = detailed_metrics(pred, truth)
    assert details.loc[details.target == "open_pr", "f1"].iloc[0] == 0.0
    assert result["macro_f1"] < 1.0


def test_threshold_tuning_improves_or_ties_calibration_objective():
    probs, truth = example()
    optimized, info = tune_thresholds(probs, truth, grid=[0.1, 0.2, 0.4, 0.5, 0.75, 0.9, 1], passes=3)
    initial = scores_from_counts(confusion_arrays(binary_with_thresholds(probs, np.full(6, .5)), truth))["score"]
    final = scores_from_counts(confusion_arrays(binary_with_thresholds(probs, optimized), truth))["score"]
    assert final >= initial
    assert len(info["trace"]) >= 1
    assert 0 < optimized.min() <= optimized.max() <= 1


def test_threshold_inputs_rejected():
    p, truth = example()
    with pytest.raises(ValueError):
        binary_with_thresholds(p, np.ones(5))
    with pytest.raises(ValueError):
        tune_thresholds(p, truth, grid=[0], passes=1)
    with pytest.raises(ValueError):
        tune_thresholds(p, truth, grid=[0.5], passes=0)


def test_weekly_official_scoring_preserves_unique_ids(synthetic):
    source = synthetic["source"]
    official = __import__("src.data_loader", fromlist=["load_official_evaluator"]).load_official_evaluator(source.official_repo)
    users = [101, 202]
    frame = pd.DataFrame({"UserID": users * 2,
                          "cutoff_utc": [pd.Timestamp("2026-08-03T00:00Z")] * 2 +
                                        [pd.Timestamp("2026-08-10T00:00Z")] * 2})
    for c in TARGETS:
        frame[c] = 0
    pred = np.zeros((4, 6), dtype=np.int8)
    result = official_week_check(official, frame, pred)
    assert result["verified_weeks"] == 2
    assert result["weekly"][0]["score"] == 1.0
    frame.loc[1, "UserID"] = 101
    with pytest.raises(ValueError):
        official_week_check(official, frame, pred)
