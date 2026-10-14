"""Shared six-label count metrics and threshold tuning matching the instructor evaluator.

IMPORTANT: Instructor's official scores() joins by UserID. In a multiweek dataset
users repeat, so call it separately per cutoff, and pool *confusion counts* across
cutoffs. Never give a multiweek frame directly to official scores().
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd

from .data_loader import TARGETS


def confusion_arrays(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(prediction)
    y = np.asarray(truth)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != len(TARGETS):
        raise ValueError(f"Expected matching arrays shaped (N, {len(TARGETS)})")
    if not (np.isin(p, [0, 1]).all() and np.isin(y, [0, 1]).all()):
        raise ValueError("Non-binary prediction or truth")
    return {
        "tp": np.count_nonzero((p == 1) & (y == 1), axis=0).astype(np.int64),
        "fp": np.count_nonzero((p == 1) & (y == 0), axis=0).astype(np.int64),
        "fn": np.count_nonzero((p == 0) & (y == 1), axis=0).astype(np.int64),
        "tn": np.count_nonzero((p == 0) & (y == 0), axis=0).astype(np.int64),
    }


def _f1(tp, fp, fn) -> float:
    """Instructor convention: score=1 when tp+fp+fn == 0."""
    denominator = 2 * int(tp) + int(fp) + int(fn)
    return float(2 * int(tp) / denominator) if denominator else 1.0


def scores_from_counts(counts: Mapping[str, np.ndarray]) -> dict:
    tp, fp, fn = (np.asarray(counts[x], dtype=np.int64) for x in ("tp", "fp", "fn"))
    f1s = [_f1(a, b, c) for a, b, c in zip(tp, fp, fn)]
    macro = float(np.mean(f1s))
    micro = _f1(tp.sum(), fp.sum(), fn.sum())
    return {"macro_f1": macro, "micro_f1": micro, "score": (macro + micro) / 2}


def detailed_metrics(prediction: np.ndarray, truth: np.ndarray) -> tuple[dict, pd.DataFrame]:
    count = confusion_arrays(prediction, truth)
    summary = scores_from_counts(count)
    rows = []
    for i, name in enumerate(TARGETS):
        tp, fp, fn, tn = (int(count[k][i]) for k in ("tp", "fp", "fn", "tn"))
        rows.append({
            "target": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": tp + fn, "predicted_positives": tp + fp,
            "prevalence": (tp + fn) / len(truth) if len(truth) else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": _f1(tp, fp, fn),
        })
    return {**summary, "rows": int(len(truth))}, pd.DataFrame(rows)


def binary_with_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    th = np.asarray(thresholds, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(TARGETS) or th.shape != (len(TARGETS),):
        raise ValueError("Probability array or threshold dimensions incorrect")
    if not (np.isfinite(p).all() and np.isfinite(th).all()):
        raise ValueError("Non-finite probability/threshold")
    if (p < 0).any() or (p > 1).any() or (th < 0).any() or (th > 1).any():
        raise ValueError("Invalid probability/threshold range")
    return (p >= th[None, :]).astype(np.int8)


def tune_thresholds(probabilities: np.ndarray, truth: np.ndarray, *,
                    grid: list[float], passes: int = 3) -> tuple[np.ndarray, dict]:
    """Deterministic coordinate ascent on exact pooled instructor combined-F1.

    IMPORTANT: tune only on an earlier calibration time slice, NEVER on report slice.
    Coordinates update in TARGETS order, candidate ties prefer threshold near 0.5.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(truth, dtype=np.int8)
    if len(p) == 0 or p.shape != y.shape or p.shape[1] != len(TARGETS):
        raise ValueError("Expected nonempty matching calibration arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    g = sorted(set(float(x) for x in grid) | {0.5})
    if not g or min(g) <= 0 or max(g) > 1:
        raise ValueError("Threshold grid must be in (0,1]")
    if passes < 1:
        raise ValueError("passes must be >= 1")
    th = np.full(len(TARGETS), 0.5, dtype=np.float64)
    trace = []
    current = scores_from_counts(confusion_arrays(binary_with_thresholds(p, th), y))["score"]
    for iteration in range(passes):
        improved = False
        for j, name in enumerate(TARGETS):
            best = current
            best_th = float(th[j])
            for candidate in g:
                option = th.copy()
                option[j] = candidate
                sc = scores_from_counts(confusion_arrays(binary_with_thresholds(p, option), y))["score"]
                if sc > best + 1e-12 or (
                    abs(sc - best) <= 1e-12 and
                    abs(candidate - 0.5) < abs(best_th - 0.5) - 1e-12
                ):
                    best, best_th = sc, candidate
            improved |= abs(th[j] - best_th) > 1e-12
            th[j] = best_th
            current = best
        trace.append({"pass": iteration + 1, "combined_f1": float(current),
                      "thresholds": {k: float(v) for k, v in zip(TARGETS, th)}})
        if not improved:
            break
    return th, {"objective": "0.5 * (pooled macro-F1 + pooled micro-F1)",
                "calibration_score": float(current), "trace": trace}


def official_week_check(official, frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    """Verify our pooled formula by comparing EACH unique-user week with official scores().

    frame must contain columns [UserID, cutoff_utc, *TARGETS]. Each UserID occurs
    at most once per cutoff. The instructor evaluator is never given duplicate IDs.
    """
    if len(frame) != len(prediction):
        raise ValueError("Prediction length mismatch")
    if frame.duplicated(["UserID", "cutoff_utc"]).any():
        raise ValueError("Repeated user inside one cutoff")
    checks = []
    for cutoff, indices in frame.groupby("cutoff_utc", sort=True).indices.items():
        subset = frame.iloc[indices]
        if subset.UserID.duplicated().any():
            raise ValueError("Duplicate UserIDs in official evaluator week")
        truth = subset[["UserID", *TARGETS]].reset_index(drop=True)
        pred = pd.DataFrame(prediction[indices], columns=TARGETS)
        pred.insert(0, "UserID", truth.UserID.to_numpy())
        official_score = official.scores(pred, truth)
        ours, _ = detailed_metrics(pred[list(TARGETS)].to_numpy(), truth[list(TARGETS)].to_numpy())
        for key in ("macro_f1", "micro_f1", "score"):
            if not np.isclose(official_score[key], ours[key], atol=1e-10):
                raise AssertionError(f"Official evaluator disagrees on {cutoff}: {key}")
        checks.append({"cutoff_utc": str(cutoff), "users": len(truth),
                       **{k: float(official_score[k]) for k in ("macro_f1", "micro_f1", "score")}})
    return {"verified_weeks": len(checks), "weekly": checks,
            "pooling_rule": "Sum TP/FP/FN across user-weeks, THEN compute F1; not mean of week F1"}
