"""Rule baselines and smooth/blended rankings for EXP-020-021 (docs/exp/EXP020_021_PREREGISTRATION.md).

Every function reads only rows with seen_utc < cutoff. event_utc is used solely as an extra
filter for label-aligned windows; it never admits a row that was not yet visible.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .data_loader import TARGETS, utc
from .decision_policies import activity_flags, top_k_decisions


def aligned_flags(events: Mapping[str, pd.DataFrame], cutoff, user_ids, lookback_days: int = 7) -> np.ndarray:
    """1 if a row has event_utc in [C - L, C) and seen_utc < C (label-aligned persistence)."""
    t = utc(cutoff)
    lower = t - pd.Timedelta(days=lookback_days)
    ids = pd.Index(user_ids, dtype="int64")
    out = np.zeros((len(ids), len(TARGETS)), dtype=np.int8)
    for j, target in enumerate(TARGETS):
        f = events[target]
        ok = f["seen_utc"].lt(t) & f["event_utc"].ge(lower) & f["event_utc"].lt(t)
        out[:, j] = ids.isin(pd.Index(f.loc[ok, "UserID"].unique())).astype(np.int8)
    return out


def multi_week_flags(events, cutoff, user_ids, weeks: int = 4, min_weeks: int = 2) -> np.ndarray:
    """P_persist7 OR active (seen_utc) in at least min_weeks of the last `weeks` weekly windows."""
    t = utc(cutoff)
    weekly = [activity_flags(events, t - pd.Timedelta(days=7 * j), user_ids, 7) for j in range(weeks)]
    return (weekly[0].astype(bool) | (np.sum(weekly, axis=0) >= min_weeks)).astype(np.int8)


def ew_scores(events, cutoff, user_ids, tau_days: float, window_days: int = 90) -> np.ndarray:
    """Sum over visible rows in [C - window, C) of exp(-(C - seen) / tau), per user and target."""
    t = utc(cutoff)
    lower = t - pd.Timedelta(days=window_days)
    ids = pd.Index(user_ids, dtype="int64")
    out = np.zeros((len(ids), len(TARGETS)), dtype=np.float64)
    for j, target in enumerate(TARGETS):
        f = events[target]
        rows = f.loc[f["seen_utc"].ge(lower) & f["seen_utc"].lt(t), ["UserID", "seen_utc"]]
        age = (t - rows["seen_utc"]).dt.total_seconds().to_numpy() / 86400.0
        w = pd.Series(np.exp(-age / tau_days), index=rows["UserID"].to_numpy()).groupby(level=0).sum()
        out[:, j] = w.reindex(ids).fillna(0.0).to_numpy()
    return out


def rank01(x: np.ndarray) -> np.ndarray:
    """Column-wise average rank scaled to (0, 1]; higher x -> higher rank."""
    frame = pd.DataFrame(np.asarray(x, dtype=np.float64))
    return (frame.rank(method="average") / len(frame)).to_numpy()


def blend_score(p: np.ndarray, e: np.ndarray, w: float) -> np.ndarray:
    return w * rank01(p) + (1.0 - w) * rank01(e)


def topk_at_persist(score: np.ndarray, persist: np.ndarray, uids) -> np.ndarray:
    return top_k_decisions(score, uids, persist.sum(0).astype(np.int64))
