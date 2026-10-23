"""EXP-014 pure helpers: time grid, roster-anchored population, persistence flags,
Platt calibration, label-free expected-score top-k selection, and rate matching.

Everything here is a pure function of arrays or frames already loaded by the runner.
No file or network I/O happens in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import TARGETS, utc

N_TARGETS = len(TARGETS)
PROB_CLIP = 1e-6


@dataclass(frozen=True)
class Timing:
    """Clock rules. Every origin is a 16:00 UTC (Thursday 00:00 Taiwan) decision grid point."""
    horizon_days: int = 7
    feature_lag_hours: int = 4
    decision_lag_hours: int = 3
    train_window_days: int = 56
    calibration_lags_days: tuple[int, int] = (21, 14)


def feature_cutoff(origin, timing: Timing) -> pd.Timestamp:
    """Rows with seen_utc strictly before this instant may enter features."""
    return utc(origin) - pd.Timedelta(hours=timing.feature_lag_hours)


def decision_time(origin, timing: Timing) -> pd.Timestamp:
    """Labels used for any decision at this origin must have been seen at or before this."""
    return utc(origin) - pd.Timedelta(hours=timing.decision_lag_hours)


def calibration_origins(report_origin, timing: Timing) -> list[pd.Timestamp]:
    base = utc(report_origin)
    return [base - pd.Timedelta(days=lag) for lag in timing.calibration_lags_days]


def training_origins(report_origin, *, timing: Timing, limit=None,
                     window_days: int | None = None) -> list[pd.Timestamp]:
    """Daily origins s with s >= S - (window + horizon) and s + horizon <= limit.

    limit defaults to the decision time A = S - decision_lag. For the M_cal fit pass
    limit = decision_time(S - 21d), which gives s + 7d <= (S - 21d) - 3h.
    """
    base = utc(report_origin)
    window = timing.train_window_days if window_days is None else window_days
    limit_ts = decision_time(base, timing) if limit is None else utc(limit)
    first = base - pd.Timedelta(days=window + timing.horizon_days)
    horizon = pd.Timedelta(days=timing.horizon_days)
    origins = []
    current = first
    while current + horizon <= limit_ts:
        origins.append(current)
        current = current + pd.Timedelta(days=1)
    return origins


def population_anchor(origin, timing: Timing) -> pd.Timestamp:
    """anchor = origin - 14d - 4h, the roster-anchored proxy reference instant."""
    return feature_cutoff(origin, timing) - pd.Timedelta(days=14)


def roster_population(events: Mapping[str, pd.DataFrame], anchor, cohort_ids, *,
                      window_days: int = 90, min_rows: int = 5) -> pd.Index:
    """Cohort users with >= min_rows visible rows in [anchor - window, anchor).

    Visibility is seen_utc, summed over every target table. A row exactly at the
    anchor is excluded; a row strictly before it is included.
    """
    anchor_ts = utc(anchor)
    lower = anchor_ts - pd.Timedelta(days=window_days)
    pieces = []
    for frame in events.values():
        seen = frame["seen_utc"]
        pieces.append(frame.loc[seen.ge(lower) & seen.lt(anchor_ts), "UserID"])
    cohort = pd.Index(cohort_ids)
    if not pieces:
        return pd.Index([], dtype="int64", name="UserID")
    rows = pd.concat(pieces, ignore_index=True)
    counts = rows.value_counts()
    eligible = counts.index[counts.to_numpy() >= min_rows]
    kept = np.sort(np.asarray(eligible.intersection(cohort), dtype=np.int64))
    return pd.Index(kept, dtype="int64", name="UserID")


def activity_flags(events: Mapping[str, pd.DataFrame], cutoff, user_ids,
                   lookback_days: int) -> np.ndarray:
    """(n, 6) int8: 1 if the user has a row seen in [cutoff - L days, cutoff)."""
    t = utc(cutoff)
    lower = t - pd.Timedelta(days=lookback_days)
    ids = pd.Index(user_ids, dtype="int64")
    out = np.zeros((len(ids), N_TARGETS), dtype=np.int8)
    for j, target in enumerate(TARGETS):
        frame = events[target]
        active = frame.loc[frame["seen_utc"].ge(lower) & frame["seen_utc"].lt(t), "UserID"]
        out[:, j] = ids.isin(pd.Index(active.unique())).astype(np.int8)
    return out


def f1_from_counts(tp, fp, fn) -> np.ndarray:
    """Instructor convention: F1 = 1 when TP + FP + FN = 0."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denom = 2 * tp + fp + fn
    out = np.ones_like(denom)
    np.divide(2 * tp, denom, out=out, where=denom > 0)
    return out


def choose_persistence_lookbacks(flags_by_lookback: Mapping[int, np.ndarray],
                                 truth: np.ndarray, lookbacks=(7, 14, 21, 28)) -> dict[str, int]:
    """Per target, the lookback maximizing pooled F1 over the stacked calibration rows.

    Ties go to the smaller lookback because candidates are scanned ascending and only a
    strictly better F1 replaces the incumbent.
    """
    y = np.asarray(truth)
    chosen: dict[str, int] = {}
    for j, target in enumerate(TARGETS):
        best_l, best_f1 = None, -np.inf
        for lookback in sorted(lookbacks):
            pred = np.asarray(flags_by_lookback[lookback])[:, j]
            tp = int(np.count_nonzero((pred == 1) & (y[:, j] == 1)))
            fp = int(np.count_nonzero((pred == 1) & (y[:, j] == 0)))
            fn = int(np.count_nonzero((pred == 0) & (y[:, j] == 1)))
            f1 = float(f1_from_counts(tp, fp, fn))
            if f1 > best_f1 + 1e-15:
                best_l, best_f1 = lookback, f1
        chosen[target] = int(best_l)
    return chosen


def persistence_decisions(flags_by_lookback: Mapping[int, np.ndarray],
                          lookbacks_by_target: Mapping[str, int]) -> np.ndarray:
    out = np.zeros_like(np.asarray(flags_by_lookback[next(iter(flags_by_lookback))]), dtype=np.int8)
    for j, target in enumerate(TARGETS):
        out[:, j] = np.asarray(flags_by_lookback[int(lookbacks_by_target[target])])[:, j]
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), PROB_CLIP, 1.0 - PROB_CLIP)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(p, y, *, ridge: float = 1e-8, max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float]:
    """One-dimensional logistic regression of y on logit(p). Returns (slope a, intercept b).

    Solved by Newton-Raphson with a tiny ridge on (a, b), so the fit is deterministic and
    does not depend on a native optimizer backend. Degenerate slices (one label class, or a
    constant score) cannot identify a slope; for those the function returns a = 0 and
    b = logit(prevalence), which is the calibrated constant.
    """
    x = _logit(p)
    labels = np.asarray(y, dtype=np.int8)
    if labels.size == 0:
        raise ValueError("Platt fit needs at least one calibration row")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Platt labels must be binary")
    if len(np.unique(labels)) < 2 or np.ptp(x) < 1e-12:
        prevalence = float(np.clip(labels.mean(), PROB_CLIP, 1.0 - PROB_CLIP))
        return 0.0, float(np.log(prevalence / (1.0 - prevalence)))
    design = np.column_stack([x, np.ones_like(x)])
    yy = labels.astype(np.float64)
    theta = np.zeros(2)
    for _ in range(max_iter):
        prob = _sigmoid(design @ theta)
        gradient = design.T @ (yy - prob) - ridge * theta
        weight = prob * (1.0 - prob)
        hessian = (design * weight[:, None]).T @ design + ridge * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        theta = theta + step
        if np.max(np.abs(step)) < tol:
            break
    return float(theta[0]), float(theta[1])


def apply_platt(p, a: float, b: float) -> np.ndarray:
    return _sigmoid(float(a) * _logit(p) + float(b))


def rate_match_k(q: np.ndarray) -> np.ndarray:
    """k_t = round(sum_u q_ut), clipped to [0, n]. Python round() semantics (ties to even)."""
    q = np.asarray(q, dtype=np.float64)
    n = q.shape[0]
    return np.array([min(n, max(0, int(round(float(s))))) for s in q.sum(axis=0)], dtype=np.int64)


def top_k_decisions(q: np.ndarray, user_ids, k) -> np.ndarray:
    """(n, T) int8: per target, the k_t users with the largest q are set to 1.

    Equal q values break ties by UserID ascending, so the selection is deterministic.
    """
    q = np.asarray(q, dtype=np.float64)
    ids = np.asarray(user_ids, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    out = np.zeros(q.shape, dtype=np.int8)
    for j in range(q.shape[1]):
        order = np.lexsort((ids, -q[:, j]))
        out[order[: int(k[j])], j] = 1
    return out


def _expected_f1_vector(etp: np.ndarray, total_q: float) -> np.ndarray:
    """E[F1_t(k)] for k = 0..n, using 2*ETP/(k + P) and the F1 = 1 convention at 0/0."""
    n = len(etp) - 1
    ks = np.arange(n + 1, dtype=np.float64)
    denom = ks + total_q
    out = np.zeros(n + 1, dtype=np.float64)
    np.divide(2.0 * etp, denom, out=out, where=denom > 0)
    if total_q == 0.0:
        out[0] = 1.0
    return out


def optimize_expected_k(q: np.ndarray, *, max_passes: int = 20) -> dict:
    """Label-free k_t maximizing the expected official score under independent Bernoulli(q).

    Objective = 0.5 * (mean_t E[F1_t(k_t)] + E[micro]), with
    E[F1_t(k)] = 2 ETP_t(k) / (k + P_t) and E[micro] = 2 sum ETP_t(k_t) / (sum k_t + sum P_t).
    Start at each target's own argmax, then sweep coordinates with exact 1-D argmax over
    0..n given the others, until a full pass changes nothing (or max_passes).
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0:
        raise ValueError("Expected a nonempty (n, T) probability matrix")
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("q must be finite probabilities in [0, 1]")
    n, t_count = q.shape
    etp = []
    ef1 = []
    totals = q.sum(axis=0)
    for j in range(t_count):
        ordered = np.sort(q[:, j])[::-1]
        prefix = np.concatenate([[0.0], np.cumsum(ordered)])
        etp.append(prefix)
        ef1.append(_expected_f1_vector(prefix, float(totals[j])))
    grand_total = float(totals.sum())
    ks = np.arange(n + 1, dtype=np.float64)

    def micro_vector(etp_sum, k_sum):
        denom = k_sum + grand_total
        out = np.ones_like(denom)
        np.divide(2.0 * etp_sum, denom, out=out, where=denom > 0)
        return out

    current = np.array([int(np.argmax(f)) for f in ef1], dtype=np.int64)
    passes = 0
    for passes in range(1, max_passes + 1):
        changed = False
        for j in range(t_count):
            others = [s for s in range(t_count) if s != j]
            other_etp = sum((etp[s][current[s]] for s in others), 0.0)
            other_k = float(sum(current[s] for s in others))
            other_f1 = sum((ef1[s][current[s]] for s in others), 0.0)
            micro = micro_vector(other_etp + etp[j], other_k + ks)
            objective = 0.5 * ((other_f1 + ef1[j]) / t_count + micro)
            best = int(np.argmax(objective))
            # Keep the incumbent on numerical ties so that "no change" means no change.
            if objective[current[j]] >= objective[best] - 1e-12:
                continue
            current[j] = best
            changed = True
        if not changed:
            break
    final_micro = micro_vector(
        sum((etp[s][current[s]] for s in range(t_count)), 0.0), float(current.sum()))
    objective_value = 0.5 * (
        sum(ef1[s][current[s]] for s in range(t_count)) / t_count + float(final_micro)
    )
    return {"k": current, "objective": float(objective_value), "passes": int(passes)}


def expected_objective(q: np.ndarray, k) -> float:
    """Objective value for a fixed k vector, using the same expected-score formulas."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.int64)
    n, t_count = q.shape
    f1_terms = []
    etp_terms = []
    for j in range(t_count):
        ordered = np.sort(q[:, j])[::-1]
        etp_j = float(ordered[: int(k[j])].sum())
        p_j = float(q[:, j].sum())
        if k[j] == 0 and p_j == 0.0:
            f1 = 1.0
        else:
            f1 = 2.0 * etp_j / (int(k[j]) + p_j)
        f1_terms.append(f1)
        etp_terms.append(etp_j)
    total_k = int(k.sum())
    grand = float(q.sum())
    micro = 1.0 if total_k + grand == 0.0 else 2.0 * sum(etp_terms) / (total_k + grand)
    return 0.5 * (sum(f1_terms) / t_count + micro)
