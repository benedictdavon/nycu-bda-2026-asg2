"""Unit tests for EXP-014 decision policies. Synthetic inputs only; no official data."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.data_loader import TARGETS
from src.decision_policies import (
    Timing, activity_flags, apply_platt, calibration_origins, choose_persistence_lookbacks,
    decision_time, expected_objective, feature_cutoff, fit_platt, optimize_expected_k,
    population_anchor, rate_match_k, roster_population, top_k_decisions, training_origins,
)


def _brute_force_best(q: np.ndarray) -> float:
    """Exhaustive max of the same objective over every k vector in {0..n}^T (vectorized)."""
    n, t_count = q.shape
    grids = np.stack(np.meshgrid(*[np.arange(n + 1)] * t_count, indexing="ij"), axis=-1)
    ks = grids.reshape(-1, t_count)
    grand = float(q.sum())
    f1_cols = []
    etp_cols = []
    for j in range(t_count):
        sorted_q = np.concatenate([[0.0], np.cumsum(np.sort(q[:, j])[::-1])])
        etp = sorted_q[ks[:, j]]
        p_j = float(q[:, j].sum())
        denom = ks[:, j] + p_j
        f1 = np.where(denom > 0, 2.0 * etp / np.where(denom > 0, denom, 1.0), 0.0)
        if p_j == 0.0:
            f1 = np.where(ks[:, j] == 0, 1.0, f1)
        f1_cols.append(f1)
        etp_cols.append(etp)
    macro = np.mean(np.stack(f1_cols, axis=1), axis=1)
    etp_total = np.sum(np.stack(etp_cols, axis=1), axis=1)
    k_total = ks.sum(axis=1).astype(np.float64)
    denom = k_total + grand
    micro = np.where(denom > 0, 2.0 * etp_total / np.where(denom > 0, denom, 1.0), 1.0)
    return float((0.5 * (macro + micro)).max())


@pytest.mark.parametrize("seed", range(12))
def test_coordinate_ascent_is_near_brute_force_optimum(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 9))
    q = rng.uniform(0.0, 1.0, size=(n, len(TARGETS))) ** int(rng.integers(1, 5))
    result = optimize_expected_k(q)
    best = _brute_force_best(q)
    assert result["objective"] <= best + 1e-12
    assert result["objective"] >= 0.98 * best
    assert np.isclose(expected_objective(q, result["k"]), result["objective"], atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_single_target_matches_brute_force_exactly(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(1, 9))
    q = rng.uniform(0.0, 1.0, size=(n, 1))
    result = optimize_expected_k(q)
    assert abs(result["objective"] - _brute_force_best(q)) <= 1e-12


def test_all_zero_q_selects_nothing_and_scores_one():
    q = np.zeros((7, len(TARGETS)))
    result = optimize_expected_k(q)
    assert result["k"].tolist() == [0] * len(TARGETS)
    assert result["objective"] == pytest.approx(1.0)


def test_all_one_q_selects_everyone_and_scores_one():
    n = 5
    q = np.ones((n, len(TARGETS)))
    result = optimize_expected_k(q)
    assert result["k"].tolist() == [n] * len(TARGETS)
    assert result["objective"] == pytest.approx(1.0)


def test_tied_q_values_still_produce_a_valid_k_vector():
    q = np.full((6, len(TARGETS)), 0.3)
    result = optimize_expected_k(q)
    assert result["k"].min() >= 0 and result["k"].max() <= 6
    best = _brute_force_best(q)
    assert result["objective"] >= 0.98 * best


def test_top_k_ties_break_by_user_id_ascending():
    q = np.full((5, 1), 0.4)
    user_ids = np.array([50, 10, 40, 20, 30])
    decisions = top_k_decisions(q, user_ids, [2])
    chosen = user_ids[decisions[:, 0] == 1]
    assert sorted(chosen.tolist()) == [10, 20]


def test_rate_match_is_round_of_sum():
    q = np.array([[0.2, 0.9], [0.3, 0.4], [0.4, 0.6]])
    assert rate_match_k(q).tolist() == [round(0.9), round(1.9)]
    q2 = np.array([[0.6, 0.0], [0.6, 0.0]])
    assert rate_match_k(q2).tolist() == [round(1.2), 0]


def test_rate_match_clips_to_population_size():
    q = np.ones((3, 1)) * 0.9
    assert rate_match_k(q).tolist() == [3]


def test_platt_is_monotone_for_fitted_slope():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.01, 0.6, size=5000)
    y = (rng.uniform(size=5000) < p).astype(np.int8)
    a, b = fit_platt(p, y)
    assert a > 0
    grid = np.linspace(0.001, 0.999, 300)
    out = apply_platt(grid, a, b)
    assert np.all(np.diff(out) >= 0)


def test_platt_recovers_known_logit_shift():
    rng = np.random.default_rng(11)
    n = 200_000
    z = rng.normal(-3.0, 1.5, size=n)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(z + 0.7)))).astype(np.int8)
    a, b = fit_platt(p, y)
    assert a == pytest.approx(1.0, abs=0.05)
    assert b == pytest.approx(0.7, abs=0.05)


def test_platt_degenerate_slice_returns_constant_prevalence():
    p = np.array([0.1, 0.4, 0.7, 0.9])
    a, b = fit_platt(p, np.zeros(4, dtype=np.int8))
    assert a == 0.0
    assert apply_platt(p, a, b) == pytest.approx(np.full(4, 1e-6), rel=1e-6)
    a2, b2 = fit_platt(np.full(4, 0.25), np.array([1, 0, 0, 0]))
    assert a2 == 0.0
    assert apply_platt(np.array([0.25]), a2, b2)[0] == pytest.approx(0.25, abs=1e-9)


def test_population_excludes_row_at_anchor_and_includes_row_just_before():
    anchor = pd.Timestamp("2026-09-01T12:00:00Z")
    before = anchor - pd.Timedelta(seconds=1)

    def frame(rows):
        return pd.DataFrame({
            "UserID": [u for u, _ in rows],
            "seen_utc": [s for _, s in rows],
        })

    # User 1: five rows strictly before the anchor -> included.
    # User 2: four rows before plus one exactly at the anchor -> excluded.
    # User 3: five rows, but one is 91 days before the anchor -> only four in window, excluded.
    rows = [(1, before - pd.Timedelta(days=d)) for d in range(5)]
    rows += [(2, before - pd.Timedelta(days=d)) for d in range(4)] + [(2, anchor)]
    rows += [(3, before - pd.Timedelta(days=d)) for d in range(4)]
    rows += [(3, anchor - pd.Timedelta(days=91))]
    events = {"open_pr": frame(rows)}
    got = roster_population(events, anchor, cohort_ids=[1, 2, 3])
    assert got.tolist() == [1]


def test_population_restricts_to_cohort_and_sums_across_targets():
    anchor = pd.Timestamp("2026-09-01T12:00:00Z")
    t = anchor - pd.Timedelta(hours=1)
    a = pd.DataFrame({"UserID": [7, 7, 7], "seen_utc": [t, t, t]})
    b = pd.DataFrame({"UserID": [7, 7, 8, 8, 8, 8, 8], "seen_utc": [t] * 7})
    got = roster_population({"open_pr": a, "push": b}, anchor, cohort_ids=[7])
    assert got.tolist() == [7]


def test_activity_flags_use_seen_time_and_strict_cutoff():
    cutoff = pd.Timestamp("2026-09-10T00:00:00Z")
    frames = {t: pd.DataFrame({"UserID": [], "seen_utc": pd.to_datetime([])}) for t in TARGETS}
    frames["push"] = pd.DataFrame({
        "UserID": [1, 2, 3],
        "seen_utc": [cutoff - pd.Timedelta(days=6), cutoff, cutoff - pd.Timedelta(days=8)],
    })
    for t in TARGETS:
        frames[t]["seen_utc"] = pd.to_datetime(frames[t]["seen_utc"], utc=True)
    flags = activity_flags(frames, cutoff, [1, 2, 3], lookback_days=7)
    push_col = list(TARGETS).index("push")
    assert flags[:, push_col].tolist() == [1, 0, 0]


def test_persistence_lookback_ties_prefer_smaller_lookback():
    truth = np.zeros((4, len(TARGETS)), dtype=np.int8)
    truth[0, :] = 1
    flags = {
        7: np.ones((4, len(TARGETS)), dtype=np.int8),
        14: np.ones((4, len(TARGETS)), dtype=np.int8),
        21: np.zeros((4, len(TARGETS)), dtype=np.int8),
        28: np.zeros((4, len(TARGETS)), dtype=np.int8),
    }
    chosen = choose_persistence_lookbacks(flags, truth)
    assert all(v == 7 for v in chosen.values())


def test_training_origins_respect_label_horizon_for_final_model():
    timing = Timing()
    for report in ["2026-09-02T16:00:00Z", "2026-09-30T16:00:00Z"]:
        decision = decision_time(report, timing)
        origins = training_origins(report, timing=timing)
        assert origins, "final training window must not be empty"
        for s in origins:
            assert s + pd.Timedelta(days=7) <= decision
            assert s.hour == 16 and s.minute == 0


def test_no_leakage_calibration_training_origins_end_before_calibration_decision_time():
    timing = Timing()
    report = pd.Timestamp("2026-09-30T16:00:00Z")
    cal_limit = decision_time(report - pd.Timedelta(days=21), timing)
    cal_train = training_origins(report, timing=timing, limit=cal_limit)
    assert cal_train, "calibration training window must not be empty"
    for s in cal_train:
        assert s + pd.Timedelta(days=7) <= report - pd.Timedelta(days=21) - pd.Timedelta(hours=3)
    final_train = training_origins(report, timing=timing)
    assert set(cal_train) <= set(final_train)


def test_calibration_origin_labels_close_before_decision_time():
    timing = Timing()
    report = pd.Timestamp("2026-09-16T16:00:00Z")
    decision = decision_time(report, timing)
    for origin in calibration_origins(report, timing):
        assert origin + pd.Timedelta(days=7) <= decision


def test_population_anchor_and_feature_cutoff_are_consistent():
    timing = Timing()
    origin = pd.Timestamp("2026-09-16T16:00:00Z")
    assert feature_cutoff(origin, timing) == pd.Timestamp("2026-09-16T12:00:00Z")
    assert population_anchor(origin, timing) < feature_cutoff(origin, timing)
    assert population_anchor(origin, timing) == pd.Timestamp("2026-09-02T12:00:00Z")
