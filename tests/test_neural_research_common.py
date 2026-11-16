from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import TARGETS
from src.decision_policies import roster_population
from src.features import FeatureBuilder
from src.labels import LabelBuilder
from experiments.neural_research_common.data import (
    CacheContext, inner_split, label_cache_identity, report_train_origins,
)


class FakeSource:
    def __init__(self, users: pd.DataFrame, events: dict[str, pd.DataFrame]):
        self.cohort = users
        self.cohort_ids = pd.Index(users.UserID.astype("int64"), name="UserID")
        self._events = events

    def events(self, target: str) -> pd.DataFrame:
        return self._events[target]


def _empty_events(user_id=1):
    return {
        name: pd.DataFrame({
            "UserID": pd.Series(dtype="int64"), "repo_id": pd.Series(dtype="int64"),
            "seen_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "event_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        }) for name in TARGETS
    }


def test_purged_split_has_42_fit_7_purge_7_validation_and_56_outer_origins():
    S = pd.Timestamp("2026-09-09T16:00:00Z")
    fit, purged, validation, V = inner_split(S)
    outer = report_train_origins(S)
    assert len(fit) == 42
    assert len(purged) == 7
    assert len(validation) == 7
    assert len(outer) == 56
    assert not set(fit) & set(purged)
    assert not set(fit) & set(validation)
    assert not set(purged) & set(validation)
    assert all(s + pd.Timedelta(days=7) <= V for s in fit)
    assert outer[0] == S - pd.Timedelta(days=63)
    assert outer[-1] == S - pd.Timedelta(days=8)


def test_population_lower_boundary_included_anchor_boundary_excluded():
    anchor = pd.Timestamp("2026-08-01T00:00:00Z")
    lower = anchor - pd.Timedelta(days=90)
    stamps = [lower, lower + pd.Timedelta(seconds=1), anchor - pd.Timedelta(days=2),
              anchor - pd.Timedelta(days=1), anchor - pd.Timedelta(microseconds=1), anchor]
    events = _empty_events()
    events["open_pr"] = pd.DataFrame({
        "UserID": [1, 1, 1, 1, 1, 1], "seen_utc": stamps,
    })
    events["push"] = pd.DataFrame({
        "UserID": [2, 2, 2, 2], "seen_utc": [lower, lower + pd.Timedelta(days=1),
                                                 anchor - pd.Timedelta(days=1), anchor],
    })
    kept = roster_population(events, anchor, pd.Index([1, 2]), window_days=90, min_rows=5)
    assert kept.tolist() == [1]


def test_features_ignore_rows_at_or_after_cutoff_and_future_injection():
    cutoff = pd.Timestamp("2026-09-01T12:00:00Z")
    users = pd.DataFrame({"UserID": [1], "first_seen_utc": [cutoff - pd.Timedelta(days=180)]})
    events = _empty_events()
    events["open_pr"] = pd.DataFrame({
        "UserID": [1, 1, 1], "repo_id": [10, 11, 12],
        "seen_utc": [cutoff - pd.Timedelta(days=1), cutoff, cutoff + pd.Timedelta(seconds=1)],
        "event_utc": [cutoff - pd.Timedelta(days=1), cutoff, cutoff + pd.Timedelta(seconds=1)],
    })
    source = FakeSource(users, events)
    before = FeatureBuilder(source).build(cutoff)
    assert int(before.loc[0, "open_pr_count_7d"]) == 1
    assert int(before.loc[0, "open_pr_active_prior_7d"]) == 1
    source._events["open_pr"] = pd.concat([
        events["open_pr"], pd.DataFrame({
            "UserID": [1], "repo_id": [99],
            "seen_utc": [cutoff + pd.Timedelta(days=90)],
            "event_utc": [cutoff + pd.Timedelta(days=90)],
        })
    ], ignore_index=True)
    after = FeatureBuilder(source).build(cutoff)
    feature_cols = [c for c in before if c not in {"UserID", "cutoff_utc"}]
    pd.testing.assert_frame_equal(before[feature_cols], after[feature_cols])


def test_label_window_is_half_open_and_visibility_must_be_at_or_before_asof():
    start = pd.Timestamp("2026-09-01T16:00:00Z")
    end = start + pd.Timedelta(days=7)
    asof = end
    users = pd.DataFrame({"UserID": [1], "first_seen_utc": [start - pd.Timedelta(days=365)]})
    events = _empty_events()
    events["open_pr"] = pd.DataFrame({
        "UserID": [1, 1, 1], "repo_id": [1, 2, 3],
        "seen_utc": [asof, end - pd.Timedelta(seconds=1), asof + pd.Timedelta(seconds=1)],
        "event_utc": [start, end, start + pd.Timedelta(days=1)],
    })
    labels = LabelBuilder(FakeSource(users, events)).build(
        start, horizon_days=7, label_as_of=asof, user_ids=pd.Index([1]),
    )
    assert labels.loc[0, "open_pr"] == 1
    assert all(labels.loc[0, target] == 0 for target in TARGETS[1:])


def test_source_and_asof_are_part_of_authenticated_cache_key():
    ctx = CacheContext(Path("."), "source-a", {"Users": "a" * 64},
                       {"builder.py": "b" * 64}, {"windows": [7, 30, 90]})
    population = pd.Index([100, 200], dtype="int64")
    origin = pd.Timestamp("2026-09-09T16:00:00Z")
    base = label_cache_identity(origin, pd.Timestamp("2026-09-09T13:00:00Z"), population)
    key_a, _ = ctx.key("labels", base)
    asof = dict(base, label_as_of_utc="2026-09-09T14:00:00Z")
    key_b, _ = ctx.key("labels", asof)
    ctx_other = CacheContext(Path("."), "source-b", {"Users": "a" * 64},
                             {"builder.py": "b" * 64}, {"windows": [7, 30, 90]})
    key_c, _ = ctx_other.key("labels", base)
    assert len({key_a, key_b, key_c}) == 3
