import numpy as np
import pandas as pd

from src.features import FeatureBuilder


def test_features_use_only_prior_seen_time(synthetic):
    src = synthetic["source"]
    t = pd.Timestamp("2026-09-29T00:00:00Z")
    builder = FeatureBuilder(src, windows=(7,30,90))
    before = builder.build(t).set_index("UserID")
    # Push by 101 is observed at Sep 27, at this cutoff it is historical.
    assert before.loc[101, "push_count_7d"] == 1
    # PR created in prediction week but first seen October 7: NEVER historical.
    assert before.loc[101, "open_pr_count_7d"] == 0
    # Issue created Sep 28 but first seen Oct 9: NEVER historical.
    assert before.loc[101, "open_issue_count_7d"] == 0
    # Comment exactly at feature cutoff is excluded (< t).
    assert before.loc[101, "pr_review_comment_count_7d"] == 0
    assert before.loc[202, "issue_comment_count_7d"] == 1
    assert before.loc[101, "push_distinct_repos_30d"] == 1
    assert before.loc[303, "push_last_seen_days"] == -1
    assert before.notna().all().all()


def test_future_mutation_cannot_change_old_features(synthetic):
    src = synthetic["source"]
    t = "2026-09-28T00:00:00Z"
    baseline = FeatureBuilder(src).build(t)
    src.events("push").loc[len(src.events("push"))] = {
        "UserID": 101, "repo_id": 42,
        "seen_utc": pd.Timestamp("2026-10-02T00:00:00Z"),
        "event_utc": pd.Timestamp("2026-09-01T00:00:00Z"),
    }
    after = FeatureBuilder(src).build(t)
    pd.testing.assert_frame_equal(baseline, after)


def test_features_user_subset_and_determinism(synthetic):
    src = synthetic["source"]
    b = FeatureBuilder(src)
    t = "2026-10-03T00:00:00Z"
    a = b.build(t, user_ids=[202, 101])
    again = b.build(t, user_ids=[202, 101])
    pd.testing.assert_frame_equal(a, again)
    assert a.UserID.tolist() == [202, 101]
    assert all(not col.startswith("y_") for col in a.columns)
    assert not any("merged_at" in col or "closed_at" in col for col in a.columns)
