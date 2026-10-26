"""Independent-loop oracle and strict source/cutoff tests for synthesized features."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from experiments.dfs_pca_20261026.features import (
    COLUMNS, DIRECT_TARGETS, EXPRESSIONS, SUPPORT, SynthesizedRepoFeatures, inventory,
)

C = pd.Timestamp("2026-09-10T12:00:00Z")
TABLES = {"push": ("Pushes", "user_id", "push_id", "github_push_id"),
          "issue_comment": ("IssueComments", "author_id", "comment_id", "github_comment_id"),
          "pr_review_comment": ("PRReviewComments", "author_id", "comment_id", "github_comment_id")}


class Source:
    snapshot_sha = "synthetic"

    def __init__(self):
        self.tables = {
            "Users": pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "github_user_id": [101, 202, 303, 404, 505],
                "is_cohort": [True, True, True, False, True],
                "first_seen_at": [C - pd.Timedelta(days=100)] * 4 + [C]}),
            "Repositories": pd.DataFrame({"repo_id": [10, 11, 12, 13], "github_repo_id": [900, 901, 902, None],
                "first_seen_at": [C - pd.Timedelta(days=100)] * 2 + [C, C - pd.Timedelta(days=100)]})}
        for table, actor, pk, stable in TABLES.values():
            self.tables[table] = pd.DataFrame(columns=[actor, pk, stable, "repo_id", "created_at"])

    def add(self, target, actor, repo, days=1, *, at=None):
        table, actor_field, pk, stable = TABLES[target]
        frame = self.tables[table]
        i = len(frame) + 1
        row = {actor_field: actor, pk: i, stable: 1000 + i, "repo_id": repo,
               "created_at": C - pd.Timedelta(days=days) if at is None else at}
        self.tables[table] = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    def _read(self, table, columns):
        return self.tables[table][columns].copy()


@pytest.fixture
def source():
    s = Source()
    s.add("push", 1, 10, 1)
    s.add("issue_comment", 1, 11, 8)
    s.add("push", 2, 10, 2)
    s.add("push", 2, 10, 3)
    s.add("push", 2, 11, 20)
    s.add("issue_comment", 3, 10, 2)
    s.add("push", 4, 10, 1)  # noncohort
    s.add("push", 5, 10, 1)  # actor first visible exactly C
    s.add("push", 1, 12, 1)  # repository first visible exactly C
    s.add("push", 1, 13, 1)  # no stable repository identity
    s.add("push", 1, 999, 1) # unresolved repository
    return s


def oracle(source, ids):
    users = source.tables["Users"].set_index("user_id").to_dict("index")
    repos = source.tables["Repositories"].set_index("repo_id").to_dict("index")
    events = []
    for target, (table, actor_field, _, _) in TABLES.items():
        for row in source.tables[table].to_dict("records"):
            user, repo = users.get(row[actor_field]), repos.get(row["repo_id"])
            time = row["created_at"]
            if not user or not repo or not user["is_cohort"] or pd.isna(repo["github_repo_id"]):
                continue
            if user["first_seen_at"] >= C or repo["first_seen_at"] >= C or not C - pd.Timedelta(days=30) <= time < C:
                continue
            events.append((user["github_user_id"], repo["github_repo_id"], target, time))
    result = []
    for uid in ids:
        exposed = sorted({repo for actor, repo, _, _ in events if actor == uid})
        row = {}
        for e in EXPRESSIONS:
            per_repo = []
            for repo in exposed:
                neighbors = [(a, t) for a, r, t, when in events if a != uid and r == repo
                             and when >= C - pd.Timedelta(days=e.window_days)]
                value = len({a for a, _ in neighbors}) if e.base_primitive == "num_unique" else sum(t == e.target for _, t in neighbors)
                per_repo.append(value)
            row[e.column] = (getattr(np, e.outer_primitive)(per_repo) if per_repo else 0.0)
        row[SUPPORT] = len(exposed)
        result.append(row)
    return pd.DataFrame(result, index=ids)[list(COLUMNS)]


def test_generated_grammar_matches_independent_oracle_and_hand_counts(source):
    features, audit = SynthesizedRepoFeatures(source).build(C, [303, 101, 202, 505])
    np.testing.assert_array_equal(features.UserID, [303, 101, 202, 505])
    np.testing.assert_allclose(features[list(COLUMNS)], oracle(source, features.UserID), rtol=0, atol=1e-14)
    user = features.set_index("UserID").loc[101]
    assert user[SUPPORT] == 2
    assert user.dfs_repo_other_push_7d_mean == 1
    assert user.dfs_repo_other_push_7d_max == 2
    assert user.dfs_repo_other_push_7d_std == 1
    assert user.dfs_repo_other_push_30d_mean == 1.5
    assert user.dfs_repo_other_all_direct_users_7d_max == 2
    assert audit["all_events_and_endpoints_before_cutoff"] is True
    assert audit["excluded_not_yet_visible_endpoints"] == 2
    assert len(inventory()) == len(COLUMNS) == 25


def test_future_event_and_endpoint_perturbations_do_not_change_features(source):
    baseline, _ = SynthesizedRepoFeatures(source).build(C, [101, 202, 303, 505])
    later = deepcopy(source)
    for _ in range(12):
        later.add("push", 2, 10, at=C)
        later.add("issue_comment", 3, 10, at=C + pd.Timedelta(seconds=1))
        later.add("push", 5, 10, days=1)
        later.add("pr_review_comment", 1, 12, days=1)
        later.add("push", 4, 10, days=1)
    augmented, _ = SynthesizedRepoFeatures(later).build(C, [101, 202, 303, 505])
    pd.testing.assert_frame_equal(baseline, augmented)


def test_exact_lower_boundaries_and_upper_exclusion(source):
    source.add("push", 3, 11, days=7)
    source.add("push", 3, 11, days=30)
    source.add("push", 3, 11, at=C - pd.Timedelta(days=30) - pd.Timedelta(nanoseconds=1))
    source.add("push", 3, 11, at=C)
    frame, _ = SynthesizedRepoFeatures(source).build(C, [101, 202, 303])
    np.testing.assert_allclose(frame[list(COLUMNS)], oracle(source, frame.UserID), rtol=0, atol=1e-14)


def test_timestamp_unit_conversion_and_namespaced_primary_keys(source):
    baseline, _ = SynthesizedRepoFeatures(source).build(C, [101, 202, 303])
    for table in source.tables.values():
        for column in ("created_at", "first_seen_at"):
            if column in table:
                table[column] = pd.to_datetime(table[column], utc=True).dt.as_unit("us")
    actual, _ = SynthesizedRepoFeatures(source).build(C, [101, 202, 303])
    pd.testing.assert_frame_equal(baseline, actual)
    # PK 1 in different direct tables remains two different atomic events.
    assert SynthesizedRepoFeatures(source).source_summary["source_event_keys_unique"] is True


def test_no_neighbor_no_exposure_and_singleton_std():
    s = Source()
    s.add("push", 1, 10)
    frame, _ = SynthesizedRepoFeatures(s).build(C, [101, 202])
    assert frame.loc[0, SUPPORT] == 1
    assert (frame.loc[0, [e.column for e in EXPRESSIONS]] == 0).all()
    assert (frame.loc[1, list(COLUMNS)] == 0).all()


def test_duplicate_or_unknown_requested_user_is_rejected(source):
    builder = SynthesizedRepoFeatures(source)
    for ids in ([101, 101], [404], [999]):
        with pytest.raises(ValueError, match="cohort"):
            builder.build(C, ids)
