import json

import numpy as np
import pandas as pd
import pytest

from src.relational_features import (
    A_COLUMNS,
    B_COLUMNS,
    COVERAGE_COLUMNS,
    RelationalFeatureBuilder,
    feature_inventory,
)


CUTOFF = pd.Timestamp("2026-10-01T00:00:00Z")


def _at(days=0, *, nanoseconds=0):
    return CUTOFF - pd.Timedelta(days=days) - pd.Timedelta(nanoseconds=nanoseconds)


def _push(pk, actor, repo, time, *, stable=None):
    return {
        "push_id": pk,
        "github_push_id": 50_000 + pk if stable is None else stable,
        "user_id": actor,
        "repo_id": repo,
        "created_at": time,
    }


def _comment(pk, actor, repo, time, *, stable=None):
    return {
        "comment_id": pk,
        "github_comment_id": 60_000 + pk if stable is None else stable,
        "author_id": actor,
        "repo_id": repo,
        "created_at": time,
    }


def _rich_raw(synthetic):
    users = synthetic["tables"]["Users"].copy()
    repo_seen = pd.Timestamp("2025-12-01T00:00:00Z")
    repositories = pd.DataFrame({
        "repo_id": [10, 11, 12, 13, 20, 30],
        "github_repo_id": [900, 901, 902, 903, 920, None],
        "first_seen_at": [repo_seen, repo_seen, repo_seen, repo_seen, CUTOFF, repo_seen],
    })
    pushes = pd.DataFrame([
        _push(1, 0, 12, _at(90)),                         # C-90 is included
        _push(2, 0, 10, _at(45)),                         # prior 60d repo return
        _push(3, 0, 10, _at(7)),                          # C-7 is included
        _push(4, 0, 13, CUTOFF),                          # C itself is excluded
        _push(5, 0, 14, _at(90, nanoseconds=1)),         # just outside 90d, unknown repo
        _push(16, 0, 30, _at(6)),                        # visible known repo without a stable ID
        _push(6, 1, 10, _at(7)),
        _push(7, 1, 10, _at(20)),
        _push(8, 1, 10, _at(30)),                         # C-30 is in prior 23d
        _push(9, 1, 11, _at(1)),
        _push(10, 1, 10, _at(45)),
        _push(11, 2, 10, _at(4), stable=70_000),
        _push(12, 2, 10, _at(4), stable=70_000),         # distinct PKs, tied time and stable ID
        _push(13, 2, 13, CUTOFF + pd.Timedelta(days=1)), # future event
        _push(14, None, 10, _at(1)),                     # null actor is audited, never attributed
        _push(15, 3, 10, _at(1)),                        # known noncohort actor is excluded
    ])
    issue_comments = pd.DataFrame([
        _comment(20, 0, 11, _at(30)),
        _comment(21, 0, 20, _at(30)),                    # known repo, not visible at C
        _comment(22, 0, None, _at(5)),                   # null repo endpoint
        _comment(23, 0, 14, _at(3)),                     # unknown repo endpoint
        _comment(24, 1, 11, _at(2)),
        _comment(25, 2, 11, _at(3)),
    ])
    review_comments = pd.DataFrame([
        _comment(24, 0, 10, _at(1), stable=60_024),      # same comment_id, different table namespace
        _comment(25, 1, 11, _at(3)),
    ])
    return {
        "Users": users,
        "Repositories": repositories,
        "Pushes": pushes,
        "IssueComments": issue_comments,
        "PRReviewComments": review_comments,
    }


def _install_raw(synthetic, monkeypatch, raw=None):
    source = synthetic["source"]
    raw = _rich_raw(synthetic) if raw is None else raw

    def read(table, columns):
        missing = set(columns).difference(raw[table].columns)
        if missing:
            raise KeyError(f"{table} missing requested fields: {sorted(missing)}")
        return raw[table].loc[:, columns].copy()

    monkeypatch.setattr(source, "_read", read)
    return source, raw


def test_exact_registered_columns_and_hand_calculated_a_and_b(synthetic, monkeypatch):
    source, _ = _install_raw(synthetic, monkeypatch)
    builder = RelationalFeatureBuilder(source)
    out = builder.build(CUTOFF)

    assert len(A_COLUMNS) == 9
    assert len(B_COLUMNS) == 7
    assert out.columns.tolist() == ["UserID", "cutoff_utc", *A_COLUMNS, *B_COLUMNS]
    assert out.UserID.tolist() == [101, 202, 303]
    assert out.cutoff_utc.eq(CUTOFF).all()
    assert np.isfinite(out[A_COLUMNS + B_COLUMNS].to_numpy(dtype=np.float64)).all()
    assert out["direct_repo_distinct_7d"].dtype == np.dtype("int32")
    assert out["neighbor_mean_events_7d"].dtype == np.dtype("float32")
    assert out["neighbor_distinct_users_7d"].dtype == np.dtype("int32")

    result = out.set_index("UserID")
    user = result.loc[101]
    # User 101 has linked 30d repo counts {repo 10: 2, repo 11: 1}.
    assert user.direct_repo_distinct_7d == 1
    assert user.direct_repo_distinct_30d == 2  # includes the exact C-30 row
    assert user.direct_repo_distinct_90d == 3  # includes C-90, excludes C-90-1ns
    assert user.direct_repo_top_share_30d == pytest.approx(2 / (1 + 3), abs=1e-7)
    assert user.direct_repo_entropy_30d == pytest.approx(
        -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3), abs=1e-7
    )
    assert user.direct_repo_returning_fraction_30d == pytest.approx(1 / 2)
    assert user.direct_repo_mean_activity_types_30d == pytest.approx(1.5)
    assert user.direct_repo_has_history_30d == 1
    assert user.direct_repo_unresolved_events_30d == 4  # includes known repo with missing stable ID

    # Repo exposure weights for user 101 are 2/3 on repo 10 and 1/3 on repo 11.
    q10 = (3 / 7) / (2 / 23 + 1e-6)
    q11 = (4 / 7) / (0 / 23 + 1e-6)
    assert user.neighbor_mean_events_7d == pytest.approx(3.5, abs=1e-6)
    assert user.neighbor_distinct_users_7d == 2
    assert user.neighbor_mean_rate_ratio_7v23d == pytest.approx((q10 + q11) / 2, rel=1e-6)
    assert user.neighbor_mean_push_rate_7d == pytest.approx(2 / 7, abs=1e-7)
    assert user.neighbor_mean_issue_comment_rate_7d == pytest.approx(1 / 7, abs=1e-7)
    assert user.neighbor_weighted_rate_ratio_7v23d == pytest.approx(
        (2 / 3) * q10 + (1 / 3) * q11, rel=1e-6
    )
    assert user.neighbor_repo_coverage_fraction_7d == pytest.approx(1.0)

    # User 202's 30d repo counts are {10: 3, 11: 3}; activity types are 1 and 3.
    user202 = result.loc[202]
    assert user202.direct_repo_top_share_30d == pytest.approx(3 / 7, abs=1e-7)
    assert user202.direct_repo_entropy_30d == pytest.approx(np.log(2), abs=1e-7)
    assert user202.direct_repo_returning_fraction_30d == pytest.approx(0.5)
    assert user202.direct_repo_mean_activity_types_30d == pytest.approx(2.0)
    assert user202.neighbor_distinct_users_7d == 2  # user 303 active on both exposed repos, counted once

    coverage = builder.coverage(CUTOFF)
    assert coverage.columns.tolist() == COVERAGE_COLUMNS
    assert len(COVERAGE_COLUMNS) == 41
    c101 = coverage.set_index("UserID").loc[101]
    assert c101.direct_events_30d_cohort == 7
    assert c101.direct_events_30d_linked == 3
    assert c101.direct_events_30d_repo_null == 1
    assert c101.direct_events_30d_repo_unknown == 1
    assert c101.direct_events_30d_repo_not_yet_visible == 1
    assert c101.direct_events_30d_repo_stable_id_missing == 1
    assert c101.direct_repos_30d == 2
    assert c101.direct_repo_support_30d == "many"
    assert c101.neighbor_exposure_repos_30d == 2
    assert c101.neighbor_repos_zero_prior23d == 1
    assert c101.neighbor_other_events_7d == 7
    assert c101.neighbor_other_events_prior23d == 2

    concentration = builder.concentration_audit(CUTOFF)
    top = concentration["highest_degree_repo_30d"]
    assert top["github_repo_id"] == 900  # repo 900/901 tie at degree 3; stable ID breaks the tie
    assert top["degree"] == 3
    assert top["event_rows"] == 7
    assert top["cohort_users_only_exposed_to_repo"] == 0
    assert concentration["aggregate_feature_diagnostics"]["direct_repo_distinct_30d"][
        "baseline_mean"
    ] == pytest.approx(2.0)
    assert concentration["aggregate_feature_diagnostics"]["direct_repo_distinct_30d"][
        "without_top_repo_mean"
    ] == pytest.approx(1.0)

    inventory = feature_inventory()
    assert len(inventory) == 16
    assert [row["column"] for row in inventory] == [*A_COLUMNS, *B_COLUMNS]
    for item in inventory:
        assert out[item["column"]].dtype == np.dtype(item["storage_dtype"])


def test_user_order_is_preserved_and_self_activity_is_excluded_with_fixed_exposure(synthetic, monkeypatch):
    source, raw = _install_raw(synthetic, monkeypatch)
    base_builder = RelationalFeatureBuilder(source)
    requested = [303, 101]
    base = base_builder.build(CUTOFF, user_ids=requested)
    assert base.UserID.tolist() == requested
    assert base_builder.coverage(CUTOFF, user_ids=requested).UserID.tolist() == requested

    before = base_builder.build(CUTOFF).set_index("UserID").loc[202, B_COLUMNS]
    # Move one of user 202's own repo-10 rows from the prior-23d window to 7d.
    # Exposure repos and 30d weights stay fixed; self-subtracted neighbor values
    # and the distinct-user union should therefore remain unchanged.
    shifted = raw["Pushes"].copy()
    shifted.loc[shifted.push_id.eq(7), "created_at"] = _at(5)
    raw["Pushes"] = shifted
    fresh_builder = RelationalFeatureBuilder(source)
    after = fresh_builder.build(CUTOFF).set_index("UserID").loc[202, B_COLUMNS]
    np.testing.assert_allclose(before.to_numpy(dtype=float), after.to_numpy(dtype=float), rtol=1e-6, atol=1e-6)


def test_future_injection_endpoint_visibility_and_source_row_conservation(synthetic, monkeypatch):
    source, raw = _install_raw(synthetic, monkeypatch)
    baseline = RelationalFeatureBuilder(source).build(CUTOFF)

    # A future direct event is invisible even when its linked endpoints are old.
    future_push = pd.DataFrame([_push(100, 0, 10, CUTOFF + pd.Timedelta(seconds=1))])
    raw["Pushes"] = pd.concat([raw["Pushes"], future_push], ignore_index=True)
    event_augmented = RelationalFeatureBuilder(source).build(CUTOFF)
    pd.testing.assert_frame_equal(baseline, event_augmented)

    # A pre-cutoff event with a repository first seen at C is not linked. It is
    # still recorded once as unresolved endpoint mass for its visible cohort actor.
    late_repo = pd.DataFrame({
        "repo_id": [99],
        "github_repo_id": [999],
        "first_seen_at": [CUTOFF],
    })
    raw["Repositories"] = pd.concat([raw["Repositories"], late_repo], ignore_index=True)
    late_endpoint = pd.DataFrame([_push(101, 0, 99, _at(2))])
    raw["Pushes"] = pd.concat([raw["Pushes"], late_endpoint], ignore_index=True)
    endpoint_builder = RelationalFeatureBuilder(source)
    after_endpoint = endpoint_builder.build(CUTOFF).set_index("UserID")
    before_by_user = baseline.set_index("UserID")
    pd.testing.assert_frame_equal(
        before_by_user.drop(columns="direct_repo_unresolved_events_30d"),
        after_endpoint.drop(columns="direct_repo_unresolved_events_30d"),
    )
    assert after_endpoint.loc[101, "direct_repo_unresolved_events_30d"] == 5
    audit = endpoint_builder.source_audit()
    cutoff_audit = audit["cutoff_audits"][str(CUTOFF)]
    assert cutoff_audit["pre_cutoff_actor_reason_conservation"] is True
    assert cutoff_audit["eligible_event_row_conservation"] is True
    assert cutoff_audit["unlinked_rows_before_cutoff_by_repo_reason"]["repo_not_yet_visible"] == 2

    # Actor visibility is also strict: an actor row first seen exactly at C cannot
    # supply an otherwise old direct event.
    late_actor_raw = _rich_raw(synthetic)
    late_actor_raw["Users"].loc[
        late_actor_raw["Users"].user_id.eq(2), "first_seen_at"
    ] = CUTOFF.tz_localize(None)
    late_actor_raw["Pushes"] = pd.DataFrame([
        _push(500, 2, 10, _at(1)),
    ])
    late_actor_source, _ = _install_raw(synthetic, monkeypatch, late_actor_raw)
    late_actor_builder = RelationalFeatureBuilder(late_actor_source)
    late_actor = late_actor_builder.build(CUTOFF).set_index("UserID")
    assert late_actor.loc[303, A_COLUMNS + B_COLUMNS].eq(0).all()
    late_actor_coverage = late_actor_builder.coverage(CUTOFF).set_index("UserID")
    assert late_actor_coverage.loc[303, "actor_observed_at_cutoff"] == 0
    assert late_actor_coverage.loc[303, "direct_events_30d_cohort"] == 0


def test_source_keys_are_namespaced_and_internal_row_renumbering_is_invariant(synthetic, monkeypatch):
    source, raw = _install_raw(synthetic, monkeypatch)
    builder = RelationalFeatureBuilder(source)
    baseline = builder.build(CUTOFF)
    loaded = builder._load()
    keys = set(loaded.events.event_key)
    assert "IssueComments:24" in keys
    assert "PRReviewComments:24" in keys
    assert loaded.events.event_key.is_unique
    audit = builder.source_audit()
    json.dumps(audit)
    assert audit["cutoff_audits"][str(CUTOFF)]["pre_cutoff_actor_reason_conservation"] is True
    assert audit["event_row_conservation_after_many_to_one_lookups"] is True
    assert audit["tables"]["Pushes"]["primary_key_column"] == "push_id"
    assert audit["tables"]["IssueComments"]["primary_key_column"] == "comment_id"
    assert audit["tables"]["PRReviewComments"]["stable_event_id_column"] == "github_comment_id"
    assert audit["tables"]["Pushes"]["stable_event_id_duplicate_rows"] == 1
    assert audit["tables"]["Pushes"]["actor_fk_null_rows"] == 1
    assert audit["tables"]["Pushes"]["actor_fk_noncohort_rows"] == 1

    # Internal user_id/repo_id values are release-local. A bijective rewrite
    # leaves stable UserID panels and all aggregate features unchanged.
    users = raw["Users"].copy()
    users["user_id"] = users.user_id.map({0: 100, 1: 200, 2: 300, 3: 900})
    raw["Users"] = users
    repos = raw["Repositories"].copy()
    repo_map = {10: 710, 11: 711, 12: 712, 13: 713, 20: 720, 30: 730}
    repos["repo_id"] = repos.repo_id.map(repo_map)
    raw["Repositories"] = repos
    actor_map = {0: 100, 1: 200, 2: 300, 3: 900}
    for table, actor_col in (("Pushes", "user_id"), ("IssueComments", "author_id"), ("PRReviewComments", "author_id")):
        frame = raw[table].copy()
        frame[actor_col] = frame[actor_col].map(actor_map)
        mapped_repo = frame.repo_id.map(repo_map)
        frame["repo_id"] = mapped_repo.where(mapped_repo.notna(), frame.repo_id)
        raw[table] = frame
    renumbered = RelationalFeatureBuilder(source).build(CUTOFF)
    pd.testing.assert_frame_equal(baseline, renumbered)


def test_duplicate_source_primary_key_fails_closed(synthetic, monkeypatch):
    source, raw = _install_raw(synthetic, monkeypatch)
    raw["Pushes"] = pd.concat([raw["Pushes"], raw["Pushes"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match=r"Pushes\.push_id primary key must be unique"):
        RelationalFeatureBuilder(source).build(CUTOFF)
