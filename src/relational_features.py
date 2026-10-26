"""Point-in-time repository and cohort-neighbor features from direct events.

This adapter reads the three direct-event Parquets with their source primary keys.
``OfficialData.events`` intentionally omits those keys, so it is not suitable for
this feature family. All joins here are validated many-to-one lookups against the
within-release Users and Repositories primary keys; only stable GitHub IDs leave
the source boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import OfficialData, utc


A_COLUMNS = [
    "direct_repo_distinct_7d",
    "direct_repo_distinct_30d",
    "direct_repo_distinct_90d",
    "direct_repo_top_share_30d",
    "direct_repo_entropy_30d",
    "direct_repo_returning_fraction_30d",
    "direct_repo_mean_activity_types_30d",
    "direct_repo_has_history_30d",
    "direct_repo_unresolved_events_30d",
]

B_COLUMNS = [
    "neighbor_mean_events_7d",
    "neighbor_distinct_users_7d",
    "neighbor_mean_rate_ratio_7v23d",
    "neighbor_mean_push_rate_7d",
    "neighbor_mean_issue_comment_rate_7d",
    "neighbor_weighted_rate_ratio_7v23d",
    "neighbor_repo_coverage_fraction_7d",
]

_SOURCE_SPECS = (
    {
        "table": "Pushes",
        "target": "push",
        "actor_fk": "user_id",
        "event_pk": "push_id",
        "stable_event_id": "github_push_id",
    },
    {
        "table": "IssueComments",
        "target": "issue_comment",
        "actor_fk": "author_id",
        "event_pk": "comment_id",
        "stable_event_id": "github_comment_id",
    },
    {
        "table": "PRReviewComments",
        "target": "pr_review_comment",
        "actor_fk": "author_id",
        "event_pk": "comment_id",
        "stable_event_id": "github_comment_id",
    },
)

_WINDOWS = (7, 30, 90)
_A_COUNT_COLUMNS = {
    "direct_repo_distinct_7d",
    "direct_repo_distinct_30d",
    "direct_repo_distinct_90d",
    "direct_repo_has_history_30d",
    "direct_repo_unresolved_events_30d",
}
_B_COUNT_COLUMNS = {
    "neighbor_distinct_users_7d",
}

COVERAGE_COLUMNS = [
    "UserID",
    "cutoff_utc",
    "actor_observed_at_cutoff",
    "actor_first_seen_days",
    *[
        name
        for days in _WINDOWS
        for name in (
            f"direct_events_{days}d_cohort",
            f"direct_events_{days}d_linked",
            f"direct_events_{days}d_repo_null",
            f"direct_events_{days}d_repo_unknown",
            f"direct_events_{days}d_repo_not_yet_visible",
            f"direct_events_{days}d_repo_stable_id_missing",
            f"direct_repos_{days}d",
        )
    ],
    "direct_events_prior60d_cohort",
    "direct_events_prior60d_linked",
    "direct_events_prior60d_repo_null",
    "direct_events_prior60d_repo_unknown",
    "direct_events_prior60d_repo_not_yet_visible",
    "direct_events_prior60d_repo_stable_id_missing",
    "direct_repos_prior60d",
    "direct_repo_support_30d",
    "direct_repo_dominant_events_30d",
    "direct_repo_dominant_share_30d",
    "neighbor_exposure_repos_30d",
    "neighbor_repos_with_events_7d",
    "neighbor_repos_zero_prior23d",
    "neighbor_other_events_7d",
    "neighbor_other_events_prior23d",
    "neighbor_distinct_users_7d",
]

@dataclass(frozen=True)
class _LoadedSource:
    users: pd.DataFrame
    repositories: pd.DataFrame
    events: pd.DataFrame
    table_audits: dict[str, dict[str, Any]]


def _count_by_user(frame: pd.DataFrame, ids: pd.Index) -> pd.Series:
    if frame.empty:
        return pd.Series(0, index=ids, dtype="int64")
    return frame.UserID.value_counts().reindex(ids, fill_value=0).astype("int64")


class RelationalFeatureBuilder:
    """Build the registered nine A and seven B additions at an exclusive cutoff.

    Source Parquets are loaded once per builder, which lets replay callers reuse
    validated event identities across cutoffs. Each cutoff is computed from the
    raw source timestamps with strict as-of endpoint masks. Results are cached by
    cutoff and returned in the caller's requested stable UserID order.
    """

    def __init__(self, source: OfficialData):
        self.source = source
        self._loaded: _LoadedSource | None = None
        self._cache: dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._cutoff_audits: dict[str, dict[str, Any]] = {}

    def _load(self) -> _LoadedSource:
        if self._loaded is not None:
            return self._loaded

        # Deliberately use only the loader's whitelisted read boundary. The public
        # events() convenience frames have discarded the atomic source PKs.
        users = self.source._read(
            "Users", ["user_id", "github_user_id", "is_cohort", "first_seen_at"]
        ).copy()
        repositories = self.source._read(
            "Repositories", ["repo_id", "github_repo_id", "first_seen_at"]
        ).copy()
        self._validate_key(users, "user_id", "Users")
        self._validate_key(repositories, "repo_id", "Repositories")
        if users.github_user_id.isna().any() or not users.github_user_id.is_unique:
            raise ValueError("Users.github_user_id must be unique and non-null")
        stable_repo_ids = repositories.github_repo_id.dropna()
        if not stable_repo_ids.is_unique:
            raise ValueError("Repositories.github_repo_id must be unique when present")

        users["UserID"] = pd.to_numeric(users.github_user_id, errors="raise").astype("int64")
        users["is_cohort"] = users.is_cohort.eq(True)
        users["first_seen_utc"] = pd.to_datetime(users.first_seen_at, utc=True, errors="coerce")
        repositories["first_seen_utc"] = pd.to_datetime(
            repositories.first_seen_at, utc=True, errors="coerce"
        )

        user_lookup = users.set_index("user_id", verify_integrity=True)
        repo_lookup = repositories.set_index("repo_id", verify_integrity=True)
        event_frames: list[pd.DataFrame] = []
        table_audits: dict[str, dict[str, Any]] = {}

        for spec in _SOURCE_SPECS:
            table = spec["table"]
            event_pk = spec["event_pk"]
            stable_id = spec["stable_event_id"]
            actor_fk = spec["actor_fk"]
            raw = self.source._read(
                table, [event_pk, stable_id, actor_fk, "repo_id", "created_at"]
            ).copy()
            self._validate_key(raw, event_pk, table)
            # Keep the original PK and its namespace as the atomic row identity.
            raw["source_table"] = table
            raw["source_pk"] = raw[event_pk]
            raw["event_key"] = table + ":" + raw[event_pk].astype("string")
            raw["stable_event_id"] = raw[stable_id]
            raw["actor_fk"] = raw[actor_fk]
            raw["repo_fk"] = raw["repo_id"]
            raw["created_at_utc"] = pd.to_datetime(raw.created_at, utc=True, errors="coerce")

            # Maps are many-to-one because their indexed source PKs were validated.
            raw["actor_user_id"] = raw.actor_fk.map(user_lookup.UserID)
            raw["actor_is_cohort"] = (
                raw.actor_fk.map(user_lookup.is_cohort).astype("boolean").fillna(False).astype(bool)
            )
            raw["actor_first_seen_utc"] = pd.to_datetime(
                raw.actor_fk.map(user_lookup.first_seen_utc), utc=True, errors="coerce"
            )
            raw["repo_github_id"] = raw.repo_fk.map(repo_lookup.github_repo_id)
            raw["repo_first_seen_utc"] = pd.to_datetime(
                raw.repo_fk.map(repo_lookup.first_seen_utc), utc=True, errors="coerce"
            )
            raw["repo_fk_known"] = raw.repo_fk.notna() & raw.repo_fk.isin(repo_lookup.index)
            raw["actor_fk_known"] = raw.actor_fk.notna() & raw.actor_fk.isin(user_lookup.index)

            normalized = raw[[
                "source_table", "source_pk", "event_key", "stable_event_id", "actor_fk",
                "actor_fk_known", "actor_user_id", "actor_is_cohort", "actor_first_seen_utc",
                "repo_fk", "repo_fk_known", "repo_github_id", "repo_first_seen_utc",
                "created_at_utc",
            ]].copy()
            # UserID is the stable cohort key used by output panels. The raw FK
            # remains alongside it for source-level join and conservation audits.
            normalized["UserID"] = normalized.actor_user_id
            normalized["target"] = spec["target"]
            event_frames.append(normalized)

            event_id_nonnull = raw[stable_id].dropna()
            table_audits[table] = {
                "source_rows": int(len(raw)),
                "primary_key_column": event_pk,
                "primary_key_null_rows": int(raw[event_pk].isna().sum()),
                "primary_key_unique": bool(raw[event_pk].is_unique),
                "namespaced_event_key_unique": bool(raw.event_key.is_unique),
                "stable_event_id_column": stable_id,
                "stable_event_id_null_rows": int(raw[stable_id].isna().sum()),
                "stable_event_id_distinct_nonnull": int(event_id_nonnull.nunique()),
                "stable_event_id_duplicate_rows": int(event_id_nonnull.duplicated().sum()),
                "actor_fk_column": actor_fk,
                "actor_fk_null_rows": int(raw.actor_fk.isna().sum()),
                "actor_fk_unknown_rows": int((raw.actor_fk.notna() & ~raw.actor_fk_known).sum()),
                "actor_fk_noncohort_rows": int((raw.actor_fk_known & ~raw.actor_is_cohort).sum()),
                "repo_fk_null_rows": int(raw.repo_fk.isna().sum()),
                "repo_fk_unknown_rows": int((raw.repo_fk.notna() & ~raw.repo_fk_known).sum()),
                "created_at_missing_rows": int(raw.created_at_utc.isna().sum()),
                "many_to_one_user_join": True,
                "many_to_one_repository_join": True,
                "rows_before_join": int(len(raw)),
                "rows_after_join": int(len(normalized)),
                "atomic_event_rows_conserved": bool(len(raw) == len(normalized)),
            }

        events = pd.concat(event_frames, ignore_index=True)
        if events.event_key.duplicated().any():
            # Namespacing means equal comment PKs in the two comment tables remain
            # separate; only a same-table duplicate is a source-contract failure.
            raise ValueError("Namespaced direct-event primary keys must be unique")
        cohort_ids = pd.Index(
            users.loc[users.is_cohort, "UserID"].sort_values(), name="UserID"
        )
        if cohort_ids.has_duplicates or cohort_ids.hasnans:
            raise ValueError("Stable cohort UserID values must be unique and non-null")
        self._loaded = _LoadedSource(
            users=users,
            repositories=repositories,
            events=events,
            table_audits=table_audits,
        )
        return self._loaded

    @staticmethod
    def _validate_key(frame: pd.DataFrame, column: str, table: str) -> None:
        if frame[column].isna().any():
            raise ValueError(f"{table}.{column} primary key cannot contain nulls")
        if not frame[column].is_unique:
            raise ValueError(f"{table}.{column} primary key must be unique")

    def _requested_ids(self, user_ids: pd.Index | list | None) -> pd.Index:
        loaded = self._load()
        cohort_ids = pd.Index(
            loaded.users.loc[loaded.users.is_cohort, "UserID"].sort_values(), name="UserID"
        )
        ids = cohort_ids if user_ids is None else pd.Index(user_ids, name="UserID")
        if ids.has_duplicates or ids.hasnans:
            raise ValueError("Feature users must have distinct, non-null stable UserIDs")
        try:
            ids = ids.astype("int64")
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature users must be integer stable UserIDs") from exc
        unknown = ids.difference(cohort_ids)
        if len(unknown):
            raise ValueError("Features requested for non-cohort or unknown users")
        return pd.Index(ids, name="UserID")

    @staticmethod
    def _repo_status(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
        status = pd.Series("linked", index=events.index, dtype="object")
        status.loc[events.repo_fk.isna()] = "repo_null"
        status.loc[events.repo_fk.notna() & ~events.repo_fk_known] = "repo_unknown"
        known = events.repo_fk_known
        not_visible = known & (
            events.repo_first_seen_utc.isna() | events.repo_first_seen_utc.ge(cutoff)
        )
        status.loc[not_visible] = "repo_not_yet_visible"
        stable_missing = (
            known & ~not_visible & events.repo_github_id.isna()
        )
        status.loc[stable_missing] = "repo_stable_id_missing"
        return status

    def _compute(
        self,
        cutoff: pd.Timestamp,
        *,
        omit_repo_github_id: Any | None = None,
        record_cutoff_audit: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        loaded = self._load()
        users = loaded.users
        events = loaded.events.copy()
        all_ids = pd.Index(
            users.loc[users.is_cohort, "UserID"].sort_values(), name="UserID"
        )

        event_pre_cutoff = events.created_at_utc.notna() & events.created_at_utc.lt(cutoff)
        actor_known_cohort = events.actor_fk_known & events.actor_is_cohort
        actor_visible = (
            events.actor_first_seen_utc.notna()
            & events.actor_first_seen_utc.lt(cutoff)
        )
        actor_eligible = actor_known_cohort & actor_visible
        events["repo_status"] = self._repo_status(events, cutoff)
        events["actor_eligible"] = actor_eligible
        events["event_pre_cutoff"] = event_pre_cutoff
        events["linked"] = events.repo_status.eq("linked")
        focal = events.loc[event_pre_cutoff & actor_eligible].copy()

        source_rows = int(len(events))
        pre_cutoff_actor_rows = event_pre_cutoff
        actor_reasons = {
            "actor_fk_null": int((pre_cutoff_actor_rows & events.actor_fk.isna()).sum()),
            "actor_fk_unknown": int((pre_cutoff_actor_rows & events.actor_fk.notna() & ~events.actor_fk_known).sum()),
            "actor_noncohort": int((pre_cutoff_actor_rows & events.actor_fk_known & ~events.actor_is_cohort).sum()),
            "cohort_actor_not_yet_visible": int((pre_cutoff_actor_rows & actor_known_cohort & ~actor_visible).sum()),
            "cohort_actor_visible": int((pre_cutoff_actor_rows & actor_eligible).sum()),
        }
        eligible_pre_cutoff = int((event_pre_cutoff & actor_eligible).sum())
        linked_pre_cutoff = int(
            (event_pre_cutoff & actor_eligible & events.linked).sum()
        )
        reason_counts = {
            reason: int((event_pre_cutoff & actor_eligible & events.repo_status.eq(reason)).sum())
            for reason in (
                "repo_null", "repo_unknown", "repo_not_yet_visible", "repo_stable_id_missing"
            )
        }
        cutoff_audit = {
            "source_event_rows": source_rows,
            "event_time_missing_rows": int(events.created_at_utc.isna().sum()),
            "event_time_before_cutoff_rows": int(event_pre_cutoff.sum()),
            "pre_cutoff_event_rows_by_actor_reason": actor_reasons,
            "pre_cutoff_actor_reason_conservation": bool(
                int(event_pre_cutoff.sum()) == sum(actor_reasons.values())
            ),
            "eligible_cohort_actor_rows_before_cutoff": eligible_pre_cutoff,
            "linked_rows_before_cutoff": linked_pre_cutoff,
            "unlinked_rows_before_cutoff_by_repo_reason": reason_counts,
            "eligible_event_row_conservation": bool(
                eligible_pre_cutoff == linked_pre_cutoff + sum(reason_counts.values())
            ),
            "atomic_rows_conserved_after_endpoint_classification": bool(
                int(events.repo_status.value_counts(dropna=False).sum()) == source_rows
            ),
        }
        if record_cutoff_audit:
            self._cutoff_audits[str(cutoff)] = cutoff_audit

        # The single 90-day focal slice supports every A window and both B
        # intervals. Event rows are never deduplicated by timestamp or stable ID.
        lower90 = cutoff - pd.Timedelta(days=90)
        history90 = focal.loc[focal.created_at_utc.ge(lower90)].copy()
        history90["repo_internal"] = history90.repo_fk
        linked90 = history90.loc[history90.linked].copy()
        all_linked90 = linked90
        baseline_current30 = all_linked90.loc[
            all_linked90.created_at_utc.ge(cutoff - pd.Timedelta(days=30))
        ]
        repo_degrees = baseline_current30.groupby("repo_github_id", sort=False).UserID.nunique()
        if len(repo_degrees):
            maximum_degree = int(repo_degrees.max())
            top_candidates = repo_degrees.index[repo_degrees.eq(maximum_degree)]
            # Stable IDs, never release-local repo row numbers, resolve ties.
            highest_degree_repo = sorted(top_candidates.tolist())[0]
            top_rows = baseline_current30.loc[
                baseline_current30.repo_github_id.eq(highest_degree_repo)
            ]
            exposed_repo_counts = baseline_current30.groupby(
                "UserID", sort=False
            ).repo_github_id.nunique()
            only_top_users = set(
                exposed_repo_counts.index[
                    exposed_repo_counts.eq(1)
                    & exposed_repo_counts.index.isin(top_rows.UserID.unique())
                ]
            )
            if isinstance(highest_degree_repo, np.generic):
                highest_degree_repo = highest_degree_repo.item()
            degree_audit = {
                "github_repo_id": highest_degree_repo,
                "degree_definition": "distinct observed cohort actors with valid linked events in 30d",
                "tie_break": "smallest stable github_repo_id among max-degree repos",
                "degree": maximum_degree,
                "event_rows": int(len(top_rows)),
                "event_share": float(len(top_rows) / len(baseline_current30)),
                "source_rows_by_table": {
                    str(name): int(count)
                    for name, count in top_rows.source_table.value_counts().items()
                },
                "cohort_users_exposed": int(top_rows.UserID.nunique()),
                "cohort_users_only_exposed_to_repo": int(len(only_top_users)),
                "all_valid_30d_event_rows": int(len(baseline_current30)),
                "rows_after_omitting_repo": int(len(baseline_current30) - len(top_rows)),
            }
        else:
            highest_degree_repo = None
            degree_audit = {
                "github_repo_id": None,
                "degree_definition": "distinct observed cohort actors with valid linked events in 30d",
                "tie_break": "smallest stable github_repo_id among max-degree repos",
                "degree": 0,
                "event_rows": 0,
                "event_share": 0.0,
                "source_rows_by_table": {},
                "cohort_users_exposed": 0,
                "cohort_users_only_exposed_to_repo": 0,
                "all_valid_30d_event_rows": 0,
                "rows_after_omitting_repo": 0,
            }
        if record_cutoff_audit:
            self._cutoff_audits[str(cutoff)]["highest_degree_repo_30d"] = degree_audit
        if omit_repo_github_id is not None:
            linked90 = linked90.loc[
                ~linked90.repo_github_id.eq(omit_repo_github_id)
            ].copy()
        linked90["repo_internal"] = linked90.repo_fk

        features = pd.DataFrame({"UserID": all_ids.astype("int64")})
        features["cutoff_utc"] = cutoff
        for column in A_COLUMNS:
            features[column] = np.int32(0) if column in _A_COUNT_COLUMNS else np.float32(0)
        for column in B_COLUMNS:
            features[column] = np.int32(0) if column in _B_COUNT_COLUMNS else np.float32(0)

        # A: repository self-history, all counts are source rows (push rows are
        # pushes, not commits) and the distinct repo key is within-release repo_id.
        for days in _WINDOWS:
            current = linked90.loc[linked90.created_at_utc.ge(cutoff - pd.Timedelta(days=days))]
            distinct = current.groupby("UserID", sort=False).repo_internal.nunique()
            features[f"direct_repo_distinct_{days}d"] = (
                features.UserID.map(distinct).fillna(0).astype("int32")
            )

        current30 = linked90.loc[
            linked90.created_at_utc.ge(cutoff - pd.Timedelta(days=30))
        ]
        pair30 = (
            current30.groupby(["UserID", "repo_internal"], sort=False)
            .size().rename("n").reset_index()
        )
        total30 = pair30.groupby("UserID", sort=False).n.sum()
        dominant30 = pair30.groupby("UserID", sort=False).n.max()
        top_share = dominant30.div(1 + total30)
        features["direct_repo_top_share_30d"] = (
            features.UserID.map(top_share).fillna(0).astype("float32")
        )

        if not pair30.empty:
            pair30["total"] = pair30.UserID.map(total30)
            pair30["p"] = pair30.n / pair30.total
            pair30["entropy_term"] = -pair30.p * np.log(pair30.p)
            entropy = pair30.groupby("UserID", sort=False).entropy_term.sum()
            activity_types = (
                current30.groupby(["UserID", "repo_internal"], sort=False)
                .source_table.nunique().rename("types").reset_index()
            )
            mean_types = activity_types.groupby("UserID", sort=False).types.mean()
        else:
            entropy = pd.Series(dtype="float64")
            mean_types = pd.Series(dtype="float64")
        features["direct_repo_entropy_30d"] = (
            features.UserID.map(entropy).fillna(0).astype("float32")
        )
        features["direct_repo_mean_activity_types_30d"] = (
            features.UserID.map(mean_types).fillna(0).astype("float32")
        )
        features["direct_repo_has_history_30d"] = (
            features.UserID.map(total30.gt(0).astype("int32")).fillna(0).astype("int32")
        )

        prior = linked90.loc[
            linked90.created_at_utc.ge(cutoff - pd.Timedelta(days=90))
            & linked90.created_at_utc.lt(cutoff - pd.Timedelta(days=30))
        ]
        current_sets = current30.groupby("UserID", sort=False).repo_internal.agg(set)
        prior_sets = prior.groupby("UserID", sort=False).repo_internal.agg(set)
        returning: dict[int, float] = {}
        for uid, repos in current_sets.items():
            returning[int(uid)] = len(repos.intersection(prior_sets.get(uid, set()))) / len(repos)
        features["direct_repo_returning_fraction_30d"] = (
            features.UserID.map(returning).fillna(0).astype("float32")
        )

        # Coverage retains each focal actor's pre-endpoint event mass so missing,
        # unknown, and late repository endpoints remain visible in diagnostics.
        coverage = pd.DataFrame({"UserID": all_ids.astype("int64")})
        coverage["cutoff_utc"] = cutoff
        user_meta = users.loc[users.is_cohort].set_index("UserID", verify_integrity=True)
        user_seen = user_meta.first_seen_utc.reindex(all_ids)
        user_observed = user_seen.notna() & user_seen.lt(cutoff)
        coverage["actor_observed_at_cutoff"] = user_observed.to_numpy(dtype="int8")
        coverage["actor_first_seen_days"] = (
            ((cutoff - user_seen).dt.total_seconds() / 86400.0)
            .where(user_observed, -1.0).astype("float32").to_numpy()
        )

        coverage_events = history90
        reason_columns = {
            "repo_null": "repo_null",
            "repo_unknown": "repo_unknown",
            "repo_not_yet_visible": "repo_not_yet_visible",
            "repo_stable_id_missing": "repo_stable_id_missing",
        }
        for days in _WINDOWS:
            rows = coverage_events.loc[
                coverage_events.created_at_utc.ge(cutoff - pd.Timedelta(days=days))
            ]
            linked = rows.loc[rows.linked]
            coverage[f"direct_events_{days}d_cohort"] = (
                coverage.UserID.map(_count_by_user(rows, all_ids)).astype("int32")
            )
            coverage[f"direct_events_{days}d_linked"] = (
                coverage.UserID.map(_count_by_user(linked, all_ids)).astype("int32")
            )
            for status, suffix in reason_columns.items():
                group = rows.loc[rows.repo_status.eq(status)]
                coverage[f"direct_events_{days}d_{suffix}"] = (
                    coverage.UserID.map(_count_by_user(group, all_ids)).astype("int32")
                )
            repo_counts = linked.groupby("UserID", sort=False).repo_internal.nunique()
            coverage[f"direct_repos_{days}d"] = (
                coverage.UserID.map(repo_counts).fillna(0).astype("int32")
            )

        for key, rows in (
            ("prior60d", coverage_events.loc[
                coverage_events.created_at_utc.ge(cutoff - pd.Timedelta(days=90))
                & coverage_events.created_at_utc.lt(cutoff - pd.Timedelta(days=30))
            ]),
        ):
            linked = rows.loc[rows.linked]
            coverage[f"direct_events_{key}_cohort"] = (
                coverage.UserID.map(_count_by_user(rows, all_ids)).astype("int32")
            )
            coverage[f"direct_events_{key}_linked"] = (
                coverage.UserID.map(_count_by_user(linked, all_ids)).astype("int32")
            )
            for status, suffix in reason_columns.items():
                group = rows.loc[rows.repo_status.eq(status)]
                coverage[f"direct_events_{key}_{suffix}"] = (
                    coverage.UserID.map(_count_by_user(group, all_ids)).astype("int32")
                )
            repo_counts = linked.groupby("UserID", sort=False).repo_internal.nunique()
            coverage[f"direct_repos_{key}"] = (
                coverage.UserID.map(repo_counts).fillna(0).astype("int32")
            )

        repo_count30 = coverage.set_index("UserID")["direct_repos_30d"]
        coverage["direct_repo_support_30d"] = coverage.UserID.map(
            repo_count30.map(lambda n: "none" if n == 0 else "one" if n == 1 else "many")
        )
        dominant = pair30.groupby("UserID", sort=False).n.max()
        total = pair30.groupby("UserID", sort=False).n.sum()
        coverage["direct_repo_dominant_events_30d"] = (
            coverage.UserID.map(dominant).fillna(0).astype("int32")
        )
        coverage["direct_repo_dominant_share_30d"] = (
            coverage.UserID.map(dominant.div(total)).fillna(0).astype("float32")
        )

        # A's unresolved companion includes every visible focal direct event in
        # the 30-day window whose repository cannot be linked by stable identity.
        # The four disjoint reasons are also retained separately in coverage.
        unresolved30 = coverage[[
            f"direct_events_30d_{suffix}" for suffix in reason_columns.values()
        ]].sum(axis=1).astype("int32")
        features["direct_repo_unresolved_events_30d"] = unresolved30.to_numpy(dtype="int32")

        # B: aggregate repo totals once, then subtract each focal user's own rows
        # for every exposed repo before any neighbor statistic is calculated.
        exposure = pair30.rename(columns={"n": "own30"}).copy()
        if not exposure.empty:
            current7 = current30.loc[
                current30.created_at_utc.ge(cutoff - pd.Timedelta(days=7))
            ]
            prior23 = current30.loc[
                current30.created_at_utc.ge(cutoff - pd.Timedelta(days=30))
                & current30.created_at_utc.lt(cutoff - pd.Timedelta(days=7))
            ]

            def repo_total(frame: pd.DataFrame, name: str) -> pd.Series:
                return frame.groupby("repo_internal", sort=False).size().rename(name)

            repo_totals = pd.concat([
                repo_total(current7, "total7"),
                repo_total(prior23, "total23"),
                repo_total(current7.loc[current7.target.eq("push")], "push7"),
                repo_total(current7.loc[current7.target.eq("issue_comment")], "issue_comment7"),
            ], axis=1).fillna(0)
            own7 = current7.groupby(["UserID", "repo_internal"], sort=False).size().rename("own7")
            own23 = prior23.groupby(["UserID", "repo_internal"], sort=False).size().rename("own23")
            own_push7 = (
                current7.loc[current7.target.eq("push")]
                .groupby(["UserID", "repo_internal"], sort=False).size().rename("own_push7")
            )
            own_issue7 = (
                current7.loc[current7.target.eq("issue_comment")]
                .groupby(["UserID", "repo_internal"], sort=False).size().rename("own_issue7")
            )
            own_counts = pd.concat([own7, own23, own_push7, own_issue7], axis=1).fillna(0)
            exposure = exposure.set_index(["UserID", "repo_internal"]).join(
                own_counts, how="left"
            ).join(repo_totals, on="repo_internal", how="left").fillna(0).reset_index()
            for col in ("own7", "own23", "own_push7", "own_issue7", "total7", "total23", "push7", "issue_comment7"):
                if col not in exposure:
                    exposure[col] = 0
            exposure["other7"] = exposure.total7 - exposure.own7
            exposure["other23"] = exposure.total23 - exposure.own23
            exposure["other_push7"] = exposure.push7 - exposure.own_push7
            exposure["other_issue7"] = exposure.issue_comment7 - exposure.own_issue7
            if (exposure[["other7", "other23", "other_push7", "other_issue7"]] < 0).any().any():
                raise AssertionError("Self-excluded neighbor counts cannot be negative")
            exposure["q"] = (exposure.other7 / 7.0) / (exposure.other23 / 23.0 + 1e-6)
            exposure["weight"] = exposure.own30 / exposure.groupby("UserID").own30.transform("sum")

            def mean_to_feature(source_col: str, feature_col: str, *, divisor: float = 1.0) -> None:
                mean = (exposure[source_col] / divisor).groupby(exposure.UserID, sort=False).mean()
                features[feature_col] = features.UserID.map(mean).fillna(0).astype("float32")

            mean_to_feature("other7", "neighbor_mean_events_7d")
            mean_to_feature("q", "neighbor_mean_rate_ratio_7v23d")
            mean_to_feature("other_push7", "neighbor_mean_push_rate_7d", divisor=7.0)
            mean_to_feature("other_issue7", "neighbor_mean_issue_comment_rate_7d", divisor=7.0)
            weighted = (exposure.weight * exposure.q).groupby(exposure.UserID, sort=False).sum()
            features["neighbor_weighted_rate_ratio_7v23d"] = (
                features.UserID.map(weighted).fillna(0).astype("float32")
            )
            repo_coverage = (
                exposure.other7.gt(0).groupby(exposure.UserID, sort=False).mean()
            )
            features["neighbor_repo_coverage_fraction_7d"] = (
                features.UserID.map(repo_coverage).fillna(0).astype("float32")
            )

            # Distinct neighbors are a union across the exposed repos. Python
            # integer bitsets avoid a potentially quadratic user×repo expansion.
            user_bit = {int(uid): 1 << i for i, uid in enumerate(all_ids)}
            active_pairs = current7[["repo_internal", "UserID"]].drop_duplicates()
            repo_bits: dict[Any, int] = {}
            for repo_id, uid in active_pairs.itertuples(index=False, name=None):
                repo_bits[repo_id] = repo_bits.get(repo_id, 0) | user_bit[int(uid)]
            distinct_neighbors: dict[int, int] = {}
            for uid, rows in exposure.groupby("UserID", sort=False):
                neighbor_bits = 0
                for repo_id in rows.repo_internal:
                    neighbor_bits |= repo_bits.get(repo_id, 0)
                neighbor_bits &= ~user_bit[int(uid)]
                distinct_neighbors[int(uid)] = neighbor_bits.bit_count()
            features["neighbor_distinct_users_7d"] = (
                features.UserID.map(distinct_neighbors).fillna(0).astype("int32")
            )

            coverage["neighbor_exposure_repos_30d"] = (
                coverage.UserID.map(exposure.groupby("UserID", sort=False).size())
                .fillna(0).astype("int32")
            )
            coverage["neighbor_repos_with_events_7d"] = (
                coverage.UserID.map(
                    exposure.loc[exposure.other7.gt(0)].groupby("UserID", sort=False).size()
                ).fillna(0).astype("int32")
            )
            coverage["neighbor_repos_zero_prior23d"] = (
                coverage.UserID.map(
                    exposure.loc[exposure.other23.eq(0)].groupby("UserID", sort=False).size()
                ).fillna(0).astype("int32")
            )
            coverage["neighbor_other_events_7d"] = (
                coverage.UserID.map(exposure.groupby("UserID", sort=False).other7.sum())
                .fillna(0).astype("int32")
            )
            coverage["neighbor_other_events_prior23d"] = (
                coverage.UserID.map(exposure.groupby("UserID", sort=False).other23.sum())
                .fillna(0).astype("int32")
            )
            coverage["neighbor_distinct_users_7d"] = (
                coverage.UserID.map(distinct_neighbors).fillna(0).astype("int32")
            )
        else:
            coverage["neighbor_exposure_repos_30d"] = np.int32(0)
            coverage["neighbor_repos_with_events_7d"] = np.int32(0)
            coverage["neighbor_repos_zero_prior23d"] = np.int32(0)
            coverage["neighbor_other_events_7d"] = np.int32(0)
            coverage["neighbor_other_events_prior23d"] = np.int32(0)
            coverage["neighbor_distinct_users_7d"] = np.int32(0)

        # Exact ordered contracts and finite values are enforced at the boundary.
        features = features[["UserID", "cutoff_utc", *A_COLUMNS, *B_COLUMNS]]
        coverage = coverage[COVERAGE_COLUMNS]
        numeric = features[A_COLUMNS + B_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise AssertionError("Relational feature outputs must all be finite")
        if features[A_COLUMNS + B_COLUMNS].isna().any().any():
            raise AssertionError("Relational feature outputs cannot contain nulls")

        # Cutoff-level source conservation is useful even if the caller requests
        # only a small output panel.
        if eligible_pre_cutoff != linked_pre_cutoff + sum(reason_counts.values()):
            raise AssertionError("Eligible event rows were not conserved by repo status")
        return features, coverage

    def _for_users(
        self,
        cutoff: pd.Timestamp,
        user_ids: pd.Index | list | None,
        result_index: int,
    ) -> pd.DataFrame:
        ids = self._requested_ids(user_ids)
        if cutoff not in self._cache:
            self._cache[cutoff] = self._compute(cutoff)
        frame = self._cache[cutoff][result_index].set_index("UserID", drop=False)
        return frame.loc[ids.to_list()].reset_index(drop=True).copy()

    def build(
        self,
        cutoff,
        user_ids: pd.Index | list | None = None,
    ) -> pd.DataFrame:
        """Return A+B additions, with cutoff and user IDs as explicit metadata."""
        return self._for_users(utc(cutoff), user_ids, 0)

    def coverage(
        self,
        cutoff,
        user_ids: pd.Index | list | None = None,
    ) -> pd.DataFrame:
        """Return keyed source, endpoint, history-support and neighbor diagnostics."""
        return self._for_users(utc(cutoff), user_ids, 1)

    def concentration_audit(self, cutoff) -> dict[str, Any]:
        """Summarize the max-degree 30d repository and recompute without it.

        Degree is the number of distinct observed cohort actors with valid linked
        direct events in ``[C-30d,C)``. Ties are resolved by smallest stable
        ``github_repo_id``. Omission features are rebuilt from the normalized
        source rows at the same cutoff; they are diagnostics, never model inputs.
        """
        point = utc(cutoff)
        if point not in self._cache:
            self._cache[point] = self._compute(point)
        base_features = self._cache[point][0]
        top = self._cutoff_audits[str(point)]["highest_degree_repo_30d"]
        top_repo = top["github_repo_id"]
        if top_repo is None:
            omitted_features = base_features
        else:
            omitted_features, _ = self._compute(
                point,
                omit_repo_github_id=top_repo,
                record_cutoff_audit=False,
            )

        aggregates: dict[str, dict[str, Any]] = {}
        for column in [*A_COLUMNS, *B_COLUMNS]:
            base = base_features[column].to_numpy(dtype=np.float64)
            omitted = omitted_features[column].to_numpy(dtype=np.float64)
            aggregates[column] = {
                "baseline_mean": float(base.mean()) if len(base) else 0.0,
                "baseline_median": float(np.median(base)) if len(base) else 0.0,
                "baseline_p95": float(np.quantile(base, 0.95)) if len(base) else 0.0,
                "baseline_max": float(base.max()) if len(base) else 0.0,
                "without_top_repo_mean": float(omitted.mean()) if len(omitted) else 0.0,
                "without_top_repo_median": float(np.median(omitted)) if len(omitted) else 0.0,
                "without_top_repo_p95": float(np.quantile(omitted, 0.95)) if len(omitted) else 0.0,
                "without_top_repo_max": float(omitted.max()) if len(omitted) else 0.0,
                "mean_change": float(omitted.mean() - base.mean()) if len(base) else 0.0,
                "users_changed": int(np.count_nonzero(omitted != base)),
            }
        return {
            "cutoff_utc": point.isoformat(),
            "highest_degree_repo_30d": dict(top),
            "omission_rule": "recompute all A/B aggregates from linked direct rows after removing the selected repo for every actor",
            "aggregate_feature_diagnostics": aggregates,
        }

    def source_audit(self) -> dict[str, Any]:
        """Return a JSON-compatible source PK, stable-ID, FK and join audit."""
        loaded = self._load()
        return {
            "schema_version": 1,
            "snapshot_sha": self.source.snapshot_sha,
            "data_directory": str(self.source.data_dir),
            "direct_event_tables": [spec["table"] for spec in _SOURCE_SPECS],
            "event_source_rows": int(len(loaded.events)),
            "namespaced_event_keys_unique": bool(loaded.events.event_key.is_unique),
            "event_row_conservation_after_many_to_one_lookups": bool(
                all(audit["atomic_event_rows_conserved"] for audit in loaded.table_audits.values())
            ),
            "users": {
                "rows": int(len(loaded.users)),
                "cohort_rows": int(loaded.users.is_cohort.sum()),
                "user_id_primary_key_unique": bool(loaded.users.user_id.is_unique),
                "github_user_id_unique_nonnull": bool(
                    loaded.users.github_user_id.notna().all()
                    and loaded.users.github_user_id.is_unique
                ),
                "first_seen_missing_rows": int(loaded.users.first_seen_utc.isna().sum()),
            },
            "repositories": {
                "rows": int(len(loaded.repositories)),
                "repo_id_primary_key_unique": bool(loaded.repositories.repo_id.is_unique),
                "github_repo_id_null_rows": int(loaded.repositories.github_repo_id.isna().sum()),
                "github_repo_id_unique_when_present": bool(
                    loaded.repositories.github_repo_id.dropna().is_unique
                ),
                "first_seen_missing_rows": int(loaded.repositories.first_seen_utc.isna().sum()),
            },
            "tables": {name: dict(audit) for name, audit in loaded.table_audits.items()},
            "cutoff_audits": {key: dict(value) for key, value in self._cutoff_audits.items()},
        }


def feature_inventory() -> list[dict[str, str]]:
    """Return the registered ordered feature definitions for freeze manifests."""
    definitions = [
        ("direct_repo_distinct_7d", "A", "R01 direct-stream variant", "distinct repos in [C-7d,C)"),
        ("direct_repo_distinct_30d", "A", "R01 direct-stream variant", "distinct repos in [C-30d,C)"),
        ("direct_repo_distinct_90d", "A", "R01 direct-stream variant", "distinct repos in [C-90d,C)"),
        ("direct_repo_top_share_30d", "A", "R02 variant", "max user-repo event rows / (1 + all user event rows) in 30d"),
        ("direct_repo_entropy_30d", "A", "R03 variant", "Shannon entropy of user-repo event-row shares in 30d, natural log"),
        ("direct_repo_returning_fraction_30d", "A", "R05 bounded-history variant", "fraction of 30d repos also seen in prior [C-90d,C-30d)"),
        ("direct_repo_mean_activity_types_30d", "A", "R12 direct-stream unweighted variant", "mean distinct source tables per user-repo pair in 30d"),
        ("direct_repo_has_history_30d", "A", "denominator support", "1 when at least one linked repo is active in 30d"),
        ("direct_repo_unresolved_events_30d", "A", "coverage companion", "focal visible direct-event rows in 30d with a null, unknown, not-yet-visible repo FK, or missing stable repo ID"),
        ("neighbor_mean_events_7d", "B", "R15 variant", "unweighted mean over 30d exposed repos of other-cohort event rows in 7d"),
        ("neighbor_distinct_users_7d", "B", "R16 union-count variant", "distinct other cohort users with 7d events on any exposed repo"),
        ("neighbor_mean_rate_ratio_7v23d", "B", "R17 variant", "unweighted mean of (other 7d/7)/(other prior 23d/23 + 1e-6)"),
        ("neighbor_mean_push_rate_7d", "B", "R19 mean variant", "unweighted mean over exposed repos of other Pushes rows / 7"),
        ("neighbor_mean_issue_comment_rate_7d", "B", "R21 mean variant", "unweighted mean over exposed repos of other IssueComments rows / 7"),
        ("neighbor_weighted_rate_ratio_7v23d", "B", "R25 exact trend choice", "own 30d event-share weighted mean of the fixed 7v23d rate ratio"),
        ("neighbor_repo_coverage_fraction_7d", "B", "coverage companion", "fraction of own 30d exposed repos with any other-cohort event in 7d"),
    ]
    return [
        {
            "column": column,
            "group": group,
            "catalog_mapping": catalog,
            "definition": definition,
            "source_tables": "Pushes,IssueComments,PRReviewComments",
            "source_primary_key": "(source_table,push_id|comment_id)",
            "event_time": "created_at",
            "window_convention": "left-inclusive, right-exclusive at exclusive cutoff C",
            "storage_dtype": "int32" if column in _A_COUNT_COLUMNS | _B_COUNT_COLUMNS else "float32",
        }
        for column, group, catalog, definition in definitions
    ]
