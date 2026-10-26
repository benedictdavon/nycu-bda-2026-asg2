"""Generate explicit COUNT/NUM_UNIQUE -> MEAN/MAX/STD repository expressions.

The source boundary is the existing relational adapter. Source row identities,
cohort restrictions and endpoint clocks are retained before any aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.data_loader import utc
from src.relational_features import RelationalFeatureBuilder

DIRECT_TARGETS = ("push", "issue_comment", "pr_review_comment")
WINDOWS = (7, 30)
PRIMITIVES = ("mean", "max", "std")


@dataclass(frozen=True)
class Expression:
    column: str
    base_primitive: str
    target: str
    window_days: int
    outer_primitive: str


def expressions():
    result = []
    for target in (*DIRECT_TARGETS, "all_direct_users"):
        base = "num_unique" if target == "all_direct_users" else "count"
        for window in WINDOWS:
            for outer in PRIMITIVES:
                result.append(Expression(f"dfs_repo_other_{target}_{window}d_{outer}", base, target, window, outer))
    return tuple(result)


EXPRESSIONS = expressions()
SUPPORT = "dfs_exposed_repositories_30d"
COLUMNS = tuple(e.column for e in EXPRESSIONS) + (SUPPORT,)


def inventory():
    return [{**asdict(e), "expression": f"{e.outer_primitive.upper()}_OVER_EXPOSED_REPOS_30D(OTHER_{e.base_primitive.upper()}({e.target},{e.window_days}d))",
             "catalogue": "M08/M09/R15/R16", "self_excluded": True,
             "std_ddof": 0 if e.outer_primitive == "std" else None, "empty_exposure": 0.0}
            for e in EXPRESSIONS] + [{"column": SUPPORT, "expression": "COUNT(DISTINCT_EXPOSED_REPOSITORIES_30D)",
                                     "catalogue": "M08/support", "empty_exposure": 0}]


class SynthesizedRepoFeatures:
    def __init__(self, source):
        self.adapter = RelationalFeatureBuilder(source)
        loaded = self.adapter._load()
        self.cohort_ids = pd.Index(loaded.users.loc[loaded.users.is_cohort, "UserID"], name="UserID")
        raw = loaded.events
        valid = (raw.actor_fk_known & raw.actor_is_cohort & raw.repo_fk_known & raw.repo_github_id.notna()
                 & raw.created_at_utc.notna() & raw.actor_first_seen_utc.notna() & raw.repo_first_seen_utc.notna())
        eligible = raw.loc[valid]
        events = pd.DataFrame({"UserID": eligible.UserID.to_numpy(dtype=np.int64),
            "RepoID": eligible.repo_github_id.to_numpy(dtype=np.int64), "target": eligible.target.to_numpy(),
            "time_ns": pd.DatetimeIndex(eligible.created_at_utc).as_unit("ns").asi8,
            "actor_seen_ns": pd.DatetimeIndex(eligible.actor_first_seen_utc).as_unit("ns").asi8,
            "repo_seen_ns": pd.DatetimeIndex(eligible.repo_first_seen_utc).as_unit("ns").asi8})
        self.events = events.sort_values("time_ns", kind="stable").reset_index(drop=True)
        self.times = self.events.time_ns.to_numpy()
        self.source_summary = {"source_rows": len(raw), "known_cohort_known_stable_endpoints_rows": len(events),
            "excluded_before_cutoff_processing": int((~valid).sum()),
            "source_pk_audits": loaded.table_audits, "source_event_keys_unique": bool(raw.event_key.is_unique)}

    def build(self, cutoff, user_ids):
        C = utc(cutoff)
        ids = pd.Index(user_ids, dtype="int64", name="UserID")
        if ids.has_duplicates or ids.hasnans or len(ids.difference(self.cohort_ids)):
            raise ValueError("Requested IDs must be distinct non-null cohort users")
        lo = int(np.searchsorted(self.times, (C - pd.Timedelta(days=30)).value, side="left"))
        hi = int(np.searchsorted(self.times, C.value, side="left"))
        window = self.events.iloc[lo:hi]
        current = window.loc[(window.actor_seen_ns < C.value) & (window.repo_seen_ns < C.value)]
        exposure = current[["UserID", "RepoID"]].drop_duplicates()
        exposure = exposure.loc[exposure.UserID.isin(ids)].set_index(["UserID", "RepoID"])
        result = pd.DataFrame({"UserID": ids.to_numpy(), "cutoff_utc": C})
        for column in COLUMNS:
            result[column] = 0.0
        pairs = exposure.index
        if len(pairs):
            support = exposure.reset_index().groupby("UserID", sort=False).size()
            result[SUPPORT] = result.UserID.map(support).fillna(0).astype("float64")
            inner = {}
            for days in WINDOWS:
                recent = current.loc[current.time_ns >= (C - pd.Timedelta(days=days)).value]
                for target in (*DIRECT_TARGETS, "all_direct_users"):
                    rows = recent if target == "all_direct_users" else recent.loc[recent.target == target]
                    by_pair = rows.groupby(["UserID", "RepoID"], sort=False).size()
                    own = by_pair.reindex(pairs, fill_value=0).to_numpy(dtype=np.int64)
                    repo_ids = pairs.get_level_values("RepoID")
                    if target == "all_direct_users":
                        totals = rows.groupby("RepoID", sort=False).UserID.nunique()
                        other = totals.reindex(repo_ids, fill_value=0).to_numpy(dtype=np.int64) - (own > 0)
                    else:
                        totals = rows.groupby("RepoID", sort=False).size()
                        other = totals.reindex(repo_ids, fill_value=0).to_numpy(dtype=np.int64) - own
                    if (other < 0).any():
                        raise AssertionError("Self-excluded repository counts cannot be negative")
                    inner[(target, days)] = pd.Series(other.astype(np.float64), index=pairs)
            for expression in EXPRESSIONS:
                series = inner[(expression.target, expression.window_days)]
                grouped = series.groupby(level="UserID", sort=False)
                if expression.outer_primitive == "std":
                    values = grouped.std(ddof=0)
                else:
                    values = getattr(grouped, expression.outer_primitive)()
                result[expression.column] = result.UserID.map(values).fillna(0).astype("float64")
        values = result[list(COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise AssertionError("Generated feature values must be finite and nonnegative")
        audit = {"cutoff_utc": C.isoformat(), "requested_users": len(ids), "window30_rows": len(window),
            "visible_endpoint_rows": len(current), "excluded_not_yet_visible_endpoints": len(window) - len(current),
            "exposure_pairs": len(pairs), "supported_users": int((result[SUPPORT] > 0).sum()),
            "max_event_utc": pd.Timestamp(current.time_ns.max(), tz="UTC").isoformat() if len(current) else None,
            "max_actor_seen_utc": pd.Timestamp(current.actor_seen_ns.max(), tz="UTC").isoformat() if len(current) else None,
            "max_repo_seen_utc": pd.Timestamp(current.repo_seen_ns.max(), tz="UTC").isoformat() if len(current) else None,
            "all_events_and_endpoints_before_cutoff": bool(len(current) == 0 or
                (current[["time_ns", "actor_seen_ns", "repo_seen_ns"]].to_numpy() < C.value).all()),
            "exact_neighbor_counts": True, "self_exclusion": True}
        return result, audit
