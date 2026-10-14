"""Point-in-time user-history features constructed exclusively from official row times.

Only cohort actor/event observations already available strictly before cutoff t enter
features; no labels, future target events, or late-known entity-state columns enter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import OfficialData, TARGETS, utc


class FeatureBuilder:
    def __init__(self, source: OfficialData, *, windows: tuple[int, ...] = (7, 30, 90),
                 eligibility_days: int = 90, eligibility_min_rows: int = 5,
                 missing_recency_days: float = -1.0):
        self.source = source
        self.windows = tuple(sorted(set(int(d) for d in windows)))
        if not self.windows or min(self.windows) <= 0:
            raise ValueError("At least one positive lookback window is required")
        if eligibility_days <= 0 or eligibility_min_rows < 0:
            raise ValueError("Invalid eligibility proxy configuration")
        self.eligibility_days = eligibility_days
        self.eligibility_min_rows = eligibility_min_rows
        self.missing_recency_days = missing_recency_days

    def build(self, cutoff, *, user_ids: pd.Index | list | None = None) -> pd.DataFrame:
        t = utc(cutoff)
        ids = self.source.cohort_ids if user_ids is None else pd.Index(user_ids, name="UserID")
        if ids.has_duplicates or ids.hasnans:
            raise ValueError("Feature users must have distinct, non-null IDs")
        if len(ids.difference(self.source.cohort_ids)):
            raise ValueError("Features requested for non-cohort users")
        out = pd.DataFrame({"UserID": ids.astype("int64")})
        out["cutoff_utc"] = t
        user_seen = out.UserID.map(self.source.cohort.set_index("UserID").first_seen_utc)
        seen_already = user_seen.lt(t) & user_seen.notna()
        out["user_observed_at_cutoff"] = seen_already.astype("int8")
        out["user_first_seen_days"] = (
            (t - user_seen).dt.total_seconds().div(86400)
            .where(seen_already, self.missing_recency_days).astype("float32")
        )
        proxy_total = pd.Series(0, index=out.index, dtype="int64")
        requested = set(ids)
        max_lookback = max(max(self.windows), self.eligibility_days, 56, 14)
        for target in TARGETS:
            src = self.source.events(target)
            # Strict historical row-time filtering: t itself is NOT observed yet.
            subset = src.loc[
                src.seen_utc.ge(t - pd.Timedelta(days=max_lookback))
                & src.seen_utc.lt(t)
                & src.UserID.isin(requested),
                ["UserID", "repo_id", "seen_utc"],
            ]
            by_user_last = subset.groupby("UserID", sort=False).seen_utc.max()
            previous = out.UserID.map(by_user_last)
            age = (t - previous).dt.total_seconds().div(86400)
            out[f"{target}_last_seen_days"] = age.fillna(self.missing_recency_days).astype("float32")
            # Count activity by row-visibility time, never entity creation time.
            for days in self.windows:
                recent = subset.loc[subset.seen_utc.ge(t - pd.Timedelta(days=days))]
                counts = recent.UserID.value_counts()
                out[f"{target}_count_{days}d"] = (
                    out.UserID.map(counts).fillna(0).astype("int32")
                )
                # Both actor and repo references come from an already-visible event row.
                if days in (30, 90):
                    breadth = recent.groupby("UserID", sort=False).repo_id.nunique()
                    out[f"{target}_distinct_repos_{days}d"] = (
                        out.UserID.map(breadth).fillna(0).astype("int32")
                    )
            recent7 = subset.loc[subset.seen_utc.ge(t - pd.Timedelta(days=7))]
            previous7 = subset.loc[
                subset.seen_utc.ge(t - pd.Timedelta(days=14))
                & subset.seen_utc.lt(t - pd.Timedelta(days=7))
            ]
            n_recent = out.UserID.map(recent7.UserID.value_counts()).fillna(0).astype("int32")
            n_before = out.UserID.map(previous7.UserID.value_counts()).fillna(0).astype("int32")
            out[f"{target}_active_prior_7d"] = (n_recent > 0).astype("int8")
            out[f"{target}_count_delta_7d"] = (n_recent - n_before).astype("int32")
            # Distinct nonoverlapping trailing 7-day bins containing observations.
            recent56 = subset.loc[subset.seen_utc.ge(t - pd.Timedelta(days=56))].copy()
            if len(recent56):
                recent56["week_bin"] = (
                    ((t - recent56.seen_utc).dt.total_seconds() - 0.000001)
                    .div(7 * 86400).clip(lower=0).astype("int8")
                )
                active_weeks = recent56.groupby("UserID").week_bin.nunique()
                out[f"{target}_active_weeks_8w"] = (
                    out.UserID.map(active_weeks).fillna(0).astype("int8")
                )
            else:
                out[f"{target}_active_weeks_8w"] = np.int8(0)
            eligibility_rows = subset.loc[
                subset.seen_utc.ge(t - pd.Timedelta(days=self.eligibility_days))
            ].UserID.value_counts()
            proxy_total = proxy_total.add(
                out.UserID.map(eligibility_rows).fillna(0).astype("int64"), fill_value=0
            ).astype("int64")
        out["historical_observation_count_90d_proxy"] = proxy_total
        out["eligible_proxy"] = (proxy_total >= self.eligibility_min_rows).astype("int8")
        out["activity_types_prior_7d"] = (
            out[[f"{name}_active_prior_7d" for name in TARGETS]]
            .sum(axis=1).astype("int8")
        )
        if out.isna().any().any():
            raise AssertionError("Unexpected nulls in features")
        return out
