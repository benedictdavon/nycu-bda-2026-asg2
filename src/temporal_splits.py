"""Chronological weekly windows with label-maturity and cross-split embargo guards."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .data_loader import utc

TZ = "Asia/Taipei"


@dataclass(frozen=True)
class SplitConfig:
    first_cutoff_taiwan: str = "2026-03-02"
    validation_start_taiwan: str = "2026-08-03"
    holdout_start_taiwan: str = "2026-10-12"
    last_cutoff_taiwan: str | None = None
    horizon_days: int = 7
    label_maturation_days: int = 7
    purge_boundary_maturity: bool = True


def taiwan_date(value: str) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize(TZ) if t.tzinfo is None else t.tz_convert(TZ)


def build_calendar(latest_observed, config: SplitConfig, *,
                   unseal_holdout: bool = False) -> pd.DataFrame:
    """Training/validation labels only mature by latest row-time in the snapshot.

    latest_observed is a heuristic availability indicator, NOT a guarantee of
    complete GH Archive ingestion. Holdout features can exist before labels mature.
    """
    if config.horizon_days <= 0 or config.label_maturation_days < 0:
        raise ValueError("Invalid horizon or label maturation")
    train_from = taiwan_date(config.first_cutoff_taiwan)
    val_from = taiwan_date(config.validation_start_taiwan)
    hold_from = taiwan_date(config.holdout_start_taiwan)
    if not (train_from < val_from < hold_from):
        raise ValueError("Require first_cutoff < validation_start < holdout_start")
    latest = utc(latest_observed)
    max_t = latest.tz_convert(TZ)
    if config.last_cutoff_taiwan:
        max_t = min(max_t, taiwan_date(config.last_cutoff_taiwan))
    if max_t < train_from:
        return pd.DataFrame(columns=["cutoff_utc", "label_end_utc", "label_as_of_utc", "split"])
    dates = pd.date_range(train_from, max_t, freq="7D")
    result = []
    for t_tw in dates:
        t = t_tw.tz_convert("UTC")
        end = t + pd.Timedelta(days=config.horizon_days)
        as_of = end + pd.Timedelta(days=config.label_maturation_days)
        split = "train" if t_tw < val_from else "validation" if t_tw < hold_from else "holdout"
        if split != "holdout" and as_of > latest:
            continue
        if split == "holdout" and unseal_holdout and as_of > latest:
            # Avoid partial/missing holdout labels when intentionally unsealing.
            continue
        if config.purge_boundary_maturity:
            if split == "train" and as_of > val_from.tz_convert("UTC"):
                continue
            if split == "validation" and as_of > hold_from.tz_convert("UTC"):
                continue
        result.append({
            "cutoff_utc": t, "label_end_utc": end, "label_as_of_utc": as_of,
            "split": split,
        })
    frame = pd.DataFrame(result)
    if frame.empty:
        return pd.DataFrame(columns=["cutoff_utc", "label_end_utc", "label_as_of_utc", "split"])
    if frame.duplicated(["cutoff_utc"]).any():
        raise AssertionError("Calendar generated duplicate cutoffs")
    for split, bound in (("train", val_from), ("validation", hold_from)):
        rows = frame.loc[frame.split.eq(split)]
        if len(rows) and config.purge_boundary_maturity:
            assert rows.label_as_of_utc.max() <= bound.tz_convert("UTC")
    return frame
