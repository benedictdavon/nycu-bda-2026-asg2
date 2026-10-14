import pandas as pd
import pytest

from src.temporal_splits import SplitConfig, build_calendar


def test_purged_split_boundaries_and_holdout_seal():
    cfg = SplitConfig(
        first_cutoff_taiwan="2026-07-13",
        validation_start_taiwan="2026-08-03",
        holdout_start_taiwan="2026-08-31",
    )
    latest = "2026-09-20T00:00:00Z"
    c = build_calendar(latest, cfg)
    assert set(c.split) == {"train", "validation", "holdout"}
    assert c.loc[c.split.eq("train"), "label_as_of_utc"].max() <= pd.Timestamp("2026-08-02T16:00:00Z")
    assert c.loc[c.split.eq("validation"), "label_as_of_utc"].max() <= pd.Timestamp("2026-08-30T16:00:00Z")
    assert c.cutoff_utc.is_unique


def test_snapshot_maturity_and_holdout_unseal(synthetic):
    cfg = SplitConfig(
        first_cutoff_taiwan="2026-09-07",
        validation_start_taiwan="2026-09-14",
        holdout_start_taiwan="2026-09-21",
        horizon_days=7, label_maturation_days=7,
    )
    latest = "2026-10-01T00:00:00Z"
    c = build_calendar(latest, cfg, unseal_holdout=False)
    assert "holdout" in set(c.split)
    # The Sept 21 label end plus 7-day buffer is Oct 5, not mature by Oct 1.
    c_sealed = build_calendar(latest, cfg, unseal_holdout=True)
    assert "holdout" not in set(c_sealed.split)


def test_invalid_split_config():
    cfg = SplitConfig(validation_start_taiwan="2026-10-12", holdout_start_taiwan="2026-09-07")
    with pytest.raises(ValueError):
        build_calendar("2026-10-13T00:00:00Z", cfg)
