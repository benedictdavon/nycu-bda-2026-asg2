import numpy as np
import pandas as pd

from src.data_loader import TARGETS
from src.phase_b_rules import aligned_flags, ew_scores, multi_week_flags, rank01

C = pd.Timestamp("2026-09-23T12:00:00Z")


def _events(rows):
    """rows: list of (UserID, seen, event) applied to every target."""
    f = pd.DataFrame(rows, columns=["UserID", "seen_utc", "event_utc"])
    f["seen_utc"] = pd.to_datetime(f.seen_utc, utc=True, format="ISO8601")
    f["event_utc"] = pd.to_datetime(f.event_utc, utc=True, format="ISO8601")
    f["repo_id"] = 0
    return {t: f for t in TARGETS}


def test_aligned_requires_event_in_window_and_visibility_before_cutoff():
    ev = _events([
        (1, "2026-09-20", "2026-09-19"),            # event and seen in window -> 1
        (2, "2026-09-23T12:00:00Z", "2026-09-20"),  # seen exactly at C -> excluded
        (3, "2026-09-20", "2026-08-01"),            # seen in window but old event -> 0
        (4, "2026-09-10", "2026-09-17"),            # impossible ordering kept literal: seen before window, event in window -> 1
    ])
    out = aligned_flags(ev, C, [1, 2, 3, 4], 7)
    assert out[:, 0].tolist() == [1, 0, 0, 1]


def test_multi_week_requires_two_weeks_or_last_week():
    ev = _events([(1, "2026-09-10", "2026-09-10"), (1, "2026-09-03", "2026-09-03"),
                  (2, "2026-09-10", "2026-09-10"), (3, "2026-09-22", "2026-09-22")])
    out = multi_week_flags(ev, C, [1, 2, 3])
    assert out[:, 0].tolist() == [1, 0, 1]


def test_ew_scores_decay_and_cutoff():
    ev = _events([(1, "2026-09-23T11:00:00Z", "2026-09-23"), (2, "2026-09-16T12:00:00Z", "2026-09-16"),
                  (3, "2026-09-23T12:00:00Z", "2026-09-23")])
    s = ew_scores(ev, C, [1, 2, 3], 7.0)
    assert s[0, 0] > s[1, 0] > 0 and s[2, 0] == 0.0
    assert np.isclose(s[1, 0], np.exp(-1.0))


def test_rank01_direction_and_ties():
    r = rank01(np.array([[0.1], [0.9], [0.1]]))
    assert r[1, 0] == 1.0 and r[0, 0] == r[2, 0]
