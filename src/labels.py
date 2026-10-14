"""Historical future-event labels with an independently configurable observation cutoff.

Only event-time labels are targets. 'seen_utc' gates entity availability as of the
selected release, not the feature cutoff. Current-snapshot event tables do not expose
an independent ingestion timestamp; their seen time equals their event time.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .data_loader import OfficialData, TARGETS, utc, load_official_evaluator


class LabelBuilder:
    def __init__(self, source: OfficialData):
        self.source = source

    def build(self, start, *, horizon_days: int = 7, label_as_of,
              user_ids: pd.Index | list | None = None) -> pd.DataFrame:
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        t = utc(start)
        end = t + pd.Timedelta(days=horizon_days)
        as_of = utc(label_as_of)
        if as_of < end:
            raise ValueError("label_as_of must not precede the future-label window end")
        ids = self.source.cohort_ids if user_ids is None else pd.Index(user_ids, name="UserID")
        if ids.has_duplicates or ids.hasnans:
            raise ValueError("Label users must be unique and non-null")
        if len(ids.difference(self.source.cohort_ids)):
            raise ValueError("Non-cohort users cannot have reliable historical labels")
        result = pd.DataFrame({"UserID": ids.astype("int64")})
        result["cutoff_utc"] = t
        for target in TARGETS:
            e = self.source.events(target)
            qualifying = (
                e.event_utc.ge(t) & e.event_utc.lt(end)
                & e.seen_utc.le(as_of)
            )
            positives = pd.Index(e.loc[qualifying, "UserID"].unique())
            result[target] = result.UserID.isin(positives).astype("int8")
        return result


def verify_sample_truth(source: OfficialData, *,
                        start="2026-09-27T13:00:00Z",
                        label_as_of="2026-10-09T12:00:00Z",
                        truth_path: str | Path | None = None) -> dict:
    """Compare to the instructor's calibrated sample (NOT an independent holdout).

    The default observation cutoff is Oct 9 20:00 Asia/Taipei. It reconstructs all
    28,710 example labels for the Oct 14 course SHA 506d860...; future snapshots may
    revise historical observations and must be audited rather than silently accepted.
    """
    truth_path = (Path(truth_path) if truth_path is not None else
                  source.official_repo / "evaluation/examples/sample_truth.csv")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Instructor sample truth missing: {truth_path}")
    evaluator = load_official_evaluator(source.official_repo)
    truth = evaluator.read_predictions(str(truth_path))
    pred = LabelBuilder(source).build(start, label_as_of=label_as_of,
                                      user_ids=pd.Index(truth.UserID))
    pred = pred[["UserID", *TARGETS]]
    score = evaluator.scores(pred, truth)
    mismatches = {target: int(pred[target].ne(truth[target]).sum()) for target in TARGETS}
    counts = {}
    for target in TARGETS:
        got, want = pred[target].eq(1), truth[target].eq(1)
        counts[target] = {
            "tp": int((got & want).sum()), "fp": int((got & ~want).sum()),
            "fn": int((~got & want).sum()),
            "f1": float(evaluator.f1_per_event(pred, truth)[list(TARGETS).index(target)]),
        }
    return {
        "sample_users": len(truth), "binary_decisions": len(truth) * len(TARGETS),
        "start_utc": str(utc(start)), "end_utc": str(utc(start) + pd.Timedelta(days=7)),
        "label_as_of_utc": str(utc(label_as_of)),
        "dataset_sha": source.snapshot_sha, "scores": score,
        "mismatches": mismatches, "total_mismatches": sum(mismatches.values()),
        "confusion": counts,
        "calibration_warning": "This example was used to select a window and cutoff; it is NOT untouched validation.",
    }
