"""EXP-006 configuration and calibration/reporting schedule; dataset I/O stays shared."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

CONFIG_FILE = Path(__file__).resolve().parents[2] / "configs/baseline_comparison.json"
EXAMPLE_START = pd.Timestamp("2026-09-27T13:00:00Z")
EXAMPLE_END = pd.Timestamp("2026-10-04T13:00:00Z")

def load_config(path: str | Path | None = None) -> dict:
    reference = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    config = json.loads(Path(path).read_text(encoding="utf-8")) if path else reference
    unknown = set(config) - set(reference)
    missing = set(reference) - set(config)
    if unknown or missing:
        raise ValueError(f"EXP-006 config wrong keys: extra={sorted(unknown)}, missing={sorted(missing)}")
    if config["tuning_population"] not in ("all", "eligible_proxy"):
        raise ValueError("tuning_population must be all or eligible_proxy")
    for key in ("logistic_C", "hgb_learning_rate"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("threshold_passes", "logistic_max_iter", "hgb_max_iter",
                "hgb_max_leaf_nodes", "hgb_min_samples_leaf"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def hash_config(config: dict, names: list[str]) -> str:
    payload = {"config": config, "models": sorted(names)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _tw_date(v: str) -> pd.Timestamp:
    result = pd.Timestamp(v)
    return result.tz_localize("Asia/Taipei") if result.tzinfo is None else result.tz_convert("Asia/Taipei")


def select_time_slices(datasets: dict, audit: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    val = datasets["validation"]
    taiwan_dates = val.cutoff_utc.dt.tz_convert("Asia/Taipei")
    a0 = _tw_date(cfg["calibration_start_taiwan"])
    a1 = _tw_date(cfg["calibration_end_exclusive_taiwan"])
    r0 = _tw_date(cfg["report_start_taiwan"])
    r1 = _tw_date(cfg["report_end_exclusive_taiwan"])
    if not a0 < a1 <= r0 < r1:
        raise ValueError("Require calibration_start < calibration_end <= report_start < report_end")
    cal = val[(taiwan_dates >= a0) & (taiwan_dates < a1)].copy()
    report = val[(taiwan_dates >= r0) & (taiwan_dates < r1)].copy()
    if cal.empty or report.empty:
        raise ValueError("No rows in calibration or primary validation period; inspect EXP-005 run and config")
    cal_weeks = sorted(pd.to_datetime(cal.cutoff_utc.unique(), utc=True))
    report_weeks = sorted(pd.to_datetime(report.cutoff_utc.unique(), utc=True))
    first_cal = cal_weeks[0]
    first_report = report_weeks[0]
    train_audit = audit[audit.split == "train"]
    if train_audit.empty or train_audit.label_as_of_utc.max() > first_cal:
        raise ValueError("Training labels are not mature before threshold-calibration window begins")
    cal_audit = audit[audit.cutoff_utc.isin(cal_weeks)]
    if len(cal_audit) != len(cal_weeks):
        raise ValueError("Missing label maturity audit rows for calibration weeks")
    latest_cal_maturity = cal_audit.label_as_of_utc.max()
    if latest_cal_maturity > first_report:
        raise ValueError(
            f"Time leakage: calibration labels mature {latest_cal_maturity}, "
            f"but primary reporting starts {first_report}"
        )
    # Do not claim untouched temporal validation on target weeks that overlap the
    # official Sep27 example used to pick the labeling definition.
    report_audit = audit[audit.cutoff_utc.isin(report_weeks)]
    if len(report_audit) != len(report_weeks):
        raise ValueError("Missing primary validation label audit rows")
    for row in report_audit.itertuples():
        if row.cutoff_utc < EXAMPLE_END and row.label_end_utc > EXAMPLE_START:
            raise ValueError(
                f"Primary validation {row.cutoff_utc} overlaps Sep27 official example; "
                "select earlier weeks for clean evaluation"
            )
    holdout = audit[audit.split == "holdout"]
    if not holdout.empty and report_audit.label_as_of_utc.max() > holdout.cutoff_utc.min():
        raise ValueError("Primary validation labels extend beyond sealed holdout cutoff")
    unused = sorted(set(pd.to_datetime(val.cutoff_utc.unique(), utc=True)) - set(cal_weeks) - set(report_weeks))
    return {"train": datasets["train"], "calibration": cal.reset_index(drop=True),
            "primary": report.reset_index(drop=True)}, {
        "training_weeks": int(datasets["train"].cutoff_utc.nunique()),
        "calibration_weeks": [str(x) for x in cal_weeks],
        "primary_validation_weeks": [str(x) for x in report_weeks],
        "excluded_validation_weeks": [str(x) for x in unused],
        "calibration_latest_label_as_of": str(latest_cal_maturity),
        "primary_first_cutoff": str(first_report),
        "calibrated_example_interval_utc": [str(EXAMPLE_START), str(EXAMPLE_END)],
        "sealed_holdout": True,
    }


