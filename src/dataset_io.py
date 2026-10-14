"""Load and validate historical dataset artifacts without reading sealed holdout labels."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .data_loader import TARGETS
from .baseline_models import feature_names
from .paths import resolve_run_path

KEYS = ["UserID", "cutoff_utc"]

def find_run(base: str | Path, *, explicit: str | Path | None = None) -> Path:
    if explicit:
        chosen = resolve_run_path(explicit)
    else:
        root = resolve_run_path(base)
        options = []
        for path in root.glob("build-*/manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("experiment") == "EXP-005" and manifest.get("mode") == "build":
                    options.append((pd.Timestamp(manifest["created_utc"]), path.parent))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if not options:
            raise FileNotFoundError(f"No EXP-005 build manifests in {root}; run --mode build first")
        chosen = sorted(options, key=lambda item: (item[0], str(item[1])))[-1][1]
    if not (chosen / "manifest.json").is_file():
        raise FileNotFoundError(f"EXP-005 manifest missing at {chosen}")
    return chosen


def load_dataset_run(path: Path) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    path = resolve_run_path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != "EXP-005" or manifest.get("mode") != "build":
        raise ValueError("Not an EXP-005 historical build")
    if manifest.get("details", {}).get("holdout_unsealed"):
        raise ValueError("Refusing EXP-005 run that intentionally unsealed historical holdout")
    if (path / "holdout_labels.parquet").exists():
        raise ValueError("Refusing run with unsealed holdout labels (do not peek)")
    required = ["train_features.parquet", "train_labels.parquet",
                "validation_features.parquet", "validation_labels.parquet", "label_audit.csv"]
    for file in required:
        if not (path / file).is_file():
            raise FileNotFoundError(f"Missing EXP-005 file: {path / file}")
    datasets: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation"):
        feature_frame = pd.read_parquet(path / f"{split}_features.parquet")
        label_frame = pd.read_parquet(path / f"{split}_labels.parquet")
        for f in (feature_frame, label_frame):
            if not set(KEYS).issubset(f.columns):
                raise ValueError(f"Missing user/cutoff columns in {split}")
            f["cutoff_utc"] = pd.to_datetime(f["cutoff_utc"], utc=True)
            if f[KEYS].isna().any().any() or f.duplicated(KEYS).any():
                raise ValueError(f"Null/duplicate EXP-005 feature or label keys in {split}")
        if set(label_frame.columns) != set(KEYS + list(TARGETS)):
            raise ValueError(f"Unexpected {split}_labels.parquet schema")
        if any(t in feature_frame for t in TARGETS):
            raise ValueError(f"Target label leaked into {split} features")
        if len(feature_frame) != len(label_frame):
            raise ValueError(f"EXP-005 row count mismatch for {split}")
        # Merge by unique (UserID, cutoff_utc) to prevent accidental label misalignment.
        joined = feature_frame.merge(label_frame, on=KEYS, how="inner", validate="one_to_one",
                                     sort=False)
        if len(joined) != len(feature_frame):
            raise ValueError(f"Unmatched EXP-005 feature/label keys in {split}")
        if not joined[list(TARGETS)].isin([0, 1]).all().all():
            raise ValueError("Non-binary labels")
        feature_names(feature_frame)
        datasets[split] = joined.sort_values(["cutoff_utc", "UserID"], kind="stable").reset_index(drop=True)
    if set(datasets["train"].cutoff_utc.unique()) & set(datasets["validation"].cutoff_utc.unique()):
        raise ValueError("Train and validation weeks overlap")
    audit = pd.read_csv(path / "label_audit.csv")
    for col in ("cutoff_utc", "label_end_utc", "label_as_of_utc"):
        audit[col] = pd.to_datetime(audit[col], utc=True)
    if audit.duplicated(["cutoff_utc"]).any():
        raise ValueError("Duplicate label audit cutoffs")
    return manifest, datasets, audit


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.drop(columns=list(TARGETS)), df[list(TARGETS)]
