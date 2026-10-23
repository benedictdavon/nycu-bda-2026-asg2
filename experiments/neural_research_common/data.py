"""Source-aware campaign inputs built only through the accepted F0/label builders.

The adapter owns the new-source cache boundary and clock/population protocol. It
does not modify the historical builders or seed from prior experiment artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.baseline_models import feature_names, feature_matrix
from src.data_loader import OfficialData, SPECS, TABLES, TARGETS, load_official_evaluator, utc
from src.decision_policies import (
    Timing, decision_time, feature_cutoff, population_anchor, roster_population,
    training_origins,
)
from src.features import FeatureBuilder
from src.labels import LabelBuilder

CAMPAIGN_ID = "neural_refresh_20261022T0033Z"
SOURCE_SHA = "2053302ba75751be975b9e779f567557b66f4071"
OLD_SOURCE_SHA = "506d860693df60a698b68aa63a8edb4a984c8e1b"
TARGET_ORDER = tuple(TARGETS)
REPORT_ORIGINS = tuple(pd.Timestamp(x) for x in (
    "2026-09-09T16:00:00Z", "2026-09-16T16:00:00Z",
    "2026-09-23T16:00:00Z", "2026-09-30T16:00:00Z",
))
TUNING_ORIGINS = tuple(pd.Timestamp(x) for x in (
    "2026-08-19T16:00:00Z", "2026-08-26T16:00:00Z",
))
CONTINUITY_ORIGIN = pd.Timestamp("2026-09-02T16:00:00Z")
TIMING = Timing(train_window_days=56)
FEATURE_CONFIG = {
    "windows": [7, 30, 90], "eligibility_days": 90,
    "eligibility_min_rows": 5, "missing_recency_days": -1.0,
}
EXPECTED_F0_WIDTH = 59
BINARY_COLUMNS = (
    "user_observed_at_cutoff", "open_pr_active_prior_7d", "open_issue_active_prior_7d",
    "issue_comment_active_prior_7d", "pr_review_active_prior_7d",
    "pr_review_comment_active_prior_7d", "push_active_prior_7d", "eligible_proxy",
)
HGB_CONFIG = {
    "learning_rate": 0.07, "max_iter": 80, "max_leaf_nodes": 15,
    "min_samples_leaf": 75, "l2_regularization": 3.0,
    "early_stopping": False, "random_state": 42,
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def ids_digest(user_ids: Iterable[int]) -> str:
    values = np.asarray(sorted(int(v) for v in user_ids), dtype="<i8")
    return sha256_bytes(values.tobytes(order="C"))


def frame_digest(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    columns = list(columns)
    selected = frame.loc[:, columns]
    hashed = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    return sha256_bytes(hashed.to_numpy(dtype="uint64", copy=False).tobytes())


def iso_utc(value) -> str:
    return utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def hgb_weight_vector(report_origin, origins, rows_per_origin) -> np.ndarray:
    """Repeat H2 origin weights over actual rows and normalize once to mean one."""
    S = utc(report_origin)
    ages_weeks = np.asarray([(S - utc(s)).total_seconds() / (7 * 86400) for s in origins], dtype=np.float64)
    per_origin = 0.5 ** (ages_weeks / 2.0)
    counts = np.asarray(rows_per_origin, dtype=np.int64)
    if counts.shape != per_origin.shape or (counts <= 0).any():
        raise ValueError("Every H2 fitting origin must have a positive eligible row count")
    weights = np.repeat(per_origin, counts)
    if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("H2 row weights must be finite and positive")
    weights /= weights.mean(dtype=np.float64)
    return weights


def report_train_origins(report_origin) -> list[pd.Timestamp]:
    result = training_origins(report_origin, timing=TIMING, window_days=56)
    expected = [utc(report_origin) - pd.Timedelta(days=i) for i in range(63, 7, -1)]
    if result != expected or len(result) != 56:
        raise AssertionError(f"Unexpected outer training schedule at {iso_utc(report_origin)}")
    return result


def inner_split(report_origin) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp], pd.Timestamp]:
    S = utc(report_origin)
    validation = [S - pd.Timedelta(days=i) for i in range(14, 7, -1)]
    V = decision_time(S - pd.Timedelta(days=14), TIMING)
    fit = training_origins(S, timing=TIMING, limit=V, window_days=56)
    purged = [S - pd.Timedelta(days=i) for i in range(21, 14, -1)]
    expected_fit = [S - pd.Timedelta(days=i) for i in range(63, 21, -1)]
    if fit != expected_fit or len(fit) != 42 or len(validation) != 7 or len(purged) != 7:
        raise AssertionError(f"Unexpected purged inner split at {iso_utc(S)}")
    if any(s + pd.Timedelta(days=7) > V for s in fit):
        raise AssertionError("An inner-fit label horizon crosses the permitted validation time")
    return fit, purged, validation, V


def input_origins() -> list[pd.Timestamp]:
    origins: set[pd.Timestamp] = set()
    for S in REPORT_ORIGINS:
        origins.update(report_train_origins(S))
        origins.add(utc(S))
    for S in TUNING_ORIGINS:
        fit, _, validation, _ = inner_split(S)
        origins.update(fit)
        origins.update(validation)
    return sorted(origins)


def label_requests(latest_observed) -> dict[tuple[str, str], dict]:
    """Map each exact (origin, as-of) label cache to all registered uses."""
    requests: dict[tuple[str, str], dict] = {}

    def add(s, as_of, purpose):
        key = (iso_utc(s), iso_utc(as_of))
        row = requests.setdefault(key, {"origin": utc(s), "label_as_of": utc(as_of), "uses": []})
        if purpose not in row["uses"]:
            row["uses"].append(purpose)

    for S in REPORT_ORIGINS:
        A = decision_time(S, TIMING)
        for s in report_train_origins(S):
            add(s, A, f"outer_train:{iso_utc(S)}")
        fit, _, validation, V = inner_split(S)
        for s in fit:
            add(s, V, f"outer_inner_fit:{iso_utc(S)}")
        for s in validation:
            add(s, A, f"outer_inner_validation:{iso_utc(S)}")
        add(S, latest_observed, "outer_report_label")

    for S in TUNING_ORIGINS:
        A = decision_time(S, TIMING)
        fit, _, validation, V = inner_split(S)
        for s in fit:
            add(s, V, f"tuning_inner_fit:{iso_utc(S)}")
        for s in validation:
            add(s, A, f"tuning_inner_validation:{iso_utc(S)}")
    return requests


@dataclass
class CacheContext:
    root: Path
    source_sha: str
    table_hashes: dict[str, str]
    builder_hashes: dict[str, str]
    feature_config: dict

    def key(self, kind: str, identity: dict) -> tuple[str, dict]:
        payload = {
            "cache_version": 1, "kind": kind,
            "source_sha": self.source_sha, "table_sha256": self.table_hashes,
            "builder_sha256": self.builder_hashes,
            "builder_config": self.feature_config,
            **identity,
        }
        return sha256_json(payload), payload

    def load(self, kind: str, key: str, identity: dict, columns: list[str]):
        stem = self.root / kind / key
        frame_path, manifest_path = stem.with_suffix(".parquet"), stem.with_suffix(".json")
        if not frame_path.exists() and not manifest_path.exists():
            return None
        if not frame_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"Incomplete cache pair; preserve and inspect {stem}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, expected = self.key(kind, identity)
        if manifest.get("identity") != expected:
            raise RuntimeError(f"Cache identity mismatch at {stem}; refusing stale reuse")
        if manifest.get("frame_sha256") != sha256_file(frame_path):
            raise RuntimeError(f"Cache content hash mismatch at {frame_path}")
        frame = pd.read_parquet(frame_path)
        if list(frame.columns) != columns:
            raise RuntimeError(f"Cache ordered columns mismatch at {frame_path}")
        if len(frame) != manifest.get("rows"):
            raise RuntimeError(f"Cache row count mismatch at {frame_path}")
        return frame, manifest

    def save(self, kind: str, key: str, identity: dict, frame: pd.DataFrame,
             *, summary: dict | None = None) -> tuple[Path, dict]:
        stem = self.root / kind / key
        stem.parent.mkdir(parents=True, exist_ok=True)
        frame_path, manifest_path = stem.with_suffix(".parquet"), stem.with_suffix(".json")
        if frame_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing cache artifact {stem}")
        frame.to_parquet(frame_path, engine="pyarrow", index=False)
        _, payload = self.key(kind, identity)
        manifest = {
            "identity": payload, "rows": int(len(frame)),
            "ordered_columns": list(frame.columns),
            "frame_sha256": sha256_file(frame_path), "summary": summary or {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return frame_path, manifest


def validate_source(data: OfficialData, manifest_path: Path) -> dict[str, str]:
    if data.snapshot_sha != SOURCE_SHA:
        raise RuntimeError(f"Expected refreshed source {SOURCE_SHA}, got {data.snapshot_sha}")
    expected_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_manifest.get("source_sha") != SOURCE_SHA:
        raise RuntimeError("Campaign source manifest has an unexpected source SHA")
    hashes = {}
    for table in TABLES:
        path = data.data_dir / f"{table}.parquet"
        got = sha256_file(path)
        expected = expected_manifest["table_sha256"].get(table)
        if got != expected:
            raise RuntimeError(f"Source table mismatch for {table}: expected {expected}, got {got}")
        hashes[table] = got
    evaluator = load_official_evaluator(data.official_repo)
    if tuple(evaluator.EVENT_COLS) != TARGET_ORDER:
        raise RuntimeError("Active official evaluator target order differs from the frozen target order")
    return hashes


def builder_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/data_loader.py", "src/features.py", "src/labels.py",
        "src/decision_policies.py", "src/baseline_models.py",
        "src/phase_b_rules.py", "experiments/neural_research_common/data.py",
        "experiments/neural_research_common/build_data.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def feature_cache_identity(origin, population, columns) -> dict:
    C = feature_cutoff(origin, TIMING)
    B = population_anchor(origin, TIMING)
    return {
        "origin_utc": iso_utc(origin), "feature_cutoff_utc": iso_utc(C),
        "population_anchor_utc": iso_utc(B), "population_sha256": ids_digest(population),
        "ordered_model_columns": list(columns),
        "population_rule": {"window_days": 90, "min_rows": 5},
    }


def label_cache_identity(origin, label_as_of, population, targets=TARGET_ORDER) -> dict:
    return {
        "origin_utc": iso_utc(origin), "target_start_utc": iso_utc(origin),
        "target_end_utc": iso_utc(utc(origin) + pd.Timedelta(days=7)),
        "label_as_of_utc": iso_utc(label_as_of),
        "population_sha256": ids_digest(population),
        "ordered_target_columns": list(targets),
    }


def assert_feature_schema(frame: pd.DataFrame, expected: list[str] | None = None) -> list[str]:
    columns = feature_names(frame)
    if len(columns) != EXPECTED_F0_WIDTH:
        raise RuntimeError(f"Expected {EXPECTED_F0_WIDTH} F0 columns, found {len(columns)}")
    if expected is not None and columns != expected:
        raise RuntimeError("Ordered F0 columns changed during the source-aware build")
    if len(set(columns)) != len(columns):
        raise RuntimeError("F0 columns must be unique")
    if not set(BINARY_COLUMNS).issubset(columns):
        raise RuntimeError("Registered binary columns are absent from F0")
    values = feature_matrix(frame, columns)
    for column in BINARY_COLUMNS:
        if not np.isin(frame[column].to_numpy(), [0, 1]).all():
            raise RuntimeError(f"Binary F0 column {column} contains values outside {{0,1}}")
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite F0 value in source build")
    return columns


def construct_feature_builder(source: OfficialData) -> FeatureBuilder:
    return FeatureBuilder(
        source, windows=tuple(FEATURE_CONFIG["windows"]),
        eligibility_days=FEATURE_CONFIG["eligibility_days"],
        eligibility_min_rows=FEATURE_CONFIG["eligibility_min_rows"],
        missing_recency_days=FEATURE_CONFIG["missing_recency_days"],
    )


def origin_population(source: OfficialData, origin) -> pd.Index:
    events = {target: source.events(target) for target in TARGET_ORDER}
    return roster_population(
        events, population_anchor(origin, TIMING), source.cohort_ids,
        window_days=90, min_rows=5,
    )

