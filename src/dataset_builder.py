"""Reusable historical and inference dataset construction with sealed holdout support."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .data_loader import OfficialData, TARGETS, locate_data_dir, utc
from .features import FeatureBuilder
from .labels import LabelBuilder
from .temporal_splits import SplitConfig, build_calendar

DEFAULTS = {
    "first_cutoff_taiwan": "2026-03-02",
    "validation_start_taiwan": "2026-08-03",
    "holdout_start_taiwan": "2026-10-12",
    "last_cutoff_taiwan": None,
    "horizon_days": 7,
    "label_maturation_days": 7,
    "purge_boundary_maturity": True,
    "lookback_days": [7, 30, 90],
    "eligibility_days": 90,
    "eligibility_min_records": 5,
}


def load_config(path: str | Path | None) -> dict:
    values = DEFAULTS.copy()
    if path is not None:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unsupported config keys: {sorted(unknown)}")
        values.update(user)
    return values


def make_features(source: OfficialData, cfg: dict) -> FeatureBuilder:
    return FeatureBuilder(source, windows=tuple(cfg["lookback_days"]),
                          eligibility_days=cfg["eligibility_days"],
                          eligibility_min_rows=cfg["eligibility_min_records"])


def _hash_config(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def dataset_run_dir(root: Path, source: OfficialData, cfg: dict,
                mode: str, base: str | Path | None) -> Path:
    prefix = source.snapshot_sha[:12] if source.snapshot_sha else "unknownsha"
    tag = f"{mode}-{prefix}-{_hash_config(cfg)}"
    base = Path(base) if base is not None else root / "experiments/exp005_temporal_dataset/runs"
    return base / tag


def _write_parquet(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)


def _write_manifest(path: Path, *, source: OfficialData, config: dict,
                    mode: str, files: dict, warnings: list[str], details: dict):
    payload = {
        "experiment": "EXP-005", "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source.fingerprint(), "config": config,
        "files": files, "warnings": warnings, "details": details,
        "not_a_model_score": True,
    }
    (path / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def build_historical(source: OfficialData, cfg: dict, *, output_dir: Path,
                     unseal_holdout: bool = False, overwrite: bool = False) -> dict:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}. Pass --overwrite intentionally.")
    split_cfg = SplitConfig(**{k: cfg[k] for k in SplitConfig.__dataclass_fields__})
    calendar = build_calendar(source.latest_observed(), split_cfg, unseal_holdout=unseal_holdout)
    if calendar.empty:
        raise ValueError("No historical cutoff windows available; inspect configuration or snapshot freshness")
    fb = make_features(source, cfg)
    lb = LabelBuilder(source)
    fs: dict[str, list[pd.DataFrame]] = {k: [] for k in ("train", "validation", "holdout")}
    ys: dict[str, list[pd.DataFrame]] = {k: [] for k in ("train", "validation", "holdout")}
    audit = []
    for row in calendar.itertuples(index=False):
        feat = fb.build(row.cutoff_utc)
        fs[row.split].append(feat)
        if row.split != "holdout" or unseal_holdout:
            label = lb.build(row.cutoff_utc, horizon_days=cfg["horizon_days"],
                             label_as_of=row.label_as_of_utc, user_ids=pd.Index(feat.UserID))
            ys[row.split].append(label)
            prevalences = {f"positives_{k}": int(label[k].sum()) for k in TARGETS}
        else:
            prevalences = {}
        audit.append({
            "split": row.split, "cutoff_utc": row.cutoff_utc,
            "label_end_utc": row.label_end_utc, "label_as_of_utc": row.label_as_of_utc,
            "users": len(feat), "eligible_proxy_users": int(feat.eligible_proxy.sum()),
            **prevalences,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    for split in ("train", "validation", "holdout"):
        if not fs[split]:
            continue
        features = pd.concat(fs[split], ignore_index=True)
        if features.duplicated(["UserID", "cutoff_utc"]).any():
            raise AssertionError("Duplicate (UserID, cutoff_utc) feature keys")
        feature_name = f"{split}_features.parquet"
        _write_parquet(features, output_dir / feature_name)
        generated[feature_name] = len(features)
        if ys[split]:
            labels = pd.concat(ys[split], ignore_index=True)
            if labels.duplicated(["UserID", "cutoff_utc"]).any():
                raise AssertionError("Duplicate label keys")
            if not labels[["UserID", "cutoff_utc"]].equals(features[["UserID", "cutoff_utc"]]):
                raise AssertionError("Feature and label keys are not aligned")
            if not labels[list(TARGETS)].isin([0, 1]).all().all():
                raise AssertionError("Nonbinary or missing labels")
            label_name = f"{split}_labels.parquet"
            _write_parquet(labels, output_dir / label_name)
            generated[label_name] = len(labels)
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(output_dir / "label_audit.csv", index=False)
    generated["label_audit.csv"] = len(audit_frame)
    if "holdout_labels.parquet" in generated and not unseal_holdout:
        raise AssertionError("Holdout protection violated")
    warnings = [
        "label_as_of is a configurable reconstruction policy, not certified ingestion completeness",
        "For direct events (push/comments), seen_utc equals created_at: actual ingestion time is not in the dataset",
        "Historical cohort membership uses the retrospective fixed 9,000-user cohort",
        "Eligible proxy (>=5 rows / 90d) may not match the official Oct 28 test roster",
        "The Sep 27 example truth was used to calibrate labels and is not an independent holdout",
        "Latest observed row timestamp is not a certified collection-completeness watermark",
    ]
    if not unseal_holdout:
        warnings.append("Historical holdout labels intentionally NOT constructed or exported")
    _write_manifest(output_dir, source=source, config=cfg, mode="build", files=generated,
                    warnings=warnings, details={
                        "windows_per_split": audit_frame.split.value_counts().to_dict(),
                        "first_cutoff_utc": str(audit_frame.cutoff_utc.iloc[0]),
                        "last_cutoff_utc": str(audit_frame.cutoff_utc.iloc[-1]),
                        "holdout_unsealed": unseal_holdout,
                    })
    return {"output_dir": str(output_dir), "files": generated,
            "weeks": audit_frame.split.value_counts().to_dict()}


def build_inference(source: OfficialData, cfg: dict, *, output_dir: Path,
                    cutoff, users_csv=None, overwrite=False) -> dict:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}. Pass --overwrite intentionally.")
    ids = source.official_test_user_ids(users_csv)
    t = utc(cutoff)
    features = make_features(source, cfg).build(t, user_ids=ids)
    if len(features) != len(ids) or set(features.UserID) != set(ids):
        raise AssertionError("Inference features do not exactly cover requested test users")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(features, output_dir / "inference_features.parquet")
    warnings = [
        "NO prediction CSV is generated: this module builds features only",
        "Missing observations between latest official snapshot and cutoff may not be recoverable",
        "Use final instructor snapshot, then train model and run inference separately",
    ]
    _write_manifest(output_dir, source=source, config=cfg, mode="inference",
                    files={"inference_features.parquet": len(features)}, warnings=warnings,
                    details={"cutoff_utc": str(t), "users": len(ids), "test_input": str(users_csv or "official")})
    return {"output_dir": str(output_dir), "users": len(ids), "cutoff_utc": str(t)}
