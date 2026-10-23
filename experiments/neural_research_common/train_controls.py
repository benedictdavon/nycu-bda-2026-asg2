"""Fit fresh refreshed-source REF/H2 controls and score fixed P016/P021 policies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.data_loader import OfficialData, TABLES, TARGETS, load_official_evaluator, utc
from src.decision_policies import activity_flags, decision_time, feature_cutoff
from src.metrics import confusion_arrays, scores_from_counts
from src.phase_b_rules import blend_score, ew_scores, topk_at_persist
from .data import (
    CAMPAIGN_ID, FEATURE_CONFIG, HGB_CONFIG, REPORT_ORIGINS, TARGET_ORDER, TIMING,
    frame_digest, hgb_weight_vector, ids_digest, iso_utc, sha256_file, sha256_json,
    validate_source,
)

ROOT = Path(__file__).resolve().parents[2]
OLD_SOURCE_SHA = "506d860693df60a698b68aa63a8edb4a984c8e1b"
SOURCE_SHA = "2053302ba75751be975b9e779f567557b66f4071"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, path)


def _runtime_info() -> dict:
    import numpy
    import pandas
    import pyarrow
    import scipy
    import threadpoolctl
    threadpools = None
    threadpool_info_error = None
    try:
        threadpools = threadpoolctl.threadpool_info()
    except OSError as exc:
        # Some Windows BLAS/OpenMP combinations load correctly for sklearn but
        # fail when threadpoolctl queries their native thread-count symbol.
        # Preserve that diagnostic without blocking the estimator fit.
        threadpool_info_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "numpy": numpy.__version__,
        "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__, "threadpools": threadpools,
        "threadpool_info_error": threadpool_info_error,
    }


def _load_inputs(run_dir: Path):
    build = run_dir / "data_build"
    feature_manifest = json.loads((build / "feature_inputs_manifest.json").read_text(encoding="utf-8"))
    label_manifest = json.loads((build / "label_inputs_manifest.json").read_text(encoding="utf-8"))
    f0_schema_path = ROOT / "reports" / CAMPAIGN_ID / "f0_schema.json"
    f0_schema = json.loads(f0_schema_path.read_text(encoding="utf-8"))
    if feature_manifest["source_sha"] != SOURCE_SHA or label_manifest["source_sha"] != SOURCE_SHA:
        raise RuntimeError("Research inputs do not all use the active refreshed source")
    if feature_manifest["table_sha256"] != label_manifest["table_sha256"]:
        raise RuntimeError("Feature and label inputs were built from different source table hashes")
    builder_hashes = feature_manifest["builder_sha256"]
    for relpath, expected_sha in builder_hashes.items():
        if relpath == "experiments/neural_research_common/build_data.py":
            code_path = ROOT / "reports" / CAMPAIGN_ID / "build_data_executed_v1.py"
        else:
            code_path = ROOT / relpath
        if not code_path.is_file() or sha256_file(code_path) != expected_sha:
            raise RuntimeError(f"Authenticated input-builder source changed: {relpath}")
    f0 = feature_manifest["f0_columns"]
    if f0 != f0_schema["ordered_columns"] or len(f0) != 59:
        raise RuntimeError("F0 schema/order differs from its frozen source manifest")
    feature_by_origin = {row["origin_utc"]: ROOT / row["cache_path"]
                         for row in feature_manifest["origins"]}
    labels_by_key = {(row["origin_utc"], row["label_as_of_utc"]): ROOT / row["cache_path"]
                     for row in label_manifest["requests"]}
    for manifest, rows, expected_columns, kind in (
        (feature_manifest, feature_manifest["origins"], ["UserID", "cutoff_utc", *f0], "features"),
        (label_manifest, label_manifest["requests"], ["UserID", "cutoff_utc", *TARGET_ORDER], "labels"),
    ):
        if manifest["source_sha"] != SOURCE_SHA:
            raise RuntimeError(f"{kind} manifest source SHA mismatch")
        for row in rows:
            path = ROOT / row["cache_path"]
            key = row["f0_cache_key"] if kind == "features" else row["label_cache_key"]
            sidecar = path.with_suffix(".json")
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"Missing authenticated {kind} cache: {path}")
            if sha256_file(sidecar) != row["feature_manifest_sha256" if kind == "features" else "label_manifest_sha256"]:
                raise RuntimeError(f"{kind} cache manifest digest mismatch: {sidecar}")
            cache_meta = json.loads(sidecar.read_text(encoding="utf-8"))
            identity = cache_meta.get("identity", {})
            if identity.get("source_sha") != SOURCE_SHA or identity.get("kind") != kind:
                raise RuntimeError(f"{kind} cache source/kind identity mismatch: {path}")
            if identity.get("table_sha256") != manifest["table_sha256"]:
                raise RuntimeError(f"{kind} cache table-hash mismatch: {path}")
            if identity.get("builder_sha256") != manifest["builder_sha256"]:
                raise RuntimeError(f"{kind} cache builder-code hash mismatch: {path}")
            if identity.get("builder_config") != FEATURE_CONFIG:
                raise RuntimeError(f"{kind} cache builder-config mismatch: {path}")
            expected_origin = row["origin_utc"]
            if identity.get("origin_utc") != expected_origin:
                raise RuntimeError(f"{kind} cache origin mismatch: {path}")
            if identity.get("population_sha256") != row.get("population_sha256"):
                raise RuntimeError(f"{kind} cache population identity mismatch: {path}")
            if kind == "labels" and identity.get("label_as_of_utc") != row["label_as_of_utc"]:
                raise RuntimeError(f"Label cache as-of identity mismatch: {path}")
            if path.stem != key or cache_meta.get("frame_sha256") != sha256_file(path):
                raise RuntimeError(f"{kind} cache key/content authentication failed: {path}")
            frame = pd.read_parquet(path)
            if (list(frame.columns) != expected_columns
                    or cache_meta.get("ordered_columns") != expected_columns
                    or len(frame) != row.get("users", row.get("population_rows"))):
                raise RuntimeError(f"{kind} cache schema/row count mismatch: {path}")
            if ids_digest(frame.UserID.to_numpy(dtype=np.int64)) != row["population_sha256"]:
                raise RuntimeError(f"{kind} cache stable-ID digest mismatch: {path}")
            if kind == "features":
                if identity.get("ordered_model_columns") != f0:
                    raise RuntimeError(f"Feature cache F0 order mismatch: {path}")
                if frame_digest(frame, f0) != row["f0_sha256"]:
                    raise RuntimeError(f"Feature cache value digest mismatch: {path}")
            else:
                if identity.get("ordered_target_columns") != list(TARGET_ORDER):
                    raise RuntimeError(f"Label cache target order mismatch: {path}")
                if frame_digest(frame, TARGET_ORDER) != cache_meta["summary"]["label_value_sha256"]:
                    raise RuntimeError(f"Label cache value digest mismatch: {path}")
    return feature_manifest, label_manifest, f0, feature_by_origin, labels_by_key


def _training_rows(S, f0, feature_by_origin, labels_by_key, A):
    origins = [utc(S) - pd.Timedelta(days=i) for i in range(63, 7, -1)]
    if len(origins) != 56:
        raise AssertionError("Outer fit must use 56 daily origins")
    x_blocks, y_blocks, row_counts = [], [], []
    key_rows = []
    for s in origins:
        origin = iso_utc(s)
        fpath = feature_by_origin.get(origin)
        lpath = labels_by_key.get((origin, iso_utc(A)))
        if fpath is None or lpath is None:
            raise FileNotFoundError(f"Missing authenticated outer input for {origin} at {iso_utc(A)}")
        features = pd.read_parquet(fpath)
        labels = pd.read_parquet(lpath)
        if not features.UserID.equals(labels.UserID):
            raise RuntimeError(f"Feature/label stable-ID mismatch at {origin}")
        if features.UserID.duplicated().any() or not features.UserID.is_monotonic_increasing:
            raise RuntimeError(f"Invalid feature key order at {origin}")
        x = features.loc[:, f0].to_numpy(dtype=np.float64, copy=True)
        y = labels.loc[:, TARGET_ORDER].to_numpy(dtype=np.int8, copy=True)
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise RuntimeError(f"Nonfinite features or nonbinary labels at {origin}")
        x_blocks.append(x)
        y_blocks.append(y)
        row_counts.append(len(features))
        key_rows.extend(zip([origin] * len(features), features.UserID.to_numpy(dtype=np.int64)))
    return origins, np.concatenate(x_blocks), np.concatenate(y_blocks), np.asarray(row_counts), key_rows


def _predict(model, x, target_index: int) -> np.ndarray:
    if isinstance(model, DummyClassifier):
        return np.full(len(x), float(model.constant), dtype=np.float64)
    classes = list(np.asarray(model.classes_).tolist())
    if 1 not in classes:
        return np.zeros(len(x), dtype=np.float64)
    return model.predict_proba(x)[:, classes.index(1)].astype(np.float64, copy=False)


def _fit_key(*, source_sha, report_origin, arm, target, x, y, weights,
             origin_keys, f0, runtime, implementation_hash) -> tuple[str, dict]:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    digest.update(np.ascontiguousarray(weights).tobytes())
    identity = {
        "source_sha": source_sha, "report_origin_utc": iso_utc(report_origin),
        "arm": arm, "target": target, "f0_order": list(f0),
        "input_values_sha256": digest.hexdigest(),
        "training_key_sha256": sha256_json(origin_keys),
        "hgb_config": HGB_CONFIG, "runtime": runtime,
        "implementation_sha256": implementation_hash,
    }
    return sha256_json(identity), identity


def _fit_or_resume(model_path: Path, manifest_path: Path, *, key: str, identity: dict,
                   x, y, weights, arm: str) -> tuple[object, dict]:
    if model_path.exists() or manifest_path.exists():
        if not model_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"Incomplete HGB target checkpoint pair: {model_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("fit_key") != key or saved.get("identity") != identity:
            raise RuntimeError(f"HGB checkpoint identity mismatch; refusing reuse: {model_path}")
        if saved.get("model_sha256") != sha256_file(model_path):
            raise RuntimeError(f"HGB checkpoint hash mismatch: {model_path}")
        return joblib.load(model_path), {**saved, "checkpoint_status": "authenticated_resume"}

    started = time.perf_counter()
    labels = np.unique(y)
    if len(labels) == 1:
        model = DummyClassifier(strategy="constant", constant=int(labels[0])).fit(x, y)
        estimator_name = "DummyClassifier(constant)"
        estimator_params = model.get_params(deep=True)
    else:
        model = HistGradientBoostingClassifier(**HGB_CONFIG)
        estimator_name = "HistGradientBoostingClassifier"
        estimator_params = model.get_params(deep=True)
        sample_weight = None if arm == "REF" else weights
        model.fit(x, y, sample_weight=sample_weight)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_name(model_path.name + f".{os.getpid()}.tmp")
    joblib.dump(model, temp, compress=3)
    os.replace(temp, model_path)
    info = {
        "fit_key": key, "identity": identity, "model_sha256": sha256_file(model_path),
        "checkpoint_status": "new_fit", "arm": arm, "estimator": estimator_name,
        "estimator_params": estimator_params, "label_values": labels.tolist(),
        "positive_count": int(y.sum()), "row_count": int(len(y)),
        "sample_weight_used": arm == "H2" and len(labels) > 1,
        "sample_weight_min": float(weights.min()) if arm == "H2" else None,
        "sample_weight_max": float(weights.max()) if arm == "H2" else None,
        "sample_weight_mean": float(weights.mean()) if arm == "H2" else None,
        "iterations": int(getattr(model, "n_iter_", 0)),
        "fit_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, info)
    return model, info


def _score_counts(y, prediction) -> dict[str, np.ndarray]:
    return confusion_arrays(np.asarray(prediction, dtype=np.int8), np.asarray(y, dtype=np.int8))


def _f1_per_target(counts) -> np.ndarray:
    tp, fp, fn = (np.asarray(counts[k], dtype=np.float64) for k in ("tp", "fp", "fn"))
    denominator = 2 * tp + fp + fn
    values = np.ones_like(denominator, dtype=np.float64)
    np.divide(2 * tp, denominator, out=values, where=denominator > 0)
    return values


def _official_score(official, uids, y, decisions) -> dict:
    truth = pd.DataFrame(np.asarray(y, dtype=np.int8), columns=TARGET_ORDER)
    truth.insert(0, "UserID", np.asarray(uids, dtype=np.int64))
    prediction = pd.DataFrame(np.asarray(decisions, dtype=np.int8), columns=TARGET_ORDER)
    prediction.insert(0, "UserID", np.asarray(uids, dtype=np.int64))
    return {k: float(v) for k, v in official.scores(prediction, truth).items()
            if k in {"macro_f1", "micro_f1", "score"}}


def train(*, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    feature_manifest, label_manifest, f0, feature_by_origin, labels_by_key = _load_inputs(run_dir)
    if feature_manifest["source_sha"] != SOURCE_SHA:
        raise RuntimeError("Refusing to train a control against an unexpected source")
    source = OfficialData(ROOT / "assignment2/data/db", expected_cohort=9000)
    if source.snapshot_sha != SOURCE_SHA:
        raise RuntimeError("Active source SHA changed before control training")
    current_hashes = validate_source(source, ROOT / "reports" / CAMPAIGN_ID / "source_manifest.json")
    if current_hashes != feature_manifest["table_sha256"]:
        raise RuntimeError("Active source table hashes changed after data/cache build")
    official = load_official_evaluator(source.official_repo)
    latest_observed = source.latest_observed()
    if iso_utc(latest_observed) != run_manifest_latest(run_dir):
        raise RuntimeError("The control runner and data-build report-label as-of differ")
    runtime = _runtime_info()
    estimator = HistGradientBoostingClassifier(**HGB_CONFIG)
    resolved_estimator_params = estimator.get_params(deep=True)
    control_root = run_dir / "controls"
    model_root = control_root / "models"
    control_root.mkdir(parents=True, exist_ok=True)
    trainer_process = {
        "pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "experiments.neural_research_common.train_controls",
                    "--run-dir", str(run_dir)],
        "working_directory": str(ROOT), "run_directory": str(control_root),
        "source_sha": SOURCE_SHA, "trainer_kind": "CPU HGB controls; target fits sequential",
    }
    _write_json(control_root / "trainer_process.json", trainer_process)
    implementation_hash = sha256_file(Path(__file__))
    all_predictions, fit_summaries = [], []
    progress = {"source_sha": SOURCE_SHA, "completed_origins": [], "current_origin": None,
                "current_target": None, "fit_count": 0}
    _write_json(control_root / "progress.json", progress)

    # Freeze the estimator recipe plus every resolved sklearn default before fits.
    _write_json(control_root / "config_resolved.json", {
        "source_sha": SOURCE_SHA, "table_sha256": feature_manifest["table_sha256"],
        "ordered_f0": f0, "hgb_registered_overrides": HGB_CONFIG,
        "hgb_resolved_estimator_defaults": resolved_estimator_params,
        "runtime": runtime,
        "h2_weight_formula": "0.5 ** (((S-s)/7d)/2), repeated over actual eligible rows and normalized once to mean 1",
        "ref_weighting": "none",
        "report_label_as_of_utc": iso_utc(latest_observed),
        "report_origins_utc": [iso_utc(s) for s in REPORT_ORIGINS],
        "continuity_origin_replay": "not_run; optional and cannot affect selection",
        "policy": {
            "persistence": "activity_flags over visible rows at C_s, lookback 7d",
            "P016": "raw p rank top-k with k=the P_persist7 positive count",
            "P021": "0.5 rank01(p)+0.5 rank01(EW7), top-k with the same k",
            "EW7": "tau=7d, lookback=90d, source visibility time",
        },
    })

    for report_index, S in enumerate(REPORT_ORIGINS, start=1):
        S_text = iso_utc(S)
        A = decision_time(S, TIMING)
        origins, x, y, row_counts, training_keys = _training_rows(
            S, f0, feature_by_origin, labels_by_key, A
        )
        weights = hgb_weight_vector(S, origins, row_counts)
        training_key_sha = sha256_json(training_keys)
        uids_frame = pd.read_parquet(feature_by_origin[S_text])
        uids = uids_frame.UserID.to_numpy(dtype=np.int64)
        if len(np.unique(uids)) != len(uids) or not np.all(uids[:-1] < uids[1:]):
            raise RuntimeError("Report population must contain unique ascending stable UserIDs")
        report_label_path = labels_by_key[(S_text, iso_utc(latest_observed))]
        y_report = pd.read_parquet(report_label_path).loc[:, TARGET_ORDER].to_numpy(dtype=np.int8)
        report_label_ids = pd.read_parquet(report_label_path).UserID.to_numpy(dtype=np.int64)
        if not np.array_equal(uids, report_label_ids):
            raise RuntimeError("Report F0/label keys are misaligned")
        pred_by_arm = {"REF": np.empty((len(uids), len(TARGET_ORDER))),
                       "H2": np.empty((len(uids), len(TARGET_ORDER)))}
        target_fit_info = {"REF": {}, "H2": {}}
        for arm in ("REF", "H2"):
            for j, target in enumerate(TARGET_ORDER):
                progress.update({"current_origin": S_text, "current_arm": arm,
                                 "current_target": target})
                _write_json(control_root / "progress.json", progress)
                y_train = y[:, j]
                target_identity = {
                    "training_origin_count": len(origins),
                    "origin_rows": list(zip([iso_utc(s) for s in origins], row_counts.tolist())),
                    "ordered_train_keys_sha256": training_key_sha,
                    "target_order": list(TARGET_ORDER),
                    "decision_time_utc": iso_utc(A),
                }
                key, fit_identity = _fit_key(
                    source_sha=SOURCE_SHA, report_origin=S, arm=arm, target=target,
                    x=x, y=y_train, weights=weights, origin_keys=target_identity,
                    f0=f0, runtime=runtime, implementation_hash=implementation_hash,
                )
                model_path = model_root / S_text.replace(":", "") / arm / f"{target}.joblib"
                manifest_path = model_path.with_suffix(".json")
                model, fit_info = _fit_or_resume(
                    model_path, manifest_path, key=key, identity=fit_identity,
                    x=x, y=y_train, weights=weights, arm=arm,
                )
                report_x = uids_frame.loc[:, f0].to_numpy(dtype=np.float64, copy=True)
                pred_by_arm[arm][:, j] = _predict(model, report_x, j)
                target_fit_info[arm][target] = {
                    **{k: fit_info[k] for k in (
                        "checkpoint_status", "estimator", "positive_count", "row_count",
                        "sample_weight_used", "sample_weight_min", "sample_weight_max",
                        "sample_weight_mean", "iterations", "fit_seconds", "model_sha256",
                    )},
                    "checkpoint_path": str(model_path.relative_to(ROOT)),
                    "checkpoint_manifest_path": str(manifest_path.relative_to(ROOT)),
                }
                progress["fit_count"] += 1
                _write_json(control_root / "progress.json", progress)
                print(f"control origin {report_index}/4 {S_text} {arm}/{target} "
                      f"rows={len(x)} seconds={fit_info.get('fit_seconds', 0):.1f} "
                      f"{fit_info['checkpoint_status']}", flush=True)

        C = feature_cutoff(S, TIMING)
        persist = activity_flags({t: source.events(t) for t in TARGET_ORDER}, C, uids, 7).astype(np.int8)
        ew = ew_scores({t: source.events(t) for t in TARGET_ORDER}, C, uids,
                       tau_days=7, window_days=90)
        if not np.array_equal(
            persist,
            uids_frame[[f"{t}_active_prior_7d" for t in TARGET_ORDER]].to_numpy(dtype=np.int8),
        ):
            raise RuntimeError(f"P_persist7 does not match F0 active_prior_7d at {S_text}")
        decisions = {"P_persist7": persist.copy()}
        for arm in ("REF", "H2"):
            p = pred_by_arm[arm]
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise RuntimeError(f"Invalid {arm} probabilities at {S_text}")
            decisions[f"{arm}_P016"] = topk_at_persist(p, persist, uids).astype(np.int8)
            decisions[f"{arm}_P021"] = topk_at_persist(blend_score(p, ew, 0.5), persist, uids).astype(np.int8)

        rows = pd.DataFrame({"origin": S_text, "UserID": uids})
        for j, target in enumerate(TARGET_ORDER):
            rows[f"y_{target}"] = y_report[:, j]
            rows[f"persist_{target}"] = persist[:, j]
            rows[f"ew7_{target}"] = ew[:, j]
            for arm in ("REF", "H2"):
                rows[f"p_{arm}_{target}"] = pred_by_arm[arm][:, j]
                rows[f"dec_{arm}_P016_{target}"] = decisions[f"{arm}_P016"][:, j]
                rows[f"dec_{arm}_P021_{target}"] = decisions[f"{arm}_P021"][:, j]
            rows[f"dec_P_persist7_{target}"] = persist[:, j]
        all_predictions.append(rows)
        fit_summaries.append({
            "origin_utc": S_text, "decision_time_utc": iso_utc(A),
            "training_origins": [iso_utc(s) for s in origins],
            "training_origin_count": len(origins), "training_rows": int(len(x)),
            "row_counts_by_origin": row_counts.tolist(),
            "training_key_sha256": sha256_json(training_keys),
            "training_label_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
            "h2_weight_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
            "h2_weight_mean": float(weights.mean()), "h2_weight_min": float(weights.min()),
            "h2_weight_max": float(weights.max()),
            "train_target_positive_counts": {t: int(y[:, j].sum()) for j, t in enumerate(TARGET_ORDER)},
            "population_users": len(uids), "population_sha256": ids_digest(uids),
            "prediction_sha256": {arm: hashlib.sha256(np.ascontiguousarray(pred_by_arm[arm]).tobytes()).hexdigest()
                                  for arm in ("REF", "H2")},
            "targets": target_fit_info,
        })
        progress["completed_origins"].append(S_text)
        progress["current_origin"] = None
        progress["current_arm"] = None
        progress["current_target"] = None
        _write_json(control_root / "progress.json", progress)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions_path = control_root / "predictions.parquet"
    prediction_sidecar = control_root / "predictions_manifest.json"
    if predictions_path.exists() or prediction_sidecar.exists():
        if not predictions_path.is_file() or not prediction_sidecar.is_file():
            raise RuntimeError(f"Incomplete prior control prediction artifact: {predictions_path}")
        saved = json.loads(prediction_sidecar.read_text(encoding="utf-8"))
        if saved.get("source_sha") != SOURCE_SHA or saved.get("file_sha256") != sha256_file(predictions_path):
            raise RuntimeError("Existing control predictions failed source/content authentication")
        prior = pd.read_parquet(predictions_path)
        if list(prior.columns) != list(predictions.columns) or not prior.equals(predictions):
            raise RuntimeError("Existing authenticated predictions disagree with resumed checkpoints")
        predictions = prior
    else:
        temp = predictions_path.with_name(predictions_path.name + f".{os.getpid()}.tmp")
        predictions.to_parquet(temp, engine="pyarrow", index=False)
        os.replace(temp, predictions_path)
        _write_json(prediction_sidecar, {
            "source_sha": SOURCE_SHA, "table_sha256": feature_manifest["table_sha256"],
            "file_sha256": sha256_file(predictions_path), "rows": len(predictions),
            "ordered_columns": list(predictions.columns),
        })
    fit_manifest = {
        "campaign_id": CAMPAIGN_ID, "source_sha": SOURCE_SHA,
        "table_sha256": feature_manifest["table_sha256"], "runtime": runtime,
        "registered_hgb_overrides": HGB_CONFIG,
        "resolved_hgb_estimator_defaults": resolved_estimator_params,
        "implementation_sha256": implementation_hash,
        "fit_count": progress["fit_count"], "origins": fit_summaries,
        "predictions_path": str(predictions_path.relative_to(ROOT)),
        "predictions_sha256": sha256_file(predictions_path),
        "continuity_origin_replay": "omitted; no selection influence",
    }
    _write_json(control_root / "fit_manifest.json", fit_manifest)
    metrics_by_origin, per_target, pooled = _summarize(predictions, official)
    metrics_by_origin.to_csv(control_root / "metrics_by_origin.csv", index=False)
    per_target.to_csv(control_root / "per_target_metrics.csv", index=False)
    pooled.to_csv(control_root / "metrics_pooled.csv", index=False)
    progress.update({"status": "fit_complete_verification_pending", "current_stage": "verification"})
    _write_json(control_root / "progress.json", progress)
    return {"fit_manifest": fit_manifest, "prediction_rows": len(predictions),
            "metrics_by_origin_rows": len(metrics_by_origin),
            "control_root": str(control_root)}


def _summarize(predictions: pd.DataFrame, official):
    systems = {
        "P_persist7": "dec_P_persist7",
        "REF/P016": "dec_REF_P016", "REF/P021": "dec_REF_P021",
        "H2/P016": "dec_H2_P016", "H2/P021": "dec_H2_P021",
    }
    by_origin, target_rows = [], []
    count_blocks = {}
    ap_by_origin = {}
    for origin in [iso_utc(s) for s in REPORT_ORIGINS]:
        block = predictions.loc[predictions.origin.eq(origin)].sort_values("UserID").reset_index(drop=True)
        uids = block.UserID.to_numpy(dtype=np.int64)
        y = block[[f"y_{t}" for t in TARGET_ORDER]].to_numpy(dtype=np.int8)
        ap_by_origin[origin] = {}
        for target_index, target in enumerate(TARGET_ORDER):
            n_pos = int(y[:, target_index].sum())
            for arm in ("REF", "H2"):
                p = block[f"p_{arm}_{target}"].to_numpy(dtype=np.float64)
                ap_by_origin[origin].setdefault(arm, []).append(
                    float(average_precision_score(y[:, target_index], p)) if 0 < n_pos < len(y) else None
                )
        for system, decision_base in systems.items():
            decision_cols = [f"{decision_base}_{t}" for t in TARGET_ORDER]
            pred = block[decision_cols].to_numpy(dtype=np.int8)
            counts = _score_counts(y, pred)
            scored = scores_from_counts(counts)
            official_scored = _official_score(official, uids, y, pred)
            if any(not np.isclose(float(scored[k]), official_scored[key], atol=1e-10)
                   for k, key in (("macro_f1", "macro_f1"), ("micro_f1", "micro_f1"), ("score", "score"))):
                raise AssertionError(f"Official evaluator parity failed for {origin}/{system}")
            mean_ap = None
            if system.startswith("REF/"):
                aps = ap_by_origin[origin]["REF"]
                mean_ap = float(np.mean([x for x in aps if x is not None])) if any(x is not None for x in aps) else None
            elif system.startswith("H2/"):
                aps = ap_by_origin[origin]["H2"]
                mean_ap = float(np.mean([x for x in aps if x is not None])) if any(x is not None for x in aps) else None
            by_origin.append({
                "origin": origin, "system": system, "users": len(uids),
                "macro_f1": float(scored["macro_f1"]), "micro_f1": float(scored["micro_f1"]),
                "combined_f1": float(scored["score"]), "mean_six_target_ap": mean_ap,
                "official_macro_f1": official_scored["macro_f1"],
                "official_micro_f1": official_scored["micro_f1"],
                "official_combined_f1": official_scored["score"],
            })
            count_blocks[(origin, system)] = counts
            for j, target in enumerate(TARGET_ORDER):
                tp, fp, fn, tn = (int(counts[k][j]) for k in ("tp", "fp", "fn", "tn"))
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                target_rows.append({
                    "origin": origin, "pool": "per_origin", "system": system,
                    "target": target, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": precision, "recall": recall,
                    "f1": float(_f1_per_target(counts)[j]),
                    "support": tp + fn, "predicted_positive_count": tp + fp,
                    "ap": (float(average_precision_score(y[:, j], block[f"p_{system.split('/')[0]}_{target}"].to_numpy(dtype=np.float64)))
                           if system.split("/")[0] in {"REF", "H2"} and 0 < int(y[:, j].sum()) < len(y)
                           else None),
                })

    scopes = {
        "all4": [iso_utc(s) for s in REPORT_ORIGINS],
        "mature3": [iso_utc(s) for s in REPORT_ORIGINS[:3]],
    }
    pooled_rows = []
    for scope, origins in scopes.items():
        for system in systems:
            counts = {k: np.sum([count_blocks[(o, system)][k] for o in origins], axis=0, dtype=np.int64)
                      for k in ("tp", "fp", "fn", "tn")}
            scored = scores_from_counts(counts)
            system_arm = system.split("/")[0]
            mean_ap_values = [
                np.mean([a for a in ap_by_origin[o][system_arm] if a is not None])
                for o in origins
                if system_arm in ap_by_origin[o]
                and any(a is not None for a in ap_by_origin[o][system_arm])
            ] if system_arm in {"REF", "H2"} else []
            pooled_rows.append({
                "scope": scope, "system": system,
                "origins": len(origins), "user_origin_rows": int(sum(
                    len(predictions.loc[predictions.origin.eq(o)].UserID.unique()) for o in origins)),
                "macro_f1": float(scored["macro_f1"]), "micro_f1": float(scored["micro_f1"]),
                "combined_f1": float(scored["score"]),
                "mean_origin_six_target_ap": float(np.mean(mean_ap_values)) if mean_ap_values else None,
                "official_all_negative_convention": True,
            })
            for j, target in enumerate(TARGET_ORDER):
                tp, fp, fn, tn = (int(counts[k][j]) for k in ("tp", "fp", "fn", "tn"))
                target_rows.append({
                    "origin": scope, "pool": scope, "system": system, "target": target,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "f1": float(_f1_per_target(counts)[j]),
                    "support": tp + fn, "predicted_positive_count": tp + fp,
                    "ap": None,
                })
    return pd.DataFrame(by_origin), pd.DataFrame(target_rows), pd.DataFrame(pooled_rows)


def run_manifest_latest(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "data_build" / "data_build_manifest.json").read_text(encoding="utf-8"))
    return manifest["source_latest_observed_utc"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / CAMPAIGN_ID)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = train(run_dir=args.run_dir)
    print(json.dumps({**result, "elapsed_seconds": time.perf_counter() - started}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
