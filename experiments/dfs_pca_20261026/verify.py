"""Verify source-generated report features, training-only statistics and official F1."""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "MKL_ENABLE_INSTRUCTIONS": "SSE4_2", "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import OfficialData
from src.decision_policies import decision_time, feature_cutoff
from src.phase_b_rules import blend_score, rank01, topk_at_persist
from experiments.neural_research_common.data import (
    TARGET_ORDER, TIMING, hgb_weight_vector, iso_utc, sha256_file, sha256_json,
)
from experiments.neural_research_common.train_controls import _load_inputs, _training_rows
from experiments.pca_fisher_20261026.run import array_sha, write_json
from experiments.pca_fisher_20261026.verify import cluster_bootstrap, matrix, metric_records
from experiments.pca_followup_20261026.verify import align
from .features import COLUMNS, EXPRESSIONS, SUPPORT, SynthesizedRepoFeatures

ROOT = Path(__file__).resolve().parents[2]


def independent_source_traces(builder, cutoff, features):
    """Five bounded user examples checked with direct row filters and Python sets."""
    raw = builder.adapter._load().events
    eligible = raw.loc[(raw.created_at_utc >= cutoff - pd.Timedelta(days=30)) & (raw.created_at_utc < cutoff)
        & raw.actor_fk_known & raw.actor_is_cohort & raw.repo_fk_known & raw.repo_github_id.notna()
        & (raw.actor_first_seen_utc < cutoff) & (raw.repo_first_seen_utc < cutoff)]
    samples = features.loc[features[SUPPORT].between(1, 5), "UserID"].head(5)
    if len(samples) != 5:
        raise AssertionError("Fewer than five supported low-degree users available for source traces")
    traces = []
    for uid in samples:
        repos = sorted(eligible.loc[eligible.UserID == uid, "repo_github_id"].unique())
        expected, per_repo = {}, []
        for repo in repos:
            other = eligible.loc[(eligible.repo_github_id == repo) & (eligible.UserID != uid)]
            counts = {}
            for days in (7, 30):
                recent = other.loc[other.created_at_utc >= cutoff - pd.Timedelta(days=days)]
                for target in ("push", "issue_comment", "pr_review_comment"):
                    counts[(target, days)] = int((recent.target == target).sum())
                counts[("all_direct_users", days)] = len(set(recent.UserID))
            per_repo.append(counts)
        for expression in EXPRESSIONS:
            values = [counts[(expression.target, expression.window_days)] for counts in per_repo]
            expected[expression.column] = float(getattr(np, expression.outer_primitive)(values))
        expected[SUPPORT] = len(repos)
        actual = features.loc[features.UserID == uid, list(COLUMNS)].to_numpy(dtype=np.float64)[0]
        np.testing.assert_allclose(actual, [expected[c] for c in COLUMNS], rtol=1e-12, atol=1e-9)
        traces.append({"UserID": int(uid), "cutoff": cutoff.isoformat(), "exposed_repo_ids": [int(r) for r in repos],
                       "independent_values": expected, "all_25_features_match": True})
    return traces


def verify_preprocessing(prep, x):
    logged = prep._logged(x)
    np.testing.assert_allclose(prep.scaler.mean_, logged.mean(axis=0), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(prep.scaler.var_, logged.var(axis=0), rtol=1e-10, atol=1e-12)
    if int(prep.scaler.n_samples_seen_) != len(x) or prep.training_rows != len(x):
        raise AssertionError("Standardization did not use exactly the registered training rows")
    z = prep.scaler.transform(logged)
    del logged
    np.testing.assert_allclose(prep.pca.mean_, z.mean(axis=0), rtol=1e-10, atol=1e-12)
    components = prep.pca.components_
    np.testing.assert_allclose(components @ components.T, np.eye(len(components)), rtol=0, atol=1e-12)
    projected = prep.pca.transform(z)
    covariance = np.cov(projected, rowvar=False)
    np.testing.assert_allclose(covariance, np.diag(prep.pca.explained_variance_), rtol=1e-9, atol=1e-9)
    total_variance = z.var(axis=0, ddof=1).sum()
    np.testing.assert_allclose(prep.pca.explained_variance_ratio_, np.diag(covariance) / total_variance, rtol=1e-9, atol=1e-9)
    ratios = prep.pca.explained_variance_ratio_
    if ratios.sum() < 0.99 or ratios[:-1].sum() >= 0.99:
        raise AssertionError("PCA dimension does not match the fixed 99% retention rule")
    return {"training_rows": len(x), "training_only_mean_variance": True, "PCA_covariance_eigenbasis": True,
            "input_dimensions": x.shape[1], "output_dimensions": int(prep.pca.n_components_),
            "explained_variance": float(ratios.sum())}


def verify_run(run_dir):
    from .run import authenticate_challenger, training_extra, DATA_RUN, PARENT

    run_dir = Path(run_dir).resolve()
    registration = json.loads((run_dir / "preregistration.json").read_text(encoding="utf-8"))
    baseline, base = authenticate_challenger()
    for filename, field in (("preregistration.json", "parent_registration_sha256"),
                            ("artifacts_manifest.json", "parent_artifact_manifest_sha256")):
        if sha256_file(PARENT / filename) != registration[field]:
            raise AssertionError("Parent identity changed")
    for rel, expected in registration["source_files_sha256"].items():
        if sha256_file(ROOT / rel) != expected or sha256_file(run_dir / "source_snapshot" / rel) != expected:
            raise AssertionError(f"Experiment implementation changed: {rel}")
    artifacts = json.loads((run_dir / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in artifacts["sha256"].items():
        if sha256_file(run_dir / rel) != expected:
            raise AssertionError(f"Artifact changed: {rel}")
    fm, lm, f0, feature_paths, label_paths = _load_inputs(DATA_RUN)
    cache = json.loads((run_dir / "feature_cache_manifest.json").read_text(encoding="utf-8"))
    if cache["ordered_columns"] != list(COLUMNS) or registration["ordered_augmented"] != [*f0, *COLUMNS]:
        raise AssertionError("Feature expression order differs from the fitted schema")
    generated = {}
    for row in cache["entries"]:
        path = run_dir / row["path"]
        if sha256_file(path) != row["sha256"] or sha256_file(feature_paths[row["origin"]]) != row["f0_cache_sha256"]:
            raise AssertionError("Generated/F0 feature cache changed")
        frame = pd.read_parquet(path)
        original_ids = pd.read_parquet(feature_paths[row["origin"]], columns=["UserID"]).UserID.to_numpy(dtype=np.int64)
        C = feature_cutoff(pd.Timestamp(row["origin"]), TIMING)
        if (list(frame.columns) != ["UserID", "cutoff_utc", *COLUMNS] or len(frame) != row["rows"]
                or not np.array_equal(frame.UserID.to_numpy(), original_ids)
                or array_sha(original_ids) != row["ordered_population_sha256"] or not frame.cutoff_utc.eq(C).all()):
            raise AssertionError("Generated feature schema, population or cutoff mismatch")
        if not row["audit"]["all_events_and_endpoints_before_cutoff"]:
            raise AssertionError("A contributing event/endpoint was not visible at the feature cutoff")
        for key in ("max_event_utc", "max_actor_seen_utc", "max_repo_seen_utc"):
            value = row["audit"][key]
            if value is not None and pd.Timestamp(value) >= C:
                raise AssertionError("Cache cutoff audit failed")
        generated[row["origin"]] = path
    controls = pd.read_parquet(run_dir / "controls.parquet").sort_values(["origin", "UserID"]).reset_index(drop=True)
    expected_controls = pd.read_parquet(PARENT / "controls.parquet").sort_values(["origin", "UserID"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(controls, expected_controls)
    if sha256_file(run_dir / "official_evaluate.py") != sha256_file(PARENT / "official_evaluate.py"):
        raise AssertionError("Frozen official evaluator changed")
    predictions = align(pd.read_parquet(run_dir / "predictions.parquet"), controls)
    np.testing.assert_array_equal(matrix(predictions, "y_"), matrix(controls, "y_"))
    report_rebuilds, replays, resources, preprocess_checks = [], [], [], []
    source = OfficialData(Path(registration["source_db"]), expected_cohort=9000)
    builder = SynthesizedRepoFeatures(source)
    source_traces = []
    for key in registration["config"]["report_origins"]:
        ids = controls.loc[controls.origin == key].UserID.to_numpy(dtype=np.int64)
        rebuilt, audit = builder.build(feature_cutoff(pd.Timestamp(key), TIMING), ids)
        pd.testing.assert_frame_equal(rebuilt, pd.read_parquet(generated[key]), check_exact=True)
        report_rebuilds.append({"origin": key, "rows": len(ids), "exact_source_rebuild": True})
        if not source_traces:
            source_traces = independent_source_traces(builder, feature_cutoff(pd.Timestamp(key), TIMING), rebuilt)
    del source, builder
    gc.collect()
    for key in registration["config"]["report_origins"]:
        S = pd.Timestamp(key)
        day = S.strftime("%Y%m%d")
        stem = run_dir / "models" / day
        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256_file(stem.with_suffix(".joblib")) != meta["model_sha256"]:
            raise AssertionError("Augmented model content changed")
        origins, x, y, row_counts, keys = _training_rows(S, f0, feature_paths, label_paths, decision_time(S, TIMING))
        extra = training_extra(origins, feature_paths, generated)
        xa = np.column_stack([x, extra])
        weights = hgb_weight_vector(S, origins, row_counts)
        values = {"x_sha256": array_sha(x), "y_sha256": array_sha(y), "training_keys_sha256": sha256_json(keys),
                  "augmented_x_sha256": array_sha(xa), "weights_sha256": array_sha(weights)}
        if any(meta[k] != value for k, value in values.items()):
            raise AssertionError("Training values, keys or weights changed")
        model = joblib.load(stem.with_suffix(".joblib"))
        preprocess_checks.append({"origin": key, **verify_preprocessing(model.preprocessor, xa)})
        feat = pd.read_parquet(feature_paths[key])
        dfs = pd.read_parquet(generated[key])
        xe = np.column_stack([feat[f0].to_numpy(dtype=np.float64), dfs[list(COLUMNS)].to_numpy(dtype=np.float64)])
        p = model.predict(xe, columns=registration["ordered_augmented"])
        subset = predictions.loc[predictions.origin == key]
        if not np.array_equal(subset.UserID.to_numpy(), feat.UserID.to_numpy()) or not np.array_equal(p, matrix(subset, "p_")):
            raise AssertionError("Saved-model prediction replay failed")
        saved = controls.loc[controls.origin == key]
        ids, persist, ew = saved.UserID.to_numpy(), matrix(saved, "persist_"), matrix(saved, "ew7_")
        scores = {"augmented_branch": blend_score(p, ew, 0.5),
                  "augmented_ensemble": 0.5 * (0.5 * rank01(matrix(saved, "p_H2_")) + 0.5 * rank01(p)) + 0.5 * rank01(ew)}
        for policy, score in scores.items():
            np.testing.assert_array_equal(matrix(subset, f"score_{policy}_"), score)
            dec = topk_at_persist(score, persist, ids)
            np.testing.assert_array_equal(matrix(subset, f"dec_{policy}_"), dec)
            np.testing.assert_array_equal(dec.sum(axis=0), persist.sum(axis=0))
        replays.append({"origin": key, "rows": len(p), "max_abs_error": 0.0, "both_policies_reconstructed": True})
        resources.append({"origin": key, "training_rows": len(xa), "raw_dimensions": xa.shape[1],
            "retained_PCs": model.preprocessor.pca.n_components_, "preprocessing_seconds": meta["preprocessing_seconds"],
            "classifier_seconds": meta["total_fit_seconds"],
            "convergence_retries": sum(t["convergence_retry_4000"] for t in meta["target_fits"])})
        del x, y, xa, extra, keys, model
        gc.collect()
    parity = json.loads((run_dir / "control_parity.json").read_text(encoding="utf-8"))
    first_key = registration["config"]["report_origins"][0]
    first_day = pd.Timestamp(first_key).strftime("%Y%m%d")
    first_features = pd.read_parquet(feature_paths[first_key])
    fitted_control = joblib.load(run_dir / "control_parity.joblib")
    expected_control = joblib.load(PARENT / "models" / first_day / "weighted_pca99_c1.joblib")
    np.testing.assert_array_equal(fitted_control.predict(first_features[f0], columns=f0),
                                  expected_control.predict(first_features[f0], columns=f0))
    spec = importlib.util.spec_from_file_location("dfs_pca_official", run_dir / "official_evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if tuple(official.EVENT_COLS) != tuple(TARGET_ORDER):
        raise AssertionError("Official target order mismatch")
    parent_weighted = pd.read_parquet(PARENT / "weighted_probabilities.parquet")
    parent_branch = align(parent_weighted.loc[parent_weighted.recipe == "weighted_pca99_c1"], controls)
    parent_pred = pd.read_parquet(PARENT / "predictions.parquet")
    parent_ensemble = align(parent_pred.loc[parent_pred.recipe == "h2_weighted_pca99_c1_a050"], controls)
    parent_p = matrix(parent_branch, "p_")
    old_branch_dec, old_model_scores, new_model_scores = np.zeros_like(parent_p, dtype=np.int8), np.zeros_like(parent_p), np.zeros_like(parent_p)
    new_p = matrix(predictions, "p_")
    for key, idx in controls.groupby("origin", sort=True).indices.items():
        saved = controls.iloc[idx]
        h = rank01(matrix(saved, "p_H2_"))
        old_model_scores[idx] = 0.5 * h + 0.5 * rank01(parent_p[idx])
        new_model_scores[idx] = 0.5 * h + 0.5 * rank01(new_p[idx])
        old_branch_dec[idx] = topk_at_persist(blend_score(parent_p[idx], matrix(saved, "ew7_"), 0.5), matrix(saved, "persist_"), saved.UserID.to_numpy())
    all_rows, target_rows = [], []
    systems = [("H2", matrix(controls, "p_H2_"), matrix(controls, "dec_H2_P021_"), "raw_H2_probability"),
        ("F0_branch", parent_p, old_branch_dec, "raw_PCA_branch_probability"),
        ("F0_ensemble", old_model_scores, matrix(parent_ensemble, "dec_"), "equal_H2_PCA_model_rank_mixture"),
        ("augmented_branch", new_p, matrix(predictions, "dec_augmented_branch_"), "raw_PCA_branch_probability"),
        ("augmented_ensemble", new_model_scores, matrix(predictions, "dec_augmented_ensemble_"), "equal_H2_PCA_model_rank_mixture")]
    for name, score, dec, ap_input in systems:
        rr, tt = metric_records(official, name, "fixed_P021", controls, score, dec)
        for row in rr:
            row["ap_input"] = ap_input
        all_rows.extend(rr)
        target_rows.extend(tt)
    metrics = pd.DataFrame(all_rows)
    old = metrics.loc[metrics.recipe == "F0_ensemble"].set_index("origin").combined_f1
    h2 = metrics.loc[metrics.recipe == "H2"].set_index("origin").combined_f1
    if abs(old["pooled"] - 0.5499040953014345) > 1e-12 or abs(h2["pooled"] - 0.5476344735753119) > 1e-12:
        raise AssertionError("Existing control scores do not reproduce")
    metrics["delta_vs_existing_ensemble"] = [r.combined_f1 - old[r.origin] for r in metrics.itertuples()]
    metrics["delta_vs_h2"] = [r.combined_f1 - h2[r.origin] for r in metrics.itertuples()]
    pooled = metrics.loc[metrics.origin == "pooled"].copy()
    primary = metrics.loc[(metrics.recipe == "augmented_ensemble") & (metrics.origin == "pooled")].iloc[0]
    weekly = metrics.loc[(metrics.recipe == "augmented_ensemble") & (metrics.origin != "pooled")]
    passes = bool(primary.delta_vs_existing_ensemble >= 0.001 and (weekly.delta_vs_existing_ensemble > 0).sum() >= 3
                  and weekly.delta_vs_existing_ensemble.min() >= -0.010)
    report_dir = ROOT / "reports/dfs_pca_20261026" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    intervals = {}
    for comparator, dec in (("existing_ensemble", matrix(parent_ensemble, "dec_")), ("H2", matrix(controls, "dec_H2_P021_"))):
        draws, summary = cluster_bootstrap(controls.UserID.to_numpy(), matrix(controls, "y_"),
            matrix(predictions, "dec_augmented_ensemble_"), dec)
        summary.update({"candidate": "augmented_ensemble", "comparator": comparator,
                        "caveat": "Exploratory four inspected weeks, after prior representation/model campaigns; interval ignores temporal and selection uncertainty"})
        draws = draws.rename(columns={"h2_combined_f1": "comparator_combined_f1"})
        draws.to_csv(report_dir / f"bootstrap_vs_{comparator}.csv", index=False)
        write_json(report_dir / f"bootstrap_vs_{comparator}.json", summary)
        intervals[comparator] = summary["delta_95_percentile_interval"]
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    pooled.to_csv(report_dir / "comparison.csv", index=False)
    pd.DataFrame(target_rows).to_csv(report_dir / "per_target_metrics.csv", index=False)
    pd.DataFrame(resources).to_csv(report_dir / "fit_resources.csv", index=False)
    shutil_source = run_dir / "training_feature_diagnostics.csv"
    (report_dir / "training_feature_diagnostics.csv").write_bytes(shutil_source.read_bytes())
    write_json(report_dir / "preprocessing_checks.json", preprocess_checks)
    write_json(report_dir / "report_feature_rebuilds.json", report_rebuilds)
    write_json(report_dir / "independent_source_traces.json", source_traces)
    write_json(report_dir / "replay_checks.json", replays)
    write_json(report_dir / "feature_inventory.json", registration["feature_inventory"])
    checks = {"status": "pass", "generated_features": len(COLUMNS), "feature_cache_origins": len(generated),
        "exact_source_report_rebuilds": len(report_rebuilds), "candidate_target_fits": len(replays) * 6,
        "independent_source_user_traces": len(source_traces),
        "fresh_control_parity_target_fits": 6, "exact_model_replays": len(replays),
        "training_only_preprocessing_checks": len(preprocess_checks), "official_metric_rows": len(metrics),
        "prediction_rows": len(predictions), "followup_screen_pass": passes, "report_dir": str(report_dir)}
    write_json(run_dir / "verification.json", checks)
    write_json(report_dir / "verification.json", checks)
    description = ["# Constrained DFS-style features in the PCA ensemble", "",
        "One predeclared repository-neighbor group was synthesized from the three audited direct-event tables. This is an explicit pandas expression grammar, not an unrestricted Featuretools run. It stacks self-excluded COUNT/NUM_UNIQUE with MEAN/MAX/population-STD over recent exposure repositories, plus exposure support: 25 columns added to F0 for 84 raw inputs.", "",
        "## Results", "", "| System | Combined F1 | Delta vs existing ensemble | Delta vs H2 |",
        "|---|---:|---:|---:|"]
    for r in pooled.itertuples():
        description.append(f"| {r.recipe} | {r.combined_f1:.6f} | {r.delta_vs_existing_ensemble:+.6f} | {r.delta_vs_h2:+.6f} |")
    description += ["", f"Primary augmented ensemble: {int((weekly.delta_vs_existing_ensemble > 0).sum())}/4 weekly wins versus the existing ensemble; worst weekly delta {weekly.delta_vs_existing_ensemble.min():+.6f}. Registered follow-up screen: **{'PASS' if passes else 'FAIL'}**.", "",
        f"Paired UserID bootstrap intervals for the primary delta (1,000 draws): vs existing ensemble [{intervals['existing_ensemble'][0]:+.6f}, {intervals['existing_ensemble'][1]:+.6f}]; vs H2 [{intervals['H2'][0]:+.6f}, {intervals['H2'][1]:+.6f}].", "",
        "## Interpretation and scope", "",
        "This compares one synthesized group under a fixed PCA99/C=1 classifier and fixed 25% H2 / 25% PCA branch / 50% EW7 rank mixture. Classifier training recency, prediction counts, population, source release, historical label clocks and report truth match the parent. No feature, model-weight, C or retained-variance search was performed. All 25 features were retained before PCA, including any constant or duplicate columns; training-only feature diagnostics are descriptive.", "",
        "The four September origins were inspected previously, and the parent ensemble was selected after earlier campaigns. Results and bootstrap intervals are exploratory; they do not establish unseen temporal performance. No October-origin outcome was evaluated and no submission policy changed.", "",
        ("The group passes the development screen and can be considered for separately frozen unseen confirmation."
         if passes else "The group fails the development screen. Keep the existing F0 ensemble as the stronger reference and close this bounded feature experiment."), "",
        "A negative result applies to this repository-neighbor grammar in this pipeline; it does not reject all feature synthesis, workflow features or relational information.", "",
        "## Verification and evidence", "",
        f"Source/cache/parent/code hashes passed. All {len(generated)} feature cutoffs passed event and endpoint timing audits. The four report feature blocks rebuilt exactly from source. Fresh F0 control probabilities reproduced the parent exactly before candidate fitting. All 24 augmented target fits completed; four bundles replayed exactly, and training-only scaler/PCA statistics and covariance/eigenbasis checks passed. All {len(metrics)} weekly/pooled metric rows match the official evaluator; both candidate policies and prediction counts reconstruct exactly.", "",
        "- `metrics.csv`, `comparison.csv`, `per_target_metrics.csv`: weekly/pooled/target evidence and AP input definitions.",
        "- `feature_inventory.json`, `training_feature_diagnostics.csv`: exact generated expressions, support/zero rates, variability and exact F0 duplicates.",
        "- `fit_resources.csv`, `preprocessing_checks.json`, `report_feature_rebuilds.json`, `replay_checks.json`, `verification.json`: source, training-statistic and model checks.",
        "- `bootstrap_vs_existing_ensemble.*`, `bootstrap_vs_H2.*`: exploratory paired uncertainty. CSV comparator-score columns are named generically.",
        f"- Run directory `{run_dir.relative_to(ROOT)}`: frozen plan/config/code, source/cutoff audits, every feature cache and manifest, saved models, probabilities and binary decisions.", ""]
    (report_dir / "FINDINGS.md").write_text("\n".join(description), encoding="utf-8")
    print(pooled[["recipe", "combined_f1", "delta_vs_existing_ensemble", "delta_vs_h2"]].to_string(index=False), flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(json.dumps(verify_run(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
