"""Reconstruct all fixed blends, replay new models, and use the official scorer."""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                     "MKL_ENABLE_INSTRUCTIONS": "SSE4_2", "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.neural_research_common.data import TARGET_ORDER, hgb_weight_vector, sha256_file
from experiments.pca_fisher_20261026.run import array_sha, write_json
from experiments.pca_fisher_20261026.verify import cluster_bootstrap, matrix, metric_records
from src.phase_b_rules import rank01, topk_at_persist
from .models import WEIGHTED_RECIPES, all_recipes, fixed_rank_scores

ROOT = Path(__file__).resolve().parents[2]


def align(frame, controls):
    frame = frame.sort_values(["origin", "UserID"]).reset_index(drop=True)
    if frame.duplicated(["origin", "UserID"]).any() or not frame[["origin", "UserID"]].equals(controls[["origin", "UserID"]]):
        raise AssertionError("Constituent keys do not match the report controls")
    return frame


def probability_map(controls, parent_predictions, weighted):
    if set(weighted.recipe.unique()) != set(WEIGHTED_RECIPES):
        raise AssertionError("Weighted classifier family differs from registration")
    probabilities = {"H2": matrix(controls, "p_H2_")}
    for name in ("pca99_lr", "lr", "fisher_lda"):
        block = align(parent_predictions.loc[parent_predictions.recipe == name], controls)
        probabilities[name] = matrix(block, "p_")
    for name in WEIGHTED_RECIPES:
        block = align(weighted.loc[weighted.recipe == name], controls)
        probabilities[name] = matrix(block, "p_")
    return probabilities


def construct_candidates(controls, parent_predictions, weighted):
    controls = controls.sort_values(["origin", "UserID"]).reset_index(drop=True)
    probabilities = probability_map(controls, parent_predictions, weighted)
    frames = []
    for origin, indices in controls.groupby("origin", sort=True).indices.items():
        saved = controls.iloc[indices]
        ids = saved.UserID.to_numpy(dtype=np.int64)
        p = {name: values[indices] for name, values in probabilities.items()}
        for name, model_weights in all_recipes().items():
            mixed, final = fixed_rank_scores(p, matrix(saved, "ew7_"), model_weights)
            decisions = topk_at_persist(final, matrix(saved, "persist_"), ids)
            single = len(model_weights) == 1
            model_score = p[name] if single else mixed
            frame = pd.DataFrame({"recipe": name, "origin": origin, "UserID": ids,
                                  "ap_input": "raw_weighted_lr_probability" if single else "mixture_of_model_probability_ranks"})
            for j, target in enumerate(TARGET_ORDER):
                frame[f"y_{target}"] = saved[f"y_{target}"].to_numpy()
                frame[f"model_score_{target}"] = model_score[:, j]
                frame[f"decision_score_{target}"] = final[:, j]
                frame[f"dec_{target}"] = decisions[:, j].astype(np.int8)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def diversity_records(controls, parent_predictions):
    pca = align(parent_predictions.loc[parent_predictions.recipe == "pca99_lr"], controls)
    h = matrix(controls, "dec_H2_P021_")
    p = matrix(pca, "dec_P021_")
    y = matrix(controls, "y_")
    hr, pr = np.zeros(y.shape), np.zeros(y.shape)
    groups = list(controls.groupby("origin", sort=True).indices.items())
    for _, indices in groups:
        hr[indices] = rank01(matrix(controls.iloc[indices], "p_H2_"))
        pr[indices] = rank01(matrix(pca.iloc[indices], "p_"))
    rows = []
    for origin, idx in [*groups, ("pooled", np.arange(len(controls)))]:
        for j, target in enumerate(TARGET_ORDER):
            a, b, truth = h[idx, j], p[idx, j], y[idx, j]
            union = int(((a == 1) | (b == 1)).sum())
            corr = (float(np.corrcoef(hr[idx, j], pr[idx, j])[0, 1])
                    if np.std(hr[idx, j]) > 0 and np.std(pr[idx, j]) > 0 else None)
            rows.append({"origin": origin, "target": target, "rows": len(idx),
                "within_origin_probability_rank_correlation": corr,
                "decision_disagreements": int((a != b).sum()), "decision_disagreement_rate": float((a != b).mean()),
                "selected_overlap": int(((a == 1) & (b == 1)).sum()),
                "selected_jaccard": float(((a == 1) & (b == 1)).sum() / union) if union else 1.0,
                "h2_unique_true_positives": int(((a == 1) & (b == 0) & (truth == 1)).sum()),
                "pca_unique_true_positives": int(((a == 0) & (b == 1) & (truth == 1)).sum())})
    return pd.DataFrame(rows)


def verify_run(run_dir):
    from .run import authenticate_parent

    run_dir = Path(run_dir).resolve()
    registration = json.loads((run_dir / "preregistration.json").read_text(encoding="utf-8"))
    parent = Path(registration["parent_run"])
    for filename, field in (("preregistration.json", "parent_registration_sha256"),
                            ("artifacts_manifest.json", "parent_artifact_manifest_sha256")):
        if sha256_file(parent / filename) != registration[field]:
            raise AssertionError("Parent registration or artifact manifest changed")
    parent_registration = authenticate_parent(parent)
    for relative, expected in registration["source_files_sha256"].items():
        if sha256_file(ROOT / relative) != expected or sha256_file(run_dir / "source_snapshot" / relative) != expected:
            raise AssertionError(f"Follow-up implementation changed: {relative}")
    manifest = json.loads((run_dir / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if sha256_file(run_dir / relative) != expected:
            raise AssertionError(f"Follow-up artifact changed: {relative}")
    controls = pd.read_parquet(run_dir / "controls.parquet").sort_values(["origin", "UserID"]).reset_index(drop=True)
    parent_predictions = pd.read_parquet(run_dir / "parent_predictions.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(parent / "controls.parquet").sort_values(["origin", "UserID"]).reset_index(drop=True), controls)
    pd.testing.assert_frame_equal(pd.read_parquet(parent / "predictions.parquet"), parent_predictions)
    if sha256_file(parent / "official_evaluate.py") != sha256_file(run_dir / "official_evaluate.py"):
        raise AssertionError("Official scorer differs from parent")
    weighted = pd.read_parquet(run_dir / "weighted_probabilities.parquet")
    predicted = pd.read_parquet(run_dir / "predictions.parquet")
    reconstructed = construct_candidates(controls, parent_predictions, weighted)
    pd.testing.assert_frame_equal(predicted, reconstructed, check_exact=True)
    replays, fit_rows = [], []
    for origin in registration["config"]["report_origins"]:
        day = pd.Timestamp(origin).strftime("%Y%m%d")
        report = pd.read_parquet(parent / "report_features" / f"{day}.parquet")
        xe = report[parent_registration["ordered_f0"]].to_numpy(dtype=np.float64)
        original = joblib.load(parent / "models" / day / "pca99_lr.joblib").preprocessor
        parent_fit = json.loads((parent / "models" / day / "pca99_lr.json").read_text(encoding="utf-8"))
        for name, C in WEIGHTED_RECIPES.items():
            stem = run_dir / "models" / day / name
            meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            if sha256_file(stem.with_suffix(".joblib")) != meta["model_sha256"] or meta["C"] != C:
                raise AssertionError("Weighted model hash/config mismatch")
            for key in ("x_sha256", "y_sha256", "training_keys_sha256"):
                if meta[key] != parent_fit[key]:
                    raise AssertionError("Weighted training inputs differ from parent")
            weights = hgb_weight_vector(pd.Timestamp(origin), [pd.Timestamp(s) for s in meta["training_origins"]], meta["row_counts"])
            if array_sha(weights) != meta["weights_sha256"]:
                raise AssertionError("Recency-weight identity mismatch")
            model = joblib.load(stem.with_suffix(".joblib"))
            prep = model.preprocessor
            for attribute in ("mean_", "scale_", "var_"):
                np.testing.assert_array_equal(getattr(prep.scaler, attribute), getattr(original.scaler, attribute))
            for attribute in ("components_", "mean_", "explained_variance_", "singular_values_"):
                np.testing.assert_array_equal(getattr(prep.pca, attribute), getattr(original.pca, attribute))
            p = model.predict(xe, columns=parent_registration["ordered_f0"])
            block = weighted.loc[(weighted.origin == origin) & (weighted.recipe == name)].sort_values("UserID")
            if not np.array_equal(block.UserID, report.UserID) or not np.array_equal(p, matrix(block, "p_")):
                raise AssertionError("Weighted classifier replay mismatch")
            replays.append({"recipe": name, "origin": origin, "rows": len(p), "max_abs_error": 0.0,
                            "parent_preprocessing_exact": True})
            fit_rows.append({"recipe": name, "origin": origin, "C": C, "dimensions": meta["dimensions"],
                             "fit_seconds": meta["total_fit_seconds"],
                             "convergence_retries": sum(r["convergence_retry_4000"] for r in meta["target_fits"])})
    spec = importlib.util.spec_from_file_location("pca_followup_official", run_dir / "official_evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if tuple(official.EVENT_COLS) != tuple(TARGET_ORDER):
        raise AssertionError("Official target order differs")
    rows, target_rows = [], []
    for name, p, dec in [("H2", matrix(controls, "p_H2_"), matrix(controls, "dec_H2_P021_"))] + [
        (name, matrix(align(parent_predictions.loc[parent_predictions.recipe == name], controls), "p_"),
         matrix(align(parent_predictions.loc[parent_predictions.recipe == name], controls), "dec_P021_"))
        for name in ("pca99_lr", "lr", "fisher_lda")]:
        rr, tt = metric_records(official, name, "P021_control", controls, p, dec)
        for row in rr:
            row["ap_input"] = "raw_model_probability"
        rows.extend(rr)
        target_rows.extend(tt)
    for name in all_recipes():
        frame = align(predicted.loc[predicted.recipe == name], controls)
        dec = matrix(frame, "dec_")
        for _, idx in frame.groupby("origin", sort=True).indices.items():
            if not np.array_equal(dec[idx].sum(axis=0), matrix(controls.iloc[idx], "persist_").sum(axis=0)):
                raise AssertionError("Prediction counts differ from registered persistence counts")
        rr, tt = metric_records(official, name, "P021_fixed", frame, matrix(frame, "model_score_"), dec)
        for row in rr:
            row["ap_input"] = frame.ap_input.iloc[0]
        rows.extend(rr)
        target_rows.extend(tt)
    metrics = pd.DataFrame(rows)
    h2 = metrics.loc[metrics.recipe == "H2"].set_index("origin").combined_f1
    pca = metrics.loc[metrics.recipe == "pca99_lr"].set_index("origin").combined_f1
    if abs(h2["pooled"] - 0.5476344735753119) > 1e-12 or abs(pca["pooled"] - 0.5464761162893046) > 1e-12:
        raise AssertionError("Parent score identity failed")
    metrics["delta_vs_h2"] = [r.combined_f1 - h2[r.origin] for r in metrics.itertuples()]
    metrics["delta_vs_unweighted_pca99"] = [r.combined_f1 - pca[r.origin] for r in metrics.itertuples()]
    primary = metrics.loc[(metrics.origin == "pooled") & metrics.recipe.isin(all_recipes())].copy()
    for index, r in primary.iterrows():
        weeks = metrics.loc[(metrics.recipe == r.recipe) & (metrics.origin != "pooled")]
        primary.loc[index, "weekly_wins_vs_h2"] = int((weeks.delta_vs_h2 > 0).sum())
        primary.loc[index, "worst_week_delta_vs_h2"] = weeks.delta_vs_h2.min()
        primary.loc[index, "followup_screen_pass"] = bool(r.delta_vs_h2 >= 0.001 and (weeks.delta_vs_h2 > 0).sum() >= 3 and weeks.delta_vs_h2.min() >= -0.010)
    primary = primary.sort_values(["combined_f1", "recipe"], ascending=[False, True])
    best = primary.iloc[0]
    best_frame = align(predicted.loc[predicted.recipe == best.recipe], controls)
    bootstrap, interval = cluster_bootstrap(controls.UserID.to_numpy(), matrix(controls, "y_"),
        matrix(best_frame, "dec_"), matrix(controls, "dec_H2_P021_"))
    interval.update({"recipe": best.recipe, "caveat": "Exploratory: strongest of 12 new candidates selected after prior campaigns on four inspected weeks; percentile interval ignores model selection and temporal uncertainty"})
    diversity = diversity_records(controls, parent_predictions)
    pooled_diversity = diversity.loc[diversity.origin == "pooled"]
    report_dir = ROOT / "reports/pca_followup_20261026" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    primary.to_csv(report_dir / "comparison.csv", index=False)
    pd.DataFrame(target_rows).to_csv(report_dir / "per_target_metrics.csv", index=False)
    diversity.to_csv(report_dir / "h2_pca_diversity.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(report_dir / "fit_resources.csv", index=False)
    bootstrap.to_csv(report_dir / "bootstrap_draws.csv", index=False)
    write_json(report_dir / "bootstrap_summary.json", interval)
    write_json(report_dir / "replay_checks.json", replays)
    verification = {"status": "pass", "candidate_count": len(primary), "prediction_rows": len(predicted),
        "new_target_fits": 6 * len(replays), "model_replays": len(replays), "official_metric_rows": len(metrics),
        "exact_candidate_policy_reconstruction": True, "parent_preprocessing_exact": True,
        "strongest_recipe": best.recipe, "any_followup_screen_pass": bool(primary.followup_screen_pass.any()),
        "report_dir": str(report_dir)}
    write_json(run_dir / "verification.json", verification)
    write_json(report_dir / "verification.json", verification)
    text = ["# PCA follow-up: ensembles and recency weighting", "",
        f"Twelve fixed candidates evaluated on the same 22,919 user-weeks from four previously inspected September origins. Source `{parent_registration['source_sha']}`; report labels and historical training clocks match the parent experiment. All results are exploratory.", "",
        "## Results", "", "| Candidate | Combined F1 | Delta vs H2 | Delta vs unweighted PCA99 | Weekly wins vs H2 | Screen |",
        "|---|---:|---:|---:|---:|---|"]
    for r in primary.itertuples():
        text.append(f"| {r.recipe} | {r.combined_f1:.6f} | {r.delta_vs_h2:+.6f} | {r.delta_vs_unweighted_pca99:+.6f} | {int(r.weekly_wins_vs_h2)}/4 | {'pass' if r.followup_screen_pass else 'fail'} |")
    text += ["", f"Controls: H2/P021 **{h2['pooled']:.6f}**; unweighted PCA99/P021 **{pca['pooled']:.6f}**.", "",
        "Names `a025/a050/a075` identify PCA's share within the learned-model half of the score. `h2_lr` and `h2_fisher` are fixed equal model blends. The three-way blend assigns equal weights to H2, PCA99 and Fisher within that half. `weighted_pca99_c01/c1/c10` use C=0.1/1/10 in the recency-weighted classifier; their `h2_..._a050` counterparts blend equally with H2.", "",
        "## Complementarity and interpretation", "",
        f"Across all user-target cells, H2/PCA99 decisions disagree in {int(pooled_diversity.decision_disagreements.sum())} cases ({100 * pooled_diversity.decision_disagreements.sum() / (len(controls) * 6):.2f}%). PCA uniquely selects {int(pooled_diversity.pca_unique_true_positives.sum())} positive cases missed by H2; H2 uniquely selects {int(pooled_diversity.h2_unique_true_positives.sum())} positive cases missed by PCA. These are descriptive diagnostics and are not used as label-dependent decision rules. Per-week/per-target rank correlations and selection overlap are in `h2_pca_diversity.csv`.", "",
        "All candidates preserve EW7's 0.5 share, the persistence-positive count and the same UserID tie rule. Model probability ranks are mixed directly, without reranking the intermediate mixture. Scaler and PCA remain unweighted and exactly reproduce the parent transforms; only the new logistic classifier losses use two-week half-life weights. The C=1 weighted-versus-unweighted PCA99 comparison isolates that loss-weighting change.", "",
        f"Strongest observed candidate: `{best.recipe}`, combined F1 {best.combined_f1:.6f}, delta {best.delta_vs_h2:+.6f}. Its paired UserID-cluster bootstrap delta interval is [{interval['delta_95_percentile_interval'][0]:+.6f}, {interval['delta_95_percentile_interval'][1]:+.6f}] (1,000 draws). This interval ignores selection optimism across 12 candidates and earlier campaigns, and does not establish performance on unseen weeks.", "",
        ("At least one recipe passes the registered development follow-up screen. It can be considered for a separately frozen unseen confirmation; no submission policy was changed."
         if verification["any_followup_screen_pass"] else
         "No recipe passes the registered follow-up screen. The bounded family is complete; no submission policy was changed."), "",
        "## Verification and evidence", "",
        f"All {len(replays) * 6} new target fits completed; {len(replays)} saved bundles replayed exactly, with exact parent preprocessing; all 12 candidate policies were reconstructed from saved constituents. All {len(metrics)} weekly/pooled metric rows match the frozen official evaluator. Parent/source/code/config/input and new-artifact hashes passed.", "",
        "- `comparison.csv`, `metrics.csv`, `per_target_metrics.csv`: all registered candidates and controls, including weekly/target behavior and counts.",
        "- `h2_pca_diversity.csv`: complementary selections, disagreement and within-origin probability-rank correlations.",
        "- `fit_resources.csv`, `replay_checks.json`, `verification.json`: fit, convergence and replay evidence.",
        "- `bootstrap_summary.json`, `bootstrap_draws.csv`: exploratory uncertainty for the strongest observed candidate.",
        "- AP uses raw probabilities for individual models and a mixture of model probability ranks for ensembles, excluding EW7; `ap_input` identifies the representation. Rank mixtures are not calibrated probabilities.",
        f"- Run directory: `{run_dir.relative_to(ROOT)}`; frozen plan/config/code, parent hashes, weighted models, constituent probabilities and keyed candidate scores/decisions.", ""]
    (report_dir / "FINDINGS.md").write_text("\n".join(text), encoding="utf-8")
    print(primary[["recipe", "combined_f1", "delta_vs_h2", "weekly_wins_vs_h2", "followup_screen_pass"]].to_string(index=False), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(json.dumps(verify_run(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
