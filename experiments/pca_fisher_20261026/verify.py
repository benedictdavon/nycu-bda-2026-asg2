"""Independently reload predictions/models and score through the official evaluator.

python -B -m experiments.pca_fisher_20261026.verify --run-dir <run directory>
"""
from __future__ import annotations

import os
for _key, _value in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                     "MKL_NUM_THREADS": "4", "MKL_ENABLE_INSTRUCTIONS": "SSE4_2",
                     "MKL_THREADING_LAYER": "SEQUENTIAL"}.items():
    os.environ.setdefault(_key, _value)

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.metrics import confusion_arrays, scores_from_counts
from src.phase_b_rules import blend_score, topk_at_persist
from experiments.neural_research_common.data import TARGET_ORDER, sha256_file
from .models import RECIPES

ROOT = Path(__file__).resolve().parents[2]


def matrix(frame, prefix):
    return frame[[f"{prefix}{t}" for t in TARGET_ORDER]].to_numpy()


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def official_scores(evaluator, truth, decisions, ids=None):
    # Pooled repeated-user weeks get unique row IDs solely for the scorer join.
    ids = np.arange(len(truth), dtype=np.int64) if ids is None else ids
    actual = pd.DataFrame(truth, columns=TARGET_ORDER).assign(UserID=ids)
    pred = pd.DataFrame(decisions, columns=TARGET_ORDER).assign(UserID=ids)
    return evaluator.scores(pred, actual), evaluator.f1_per_event(pred, actual)


def mean_ap(truth, probability):
    return float(np.mean([average_precision_score(truth[:, j], probability[:, j])
                          if 0 < truth[:, j].sum() < len(truth) else truth[:, j].mean()
                          for j in range(len(TARGET_ORDER))]))


def metric_records(evaluator, name, policy, frame, probability, decisions):
    truth = matrix(frame, "y_")
    records, target_rows = [], []
    weeks = list(frame.groupby("origin", sort=True).indices.items())
    groups = [(str(origin), indices) for origin, indices in weeks] + [("pooled", np.arange(len(frame)))]
    for origin, idx in groups:
        y, p, d = truth[idx], probability[idx], decisions[idx]
        official, f1 = official_scores(evaluator, y, d, None if origin == "pooled" else frame.iloc[idx].UserID.to_numpy())
        counts = confusion_arrays(d, y)
        local = scores_from_counts(counts)
        for field in ("macro_f1", "micro_f1", "score"):
            if abs(official[field] - local[field]) > 1e-12:
                raise AssertionError("Official scorer/count metric mismatch")
        ap = (float(np.mean([mean_ap(truth[wi], probability[wi]) for _, wi in weeks]))
              if origin == "pooled" else mean_ap(y, p))
        records.append({"recipe": name, "policy": policy, "origin": origin, "rows": len(idx),
                        "combined_f1": float(official["score"]), "macro_f1": float(official["macro_f1"]),
                        "micro_f1": float(official["micro_f1"]), "mean_origin_ap": ap})
        for j, target in enumerate(TARGET_ORDER):
            target_rows.append({"recipe": name, "policy": policy, "origin": origin, "target": target,
                                "f1": float(f1[j]), "positives": int(y[:, j].sum()),
                                "predicted_positives": int(d[:, j].sum()),
                                **{k: int(v[j]) for k, v in counts.items()}})
    return records, target_rows


def combined_from_contributions(counts):
    """Counts shape: systems x (TP, FP, FN) x targets, matching official zero division."""
    tp, fp, fn = counts[:, 0], counts[:, 1], counts[:, 2]
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.ones_like(tp, dtype=float), where=denom != 0)
    summed = denom.sum(axis=1)
    micro = np.divide(2 * tp.sum(axis=1), summed, out=np.ones(len(tp)), where=summed != 0)
    return (f1.mean(axis=1) + micro) / 2


def cluster_bootstrap(ids, truth, candidate, baseline, *, draws=1000, seed=42):
    users, inverse = np.unique(ids, return_inverse=True)
    contributions = np.zeros((len(users), 2, 3, len(TARGET_ORDER)), dtype=np.int64)
    for system, d in enumerate((candidate, baseline)):
        for component, cells in enumerate(((d == 1) & (truth == 1), (d == 1) & (truth == 0),
                                           (d == 0) & (truth == 1))):
            np.add.at(contributions[:, system, component], inverse, cells.astype(np.int64))
    rng = np.random.default_rng(seed)
    results = []
    for draw in range(draws):
        multiplicity = np.bincount(rng.integers(0, len(users), size=len(users)), minlength=len(users))
        counts = np.einsum("u,usct->sct", multiplicity, contributions)
        scores = combined_from_contributions(counts)
        results.append({"draw": draw, "candidate_combined_f1": scores[0],
                        "h2_combined_f1": scores[1], "delta": scores[0] - scores[1]})
    return pd.DataFrame(results), {"draws": draws, "seed": seed, "clusters": len(users),
        "delta_95_percentile_interval": np.quantile([r["delta"] for r in results], [0.025, 0.975]).tolist(),
        "caveat": "Exploratory, selected strongest of five recipes; September inspected previously; only four weeks"}


def verify(run_dir):
    run_dir = Path(run_dir).resolve()
    registration = json.loads((run_dir / "preregistration.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "artifacts_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if sha256_file(run_dir / relative) != expected:
            raise AssertionError(f"Artifact content changed: {relative}")
    for relative, expected in registration["source_files_sha256"].items():
        if sha256_file(run_dir / "source_snapshot" / relative) != expected:
            raise AssertionError(f"Frozen implementation changed: {relative}")
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"Current implementation differs from frozen code: {relative}")
    for relative, expected in registration["inputs_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"Controlling input changed: {relative}")
    for table, expected in registration["source_table_sha256"].items():
        if sha256_file(Path(registration["source_db"]) / f"{table}.parquet") != expected:
            raise AssertionError(f"Source table changed: {table}")
    spec = importlib.util.spec_from_file_location("pca_fisher_frozen_official", run_dir / "official_evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if tuple(official.EVENT_COLS) != tuple(TARGET_ORDER):
        raise AssertionError("Official target order mismatch")
    controls = pd.read_parquet(run_dir / "controls.parquet").sort_values(["origin", "UserID"]).reset_index(drop=True)
    original = pd.read_parquet(registration["control_path"]).sort_values(["origin", "UserID"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(controls, original)
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    if set(predictions.recipe.unique()) != set(RECIPES) or predictions.duplicated(["recipe", "origin", "UserID"]).any():
        raise AssertionError("Missing/duplicate candidate keys")
    records, per_target, replay_checks, artifacts = [], [], [], []
    for name, policy, p, d in (
        ("H2", "P021", matrix(controls, "p_H2_"), matrix(controls, "dec_H2_P021_")),
        ("persistence", "persist7", matrix(controls, "persist_"), matrix(controls, "persist_")),
    ):
        rr, tt = metric_records(official, name, policy, controls, p, d)
        records.extend(rr)
        per_target.extend(tt)
    h2_row = next(r for r in records if r["recipe"] == "H2" and r["origin"] == "pooled")
    if abs(h2_row["combined_f1"] - registration["h2_p021_combined_f1"]) > 1e-12:
        raise AssertionError("Official pooled H2 baseline does not reproduce")
    for recipe in RECIPES:
        frame = predictions.loc[predictions.recipe == recipe].sort_values(["origin", "UserID"]).reset_index(drop=True)
        if not frame[["origin", "UserID"]].equals(controls[["origin", "UserID"]]) or not np.array_equal(matrix(frame, "y_"), matrix(controls, "y_")):
            raise AssertionError("Candidate population/truth differs from controls")
        p = matrix(frame, "p_")
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise AssertionError("Invalid candidate probability")
        for origin, indices in frame.groupby("origin", sort=True).indices.items():
            day = pd.Timestamp(origin).strftime("%Y%m%d")
            stem = run_dir / "models" / day / recipe
            fit_meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            if sha256_file(stem.with_suffix(".joblib")) != fit_meta["model_sha256"]:
                raise AssertionError("Model hash differs from fit manifest")
            report = pd.read_parquet(run_dir / "report_features" / f"{day}.parquet")
            xe = report[registration["ordered_f0"]].to_numpy(dtype=np.float64)
            if not np.array_equal(report.UserID, frame.iloc[indices].UserID):
                raise AssertionError("Replay features have different stable UserIDs")
            model = joblib.load(stem.with_suffix(".joblib"))
            replay = model.predict(xe, columns=registration["ordered_f0"])
            if not np.array_equal(replay, p[indices]):
                raise AssertionError("Reloaded prediction replay failed")
            saved = controls.iloc[indices]
            ids = saved.UserID.to_numpy()
            persist = matrix(saved, "persist_")
            for policy in ("P016", "P021"):
                score = replay if policy == "P016" else blend_score(replay, matrix(saved, "ew7_"), 0.5)
                expected = topk_at_persist(score, persist, ids)
                dec = matrix(frame.iloc[indices], f"dec_{policy}_")
                if not np.array_equal(dec, expected) or not np.array_equal(dec.sum(axis=0), persist.sum(axis=0)):
                    raise AssertionError("Saved decisions/counts do not reproduce registered policy")
            replay_checks.append({"recipe": recipe, "origin": origin, "rows": len(indices), "max_abs_error": 0.0})
            artifacts.append({"recipe": recipe, "origin": origin,
                              "dimensions": fit_meta["preprocessing"]["output_dimensions"],
                              "retained_variance": sum(fit_meta["preprocessing"]["explained_variance_ratio"] or []),
                              "fit_seconds": fit_meta["total_fit_seconds"],
                              "convergence_retries": sum(f["convergence_retry_4000"] for f in fit_meta["target_fits"]),
                              "rss_after_fit_bytes": fit_meta["rss_after_fit_bytes"]})
        for policy in ("P021", "P016"):
            rr, tt = metric_records(official, recipe, policy, frame, p, matrix(frame, f"dec_{policy}_"))
            records.extend(rr)
            per_target.extend(tt)
    metrics = pd.DataFrame(records)
    baseline = metrics.loc[metrics.recipe == "H2"].set_index("origin").combined_f1
    lr = metrics.loc[(metrics.recipe == "lr") & (metrics.policy == "P021")].set_index("origin").combined_f1
    metrics["delta_vs_h2_p021"] = [r.combined_f1 - baseline[r.origin] for r in metrics.itertuples()]
    metrics["delta_vs_lr_p021"] = [r.combined_f1 - lr[r.origin] for r in metrics.itertuples()]
    primary = metrics.loc[(metrics.recipe.isin(RECIPES)) & (metrics.policy == "P021") & (metrics.origin == "pooled")].copy()
    for index, row in primary.iterrows():
        weekly = metrics.loc[(metrics.recipe == row.recipe) & (metrics.policy == "P021") & (metrics.origin != "pooled")]
        primary.loc[index, "weekly_wins_vs_h2"] = int((weekly.delta_vs_h2_p021 > 0).sum())
        primary.loc[index, "worst_week_delta_vs_h2"] = weekly.delta_vs_h2_p021.min()
        primary.loc[index, "followup_screen_pass"] = bool(row.delta_vs_h2_p021 >= 0.001 and
            (weekly.delta_vs_h2_p021 > 0).sum() >= 3 and weekly.delta_vs_h2_p021.min() >= -0.010)
    primary = primary.sort_values(["combined_f1", "recipe"], ascending=[False, True])
    strongest = primary.iloc[0].recipe
    best = predictions.loc[predictions.recipe == strongest].sort_values(["origin", "UserID"]).reset_index(drop=True)
    bootstrap, boot_summary = cluster_bootstrap(controls.UserID.to_numpy(), matrix(controls, "y_"),
        matrix(best, "dec_P021_"), matrix(controls, "dec_H2_P021_"))
    boot_summary["recipe"] = strongest
    report_dir = ROOT / "reports/pca_fisher_20261026" / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    primary.to_csv(report_dir / "comparison.csv", index=False)
    pd.DataFrame(per_target).to_csv(report_dir / "per_target_metrics.csv", index=False)
    pd.DataFrame(artifacts).to_csv(report_dir / "fit_resources.csv", index=False)
    bootstrap.to_csv(report_dir / "bootstrap_draws.csv", index=False)
    json_write(report_dir / "bootstrap_summary.json", boot_summary)
    json_write(report_dir / "replay_checks.json", replay_checks)
    verification = {"status": "pass", "row_predictions": len(predictions), "origin_recipe_replays": len(replay_checks),
                    "target_classifiers": 6 * len(replay_checks), "metric_rows_official_parity": len(metrics),
                    "artifact_hashes_verified": len(manifest["sha256"]), "source_tables_verified": 8,
                    "strongest_recipe": strongest, "any_followup_screen_pass": bool(primary.followup_screen_pass.any()),
                    "report_dir": str(report_dir)}
    json_write(run_dir / "verification.json", verification)
    json_write(report_dir / "verification.json", verification)
    lines = ["# PCA and Fisher/LDA: fixed-recipe development results", "",
        "All five recipes completed. Official scoring, artifact authentication, exact model replay and decision-policy/count checks passed.", "",
        f"Run: `{run_dir.relative_to(ROOT)}`. Source: `{registration['source_sha']}`. Four previously inspected September weeks; these are exploratory development results.", "",
        "| Recipe | Combined F1 (P021) | Delta vs H2/P021 | Delta vs matched LR | Weekly wins vs H2 | Follow-up screen |",
        "|---|---:|---:|---:|---:|---|"]
    for row in primary.itertuples():
        lines.append(f"| {row.recipe} | {row.combined_f1:.6f} | {row.delta_vs_h2_p021:+.6f} | {row.delta_vs_lr_p021:+.6f} | {int(row.weekly_wins_vs_h2)}/4 | {'pass' if row.followup_screen_pass else 'fail'} |")
    lines += ["", f"H2/P021 reference: **{h2_row['combined_f1']:.6f}** combined F1.", "",
        "## Interpretation", "",
        "PCA-versus-LR deltas isolate compression under identical preprocessing and uniform weighting. H2 comparisons assess complete systems: H2 uses recency weighting and different preprocessing. All configurations and both decision policies were fixed before fitting; August tuning outcomes were not used for selection.", "",
        "Report truth uses the authenticated controls' observation cutoff, 2026-10-21T11:59:00Z, which exceeds the minimum S+12d21h maturity cutoff. Prefit checks found 4/8/7/0 late-observed binary labels beyond the minimum lag across the four weeks; the plan was amended before any candidate fit to retain identical truth for every system. Training labels remain limited by each historical decision clock. No October target week was evaluated.", "",
        "P021 blends model-probability rank equally with EW7 rank and predicts the persistence-positive count. P016 is the fixed candidate-only ranking diagnostic at the same count. AP is computed from raw model probabilities, separately from the decision policies; pooled-table AP is the equally weighted mean across origins and targets.", "",
        f"The strongest candidate is `{strongest}`. Its paired stable-UserID bootstrap delta interval versus H2/P021 is [{boot_summary['delta_95_percentile_interval'][0]:+.6f}, {boot_summary['delta_95_percentile_interval'][1]:+.6f}] (1,000 draws). All repeated-week rows for a sampled user remain together. This interval is exploratory because the strongest of five recipes was selected and only four weeks were evaluated.", "",
        ("At least one candidate passes the development follow-up screen. Freeze any proposed challenger separately before evaluating unseen outcomes; no submission policy was changed."
         if verification["any_followup_screen_pass"] else
         "No candidate passes the registered development follow-up screen. Close this bounded experiment; retain H2/P021 and do not expand the search."), "",
        "## Evidence", "", "- `metrics.csv`: pooled and weekly F1/AP for all fixed policies and controls.",
        "- `per_target_metrics.csv`: target F1, confusion counts and predicted-positive counts.",
        "- `fit_resources.csv`: dimensions, fit times, convergence retries and RSS after each fit (not peak memory).",
        "- `bootstrap_summary.json`, `bootstrap_draws.csv`: exploratory paired uncertainty.",
        "- `replay_checks.json`, `verification.json`: independent saved-model and official-score checks.",
        "- Run directory: frozen code/config/input hashes, models, report features, keyed probabilities/decisions, fit metadata and artifact hashes.", ""]
    (report_dir / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(primary[["recipe", "combined_f1", "delta_vs_h2_p021", "delta_vs_lr_p021"]].to_string(index=False), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().run_dir), indent=2))


if __name__ == "__main__":
    main()
