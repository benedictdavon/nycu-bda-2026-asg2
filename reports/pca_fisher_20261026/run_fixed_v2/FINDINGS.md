# PCA and Fisher/LDA: fixed-recipe development results

All five recipes completed. Official scoring, artifact authentication, exact model replay and decision-policy/count checks passed.

Run: `outputs\pca_fisher_20261026\run_fixed_v2`. Source: `2053302ba75751be975b9e779f567557b66f4071`. Four previously inspected September weeks; these are exploratory development results.

| Recipe | Combined F1 (P021) | Delta vs H2/P021 | Delta vs matched LR | Weekly wins vs H2 | Follow-up screen |
|---|---:|---:|---:|---:|---|
| pca99_lr | 0.546476 | -0.001158 | +0.000743 | 2/4 | fail |
| lr | 0.545733 | -0.001901 | +0.000000 | 2/4 | fail |
| pca95_lr | 0.545611 | -0.002024 | -0.000122 | 2/4 | fail |
| pca90_lr | 0.545110 | -0.002524 | -0.000623 | 1/4 | fail |
| fisher_lda | 0.544215 | -0.003420 | -0.001518 | 1/4 | fail |

H2/P021 reference: **0.547634** combined F1.

## Interpretation

PCA-versus-LR deltas isolate compression under identical preprocessing and uniform weighting. H2 comparisons assess complete systems: H2 uses recency weighting and different preprocessing. All configurations and both decision policies were fixed before fitting; August tuning outcomes were not used for selection.

Report truth uses the authenticated controls' observation cutoff, 2026-10-21T11:59:00Z, which exceeds the minimum S+12d21h maturity cutoff. Prefit checks found 4/8/7/0 late-observed binary labels beyond the minimum lag across the four weeks; the plan was amended before any candidate fit to retain identical truth for every system. Training labels remain limited by each historical decision clock. No October target week was evaluated.

P021 blends model-probability rank equally with EW7 rank and predicts the persistence-positive count. P016 is the fixed candidate-only ranking diagnostic at the same count. AP is computed from raw model probabilities, separately from the decision policies; pooled-table AP is the equally weighted mean across origins and targets.

The strongest candidate is `pca99_lr`. Its paired stable-UserID bootstrap delta interval versus H2/P021 is [-0.004112, +0.001898] (1,000 draws). All repeated-week rows for a sampled user remain together. This interval is exploratory because the strongest of five recipes was selected and only four weeks were evaluated.

No candidate passes the registered development follow-up screen. Close this bounded experiment; retain H2/P021 and do not expand the search.

## Evidence

- `metrics.csv`: pooled and weekly F1/AP for all fixed policies and controls.
- `per_target_metrics.csv`: target F1, confusion counts and predicted-positive counts.
- `fit_resources.csv`: dimensions, fit times, convergence retries and RSS after each fit (not peak memory).
- `bootstrap_summary.json`, `bootstrap_draws.csv`: exploratory paired uncertainty.
- `replay_checks.json`, `verification.json`: independent saved-model and official-score checks.
- Run directory: frozen code/config/input hashes, models, report features, keyed probabilities/decisions, fit metadata and artifact hashes.
