# Recommended candidate for a separately frozen unseen comparison

Candidate: **`h2_weighted_pca99_c1_a050`**. This document specifies a recommendation after the registered development experiment; it does not record unseen confirmation or change submission policy.

## Exact recipe

- H2: unchanged 59-column F0 histogram-gradient-boosting recipe, historical clocks, permitted 56 daily training origins and normalized two-week half-life weights.
- PCA branch: signed log1p on continuous F0 inputs, original binary treatment and StandardScaler; PCA retaining 99% variance with full SVD and no whitening. Fit scaler and PCA using only the permitted training rows, with uniform preprocessing weights.
- PCA classifier: six independent logistic regressions, C=1, lbfgs, tolerance 1e-4, max_iter=2000, seed 42; use the same normalized two-week half-life training-row weights as H2 in the classifier loss. The documented convergence-only retry remains max_iter=4000.
- At each new report origin, refit both branches and the training-only PCA transform; do not reuse September fitted models as future models.
- Rank each branch's positive-class probabilities and EW7 scores independently within the scored population for each target, using the existing average-tie ranks.
- Final score: **0.25 × rank(H2) + 0.25 × rank(weighted PCA99+LR) + 0.50 × rank(EW7)**. Do not rerank the intermediate model mixture. EW7 retains its existing tau=7 days and 90-day visibility-time history.
- Predict the existing persistence-positive count per target, with the existing ascending-UserID tie break. No target-specific weights, probability thresholds or prediction-count changes.

## Why this candidate

C=1 and C=10 tied on pooled combined F1 at 0.5499040953014345, +0.0022696217261225504 over H2/P021. C=1 keeps the original LR regularization default and had the smaller worst-week loss in this inspected sample. The candidate recommendation follows the observed tie; it is not presented as a preregistered model-selection decision.

| September origin (UTC) | H2/P021 | Candidate | Delta |
|---|---:|---:|---:|
| Sep 9, 16:00 | 0.549048 | 0.552343 | +0.003294 |
| Sep 16, 16:00 | 0.519272 | 0.520291 | +0.001019 |
| Sep 23, 16:00 | 0.554211 | 0.553000 | -0.001211 |
| Sep 30, 16:00 | 0.554036 | 0.559579 | +0.005543 |
| Pooled | 0.547634 | 0.549904 | +0.002270 |

It passes the registered development screen: gain >=0.001, wins in at least 3/4 weeks, worst weekly delta >=-0.010. Its separate 1,000-draw UserID-cluster bootstrap delta interval is [+0.000201, +0.004349]. This is exploratory and ignores selecting among 12 follow-up candidates and earlier campaigns on the same four inspected weeks.

The C=1 interval is saved in `bootstrap_tied_best_c1.json` and `bootstrap_tied_best_c1_draws.csv`. To reproduce it, call the existing `experiments.pca_fisher_20261026.verify.cluster_bootstrap` with seed=42 and draws=1000, pairing this candidate's `dec_*` columns from the follow-up `predictions.parquet` against the aligned `controls.parquet` `dec_H2_P021_*` columns and `y_*` truth, sorted by origin/UserID. The report's main bootstrap uses C=10, which wins only the deterministic alphabetical display tie.

## Next evaluation boundary

Keep this exact recipe fixed for any future comparison. Identify the unseen origins, compatible source snapshot, population, label-maturity/observation cutoffs and any adoption rule before reading their outcomes. Evaluate both candidate and H2 on identical rows and truth, reporting all reserved origins and pooled results. The current result supports an unseen comparison; its selection-adjusted temporal performance remains unknown.
