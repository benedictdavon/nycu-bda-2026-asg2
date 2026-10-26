# PCA follow-up: ensembles and recency weighting

Twelve fixed candidates evaluated on the same 22,919 user-weeks from four previously inspected September origins. Source `2053302ba75751be975b9e779f567557b66f4071`; report labels and historical training clocks match the parent experiment. All results are exploratory.

## Results

| Candidate | Combined F1 | Delta vs H2 | Delta vs unweighted PCA99 | Weekly wins vs H2 | Screen |
|---|---:|---:|---:|---:|---|
| h2_weighted_pca99_c10_a050 | 0.549904 | +0.002270 | +0.003428 | 3/4 | pass |
| h2_weighted_pca99_c1_a050 | 0.549904 | +0.002270 | +0.003428 | 3/4 | pass |
| h2_weighted_pca99_c01_a050 | 0.549843 | +0.002209 | +0.003367 | 3/4 | pass |
| h2_pca99_a050 | 0.549064 | +0.001429 | +0.002588 | 3/4 | pass |
| h2_pca99_a025 | 0.548908 | +0.001274 | +0.002432 | 3/4 | pass |
| h2_pca99_fisher_equal | 0.548469 | +0.000834 | +0.001993 | 3/4 | fail |
| h2_lr_a050 | 0.548314 | +0.000680 | +0.001838 | 3/4 | fail |
| h2_pca99_a075 | 0.548259 | +0.000624 | +0.001783 | 2/4 | fail |
| h2_fisher_a050 | 0.548081 | +0.000446 | +0.001604 | 2/4 | fail |
| weighted_pca99_c01 | 0.546738 | -0.000897 | +0.000262 | 2/4 | fail |
| weighted_pca99_c1 | 0.546698 | -0.000936 | +0.000222 | 2/4 | fail |
| weighted_pca99_c10 | 0.546698 | -0.000936 | +0.000222 | 2/4 | fail |

Controls: H2/P021 **0.547634**; unweighted PCA99/P021 **0.546476**.

Names `a025/a050/a075` identify PCA's share within the learned-model half of the score. `h2_lr` and `h2_fisher` are fixed equal model blends. The three-way blend assigns equal weights to H2, PCA99 and Fisher within that half. `weighted_pca99_c01/c1/c10` use C=0.1/1/10 in the recency-weighted classifier; their `h2_..._a050` counterparts blend equally with H2.

## Complementarity and interpretation

Across all user-target cells, H2/PCA99 decisions disagree in 3036 cases (2.21%). PCA uniquely selects 487 positive cases missed by H2; H2 uniquely selects 520 positive cases missed by PCA. These are descriptive diagnostics and are not used as label-dependent decision rules. Per-week/per-target rank correlations and selection overlap are in `h2_pca_diversity.csv`.

All candidates preserve EW7's 0.5 share, the persistence-positive count and the same UserID tie rule. Model probability ranks are mixed directly, without reranking the intermediate mixture. Scaler and PCA remain unweighted and exactly reproduce the parent transforms; only the new logistic classifier losses use two-week half-life weights. The C=1 weighted-versus-unweighted PCA99 comparison isolates that loss-weighting change.

Strongest observed candidate: `h2_weighted_pca99_c10_a050`, combined F1 0.549904, delta +0.002270. Its paired UserID-cluster bootstrap delta interval is [+0.000219, +0.004367] (1,000 draws). This interval ignores selection optimism across 12 candidates and earlier campaigns, and does not establish performance on unseen weeks.

At least one recipe passes the registered development follow-up screen. It can be considered for a separately frozen unseen confirmation; no submission policy was changed.

## Verification and evidence

All 72 new target fits completed; 12 saved bundles replayed exactly, with exact parent preprocessing; all 12 candidate policies were reconstructed from saved constituents. All 80 weekly/pooled metric rows match the frozen official evaluator. Parent/source/code/config/input and new-artifact hashes passed.

- `comparison.csv`, `metrics.csv`, `per_target_metrics.csv`: all registered candidates and controls, including weekly/target behavior and counts.
- `h2_pca_diversity.csv`: complementary selections, disagreement and within-origin probability-rank correlations.
- `fit_resources.csv`, `replay_checks.json`, `verification.json`: fit, convergence and replay evidence.
- `bootstrap_summary.json`, `bootstrap_draws.csv`: exploratory uncertainty for the strongest observed candidate.
- AP uses raw probabilities for individual models and a mixture of model probability ranks for ensembles, excluding EW7; `ap_input` identifies the representation. Rank mixtures are not calibrated probabilities.
- Run directory: `outputs\pca_followup_20261026\run_fixed_v1`; frozen plan/config/code, parent hashes, weighted models, constituent probabilities and keyed candidate scores/decisions.
