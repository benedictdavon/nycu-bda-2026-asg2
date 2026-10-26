# Constrained DFS-style features in the PCA ensemble

One predeclared repository-neighbor group was synthesized from the three audited direct-event tables. This is an explicit pandas expression grammar, not an unrestricted Featuretools run. It stacks self-excluded COUNT/NUM_UNIQUE with MEAN/MAX/population-STD over recent exposure repositories, plus exposure support: 25 columns added to F0 for 84 raw inputs.

## Results

| System | Combined F1 | Delta vs existing ensemble | Delta vs H2 |
|---|---:|---:|---:|
| H2 | 0.547634 | -0.002270 | +0.000000 |
| F0_branch | 0.546698 | -0.003206 | -0.000936 |
| F0_ensemble | 0.549904 | +0.000000 | +0.002270 |
| augmented_branch | 0.546651 | -0.003253 | -0.000983 |
| augmented_ensemble | 0.550417 | +0.000513 | +0.002783 |

Primary augmented ensemble: 3/4 weekly wins versus the existing ensemble; worst weekly delta -0.000261. Registered follow-up screen: **FAIL**.

Paired UserID bootstrap intervals for the primary delta (1,000 draws): vs existing ensemble [-0.000406, +0.001422]; vs H2 [+0.000758, +0.004806].

## Interpretation and scope

This compares one synthesized group under a fixed PCA99/C=1 classifier and fixed 25% H2 / 25% PCA branch / 50% EW7 rank mixture. Classifier training recency, prediction counts, population, source release, historical label clocks and report truth match the parent. No feature, model-weight, C or retained-variance search was performed. All 25 features were retained before PCA, including any constant or duplicate columns; training-only feature diagnostics are descriptive.

The four September origins were inspected previously, and the parent ensemble was selected after earlier campaigns. Results and bootstrap intervals are exploratory; they do not establish unseen temporal performance. No October-origin outcome was evaluated and no submission policy changed.

The augmented ensemble has the higher observed development score. It passes the weekly-win and worst-week guards, but its +0.000513 pooled increment is below the predeclared +0.001 minimum, so the development screen fails on effect size. The interval versus the existing ensemble includes zero. Retain this as a small positive exploratory result, with the F0 ensemble as the incumbent comparator for any separately frozen unseen evaluation. This bounded feature experiment is complete.

The standalone PCA branch changes from 0.546698 to 0.546651, essentially unchanged; the observed benefit appears in its combination with H2. This single-group experiment supports only a small, uncertain incremental ensemble effect. The broader usefulness of feature synthesis and other relational/workflow groups remains unresolved.

## Verification and evidence

Source/cache/parent/code hashes passed. All 79 feature cutoffs passed event and endpoint timing audits. The four report feature blocks rebuilt exactly from source. Fresh F0 control probabilities reproduced the parent exactly before candidate fitting. All 24 augmented target fits completed; four bundles replayed exactly, and training-only scaler/PCA statistics and covariance/eigenbasis checks passed. All 25 weekly/pooled metric rows match the official evaluator; both candidate policies and prediction counts reconstruct exactly.

- `metrics.csv`, `comparison.csv`, `per_target_metrics.csv`: weekly/pooled/target evidence and AP input definitions.
- `feature_inventory.json`, `training_feature_diagnostics.csv`: exact generated expressions, support/zero rates, variability and exact F0 duplicates.
- `fit_resources.csv`, `preprocessing_checks.json`, `report_feature_rebuilds.json`, `replay_checks.json`, `verification.json`: source, training-statistic and model checks.
- `bootstrap_vs_existing_ensemble.*`, `bootstrap_vs_H2.*`: exploratory paired uncertainty. CSV comparator-score columns are named generically.
- Run directory `outputs\dfs_pca_20261026\run_fixed_v1`: frozen plan/config/code, source/cutoff audits, every feature cache and manifest, saved models, probabilities and binary decisions.
