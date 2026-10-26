# PCA follow-up: complementarity, fixed rank ensembles and recency-weighted LR

Registered October 26, 2026, before computing follow-up candidate scores or fitting follow-up candidates.

The user authorized further experiments after PCA99+LR/P021 reached 0.546476 versus H2/P021 0.547634. The question is whether PCA99 adds complementary ranking information, or whether adapting its classifier to recent activity improves the complete prediction system.

## Fixed scope

Reuse the authenticated `outputs/pca_fisher_20261026/run_fixed_v2` artifacts and their preserved source `2053302b`, ordered F0 features, four September report origins, historical training clocks, and saved report truth at `2026-10-21T11:59:00Z`. No October-origin outcome is read. Revalidate the parent run and cache manifests. Freeze this plan, executable implementation/configuration, runtime, parent hashes and input-manifest hashes before computing any follow-up candidate scores.

The report weeks have already been inspected. All results are exploratory; the follow-up is not a fresh holdout. A fixed family of 12 candidates is evaluated without adaptive additions, per-target switches, or changes to prediction counts. All candidates keep P021's EW7 share at 0.5 and use the persistence-positive count per target with the existing UserID tie break.

## Phase A: six ensembles, using saved predictions

Within each report population and target, rank each constituent model's probabilities independently with the existing average-tie rank convention. For model weights summing to one, use:

`decision_score = 0.5 * sum(model_weight * model_probability_rank) + 0.5 * EW7_rank`.

Do not rank the intermediate mixture again. This preserves the intended EW7 share. Predict the top k using the existing persistence-count policy.

1. H2/PCA99 model weights 0.75/0.25.
2. H2/PCA99 model weights 0.50/0.50.
3. H2/PCA99 model weights 0.25/0.75.
4. H2/plain LR model weights 0.50/0.50: matched ensemble control for the value added by PCA.
5. H2/Fisher-LDA model weights 0.50/0.50: comparison with a supervised linear representation.
6. H2/PCA99/Fisher-LDA equal model weights 1/3 each.

Also report H2/PCA99 probability-rank correlation, decision disagreement, overlap in selected positives, and the number of true positives uniquely selected by each system, per target and week. These diagnostics explain complementarity; label-dependent diagnostics do not become decision rules.

## Phase B: three classifier fits, each with two fixed policies

Reuse the exact parent PCA99 preprocessing for each origin: signed log, StandardScaler and PCA remain fitted only on that origin's original unweighted training rows. Check the parent input-value/training-key digests and fitted preprocessing replay before reuse. No custom weighted PCA implementation or resampling is introduced.

Refit only logistic regression on those PCA features using the existing H2 two-week half-life row weights, normalized to mean one, and `C in {0.1, 1.0, 10.0}`. Other settings remain the parent's LR defaults (`lbfgs`, tolerance 1e-4, maximum 2000 iterations; one recorded retry at 4000 if necessary). Fit six independent binary targets for four report origins: 72 target fits total. Constant-label targets receive constant probabilities. The preprocessing remains unweighted; only the classifier loss is recency-weighted.

For each C, report (a) its standalone P021 model/EW7 ranking and (b) its equal-model-weight blend with H2 using the Phase A formula. This adds six candidates. The weighted C=1 versus saved unweighted PCA99/C=1 contrast isolates classifier recency weighting. The three C values probe a small predeclared regularization range.

## Evaluation, verification and stopping

- Compare all 12 candidates with saved H2/P021 and saved unweighted PCA99/P021 on exactly matching rows/truth. Include weekly/pooled combined, macro and micro F1, target metrics/counts, model-ranking AP, fit times and convergence/replay records. Label AP inputs: mixture-of-model-ranks for ensembles, raw probabilities for single models. No probability calibration is inferred for rank mixtures.
- Save keyed model scores, final decision scores and binary decisions, plus weighted LR bundles and manifests. Recompute policies from the saved constituents and replay all new models. Check every score against the frozen official evaluator.
- Report a paired stable-UserID 1,000-draw bootstrap for the strongest candidate versus H2. Its interval is exploratory and ignores selection optimism across this follow-up and previous campaigns.
- Retain the prior follow-up screen: pooled gain >=0.001 over H2, positive weekly deltas in at least 3/4 origins, and no weekly delta below -0.010. Passing identifies a challenger for separately frozen unseen confirmation; it does not authorize promotion. Report every candidate even if an earlier result is negative or positive. Stop after the registered family.
- Implementation and artifacts use new `experiments/pca_followup_20261026`, `outputs/pca_followup_20261026`, and `reports/pca_followup_20261026` directories. Parent experiment code, parent artifacts and submission behavior remain unchanged. Proper chronological out-of-fold predictions would be required before a learned stacking model; this bounded follow-up uses fixed weights instead.
