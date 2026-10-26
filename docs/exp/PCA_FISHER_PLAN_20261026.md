# PCA and Fisher linear discriminant experiment plan

Date: October 26, 2026 (Asia/Taipei).
Status: planning complete; no candidate implementation or model fits performed.

## Question and scope

Test whether compression of correlated F0 features or supervised linear separation improves GitHub activity prediction. This is a bounded development experiment with five fixed recipes. Reuse the existing point-in-time data, population, training-origin, decision-policy and official-scoring functions through a new experiment runner. Extending the global baseline model registry was considered; an isolated runner keeps the comparison reproducible without changing submission behavior or historical experiments.

## Fixed inputs and evaluation boundary

- Instructor source: preserved snapshot `2053302ba75751be975b9e779f567557b66f4071`, under `assignment2_backup/2053302ba75751be975b9e779f567557b66f4071_20261023/assignment2_moved_aside/data/db`.
- Authenticate the table hashes, ordered 59-column F0 schema, populations, and saved control predictions against their existing manifests before candidate fitting. Reuse authenticated feature/label caches where available; otherwise rebuild through the accepted builders with matching clocks.
- Report origins: September 9, 16, 23, and 30 at 16:00 UTC, matching the existing controls. Reuse the 56 permitted daily training origins for each report origin, with point-in-time features and labels available by its decision clock. Report labels use the established maturity convention and must agree with saved control truth.
- Do not use August tuning origins to select configurations: the prior group-elimination experiment documented collection-gap distortion there. All five recipes below are fixed before fitting and all five are reported.
- These September outcomes have been inspected in previous experiments. Treat results as exploratory development evidence, not independent confirmation. Do not read October outcomes or change the submission policy in this experiment.

## Five model recipes

Each recipe fits six independent binary classifiers, one per activity. No PCA-to-LDA chain or additional blends are searched.

| Recipe | Configuration |
|---|---|
| LR control | LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, tol=1e-4, class_weight=None, random_state=42) |
| PCA90 + LR | PCA(n_components=0.90, svd_solver='full', whiten=False), then the identical LR |
| PCA95 + LR | PCA(n_components=0.95, svd_solver='full', whiten=False), then the identical LR |
| PCA99 + LR | PCA(n_components=0.99, svd_solver='full', whiten=False), then the identical LR |
| Fisher/LDA | LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=None) |

Preprocessing is identical across recipes: signed log1p on continuous F0 columns; binary indicators remain 0/1 before standardization; StandardScaler fitted on training rows only. Preserve the existing missing-recency encoding through the signed transformation. Fit PCA only on transformed training rows, once per retained-variance recipe and origin, sharing that unsupervised transform across targets. Record retained dimensions and explained variance. LDA is used directly as a classifier; each binary target has at most one Fisher discriminant direction, and the lsqr classifier does not require transform().

Use uniform training-row weights for all five recipes. Local scikit-learn 1.9.1 inspection confirms that PCA.fit and LinearDiscriminantAnalysis.fit do not accept sample_weight. Do not silently approximate H2 weights by resampling. PCA-vs-LR comparisons isolate the effect of compression; comparisons against H2/P021 assess complete systems and also differ in weighting and preprocessing. No claim that PCA alone explains a difference against H2 is permitted.

Handle single-class training targets with constant probabilities. Fail on nonfinite inputs or predictions, inconsistent target ordering, failed convergence, or mismatched population keys. A convergence failure may receive one documented increase in iteration budget, without changing model regularization or selecting on scores.

## Decisions, controls and metrics

- Primary: existing P021 decision rule applied to each candidate's probabilities: equal-weight candidate-probability rank and EW7 rank, followed by top-k at the persistence-positive count per target, with the established UserID tie-break.
- Diagnostic: candidate-only ranking at the same persistence-positive count. This is fixed in advance and does not introduce probability-threshold tuning.
- Controls: saved H2/P021 and persistence on identical rows and truth. Verify the saved H2/P021 pooled combined F1 of 0.5476344735753119 before interpreting candidate scores.
- Report pooled combined F1, macro-F1, micro-F1, each week's combined F1, per-target F1, predicted-positive counts, mean-origin six-target AP, and elapsed fit time. Distinguish candidate raw AP from decision-policy results.
- Compare each PCA recipe against the matched LR control and each candidate P021 against H2/P021. Recompute scores from saved row-level predictions using the official evaluator.
- For the strongest candidate, report a paired 1,000-draw bootstrap by stable UserID, retaining all report-origin rows for a sampled user. Label its interval exploratory: the candidate is selected from five recipes and four weeks do not establish broad temporal robustness.

Development follow-up screen: pooled P021 combined-F1 gain of at least 0.001 over H2/P021, positive weekly deltas on at least three of four origins, and no weekly delta below -0.010. This screen only identifies a candidate worth freezing for future unseen evaluation. It does not authorize automatic promotion or make the historical weeks an untouched holdout. Report smaller or inconsistent gains as such. If no candidate passes, close the experiment without expanding the search.

## Implementation and verification sequence

1. Freeze a machine-readable registration and hashes of this plan, configuration, implementation, runtime, source tables, and control inputs before candidate fitting.
2. Add `experiments/pca_fisher_20261026/run.py` with a small experiment-local preprocessing/model module if needed. Reuse existing builders and policy/scoring functions rather than copying their rules. Store run artifacts under a new directory in `outputs/pca_fisher_20261026/` and the findings under `reports/pca_fisher_20261026/`.
3. Add focused tests for fit-only standardization/PCA, label-free PCA input, single-class predictions, six-target ordering, singular/constant feature handling, and deterministic replay. Verify that held-out perturbations do not change fitted preprocessing parameters. Reuse existing policy and scorer tests.
4. Complete source/control preflight and a runtime smoke check. Then fit the five fixed recipes across four origins: 120 target classifiers, with PCA shared across targets. Record resources and convergence; runtime changes must not change statistical recipes.
5. Save keyed row-level probabilities and decisions, fitted preprocessing/models, effective PCA dimensions, configuration/runtime manifests, and resource logs. Reload models for a prediction replay and independently recompute official scores from saved artifacts.
6. Write `FINDINGS.md` and compact CSV comparisons, stating whether PCA helps the matched LR and whether any complete candidate improves H2/P021. Document the uniform-weight comparison and previously inspected development weeks prominently. Keep any future confirmation proposal separate from this completed development experiment.

No model fitting, candidate-score inspection, or submission-policy change is part of this planning step.

## Prefit implementation amendment: report truth observation cutoff

Source/control preflight, before any candidate fit, found that report truth rebuilt at exactly S+12d21h differs from the saved controls in 4 / 8 / 7 / 0 binary cells across September 9 / 16 / 23 / 30. These are late-observed PR/issue labels. Rebuilding at the saved control configuration's documented report-label as-of (`2026-10-21T11:59:00Z`) reproduces all four control-truth blocks exactly.

Therefore the primary comparison uses that authenticated saved-control observation cutoff, which exceeds the minimum maturity cutoff for every report origin. Training labels still use each historical decision clock. Preflight records both the minimum maturity cutoff and the actual report-label cutoff, plus the diagnostic mismatch counts. This resolves the plan's incompatible requirements of exact minimum-lag truth and exact saved-control truth; it preserves the intended comparison to H2/P021 = 0.5476344735753119 on identical labels. No October target week is constructed or read, and no candidate results informed this amendment. The failed preflight log is retained as `outputs/pca_fisher_20261026_run_fixed_v1.log`.
