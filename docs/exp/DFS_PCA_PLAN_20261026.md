# Constrained relational synthesis in the PCA ensemble

Date: October 26, 2026. Written before new feature scores or candidate fits.

## Question and fixed design

Does one cutoff-safe, synthesized repository-neighbor feature group improve the recency-weighted PCA99 branch and its fixed ensemble? The user authorized trying DFS after discussion of `docs/FEATURE_ENGINEERING.md`. Catalogue mappings: M08/M09 (whitelisted COUNT/NUM_UNIQUE and stacked MEAN/MAX/STD), R15/R16 (other-cohort repository activity), and the existing direct-event repository adapter. Earlier feature campaigns rejected several additions with HGB; this experiment asks a separate question about the new PCA branch and does not claim the signals have never been considered.

Featuretools is not installed in the project runtime. Implement an explicit, bounded DFS-style expression grammar with pandas, rather than an unrestricted Featuretools search or a dependency installation. References: https://docs.featuretools.com/en/stable/getting_started/afe.html and https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html explain stacked primitives and whitelisting/depth controls. Population STD (ddof=0) and the focal-user exclusion below are explicit local semantics, not claims of matching every Featuretools default.

## The single feature group (25 generated columns)

For user u at exclusive feature cutoff C, exposure repositories are the distinct repositories with an eligible direct event by u in [C-30d,C). Use only Pushes, IssueComments and PRReviewComments from the existing validated `RelationalFeatureBuilder._load` boundary: source primary keys remain namespaced and unique; user/repository joins are many-to-one; only cohort actors, known stable repo identities, and actor/repository endpoints visible strictly before C contribute. Do not use PR/issue/review fields whose initial availability is ambiguous, noncohort traffic, future attributes, labels, or external data.

Systematically enumerate:

- COUNT(other cohort events of each direct type), windows 7 and 30 days, then MEAN/MAX/population-STD over u's distinct 30-day exposure repositories: 3 types x 2 windows x 3 primitives = 18 features.
- NUM_UNIQUE(other cohort actors across the three direct types), windows 7 and 30 days, then the same three repository-to-user aggregations: 6 features. Deduplicate within each repository; this is a statistic of repository-level neighbor counts, not a union of neighbors across repositories.
- COUNT(exposure repositories): 1 support feature.

Subtract u's own event count before repository aggregation; for NUM_UNIQUE subtract one only where u has at least one eligible event in that window. Do not duplicate rows by joining raw user and neighbor events. Empty exposure gives feature zeros and support=0; one-exposure population STD=0. These conventions distinguish absent exposure via support. No neighbor caps or sampling approximations. Windows include their lower boundary and exclude C. Both endpoint first-seen times must be < C even when the event itself is older.

Save ordered expressions, primitives, catalogue mapping, windows, self-exclusion, empty/singleton conventions and source hashes. Audit every generated cutoff and cache by origin and ordered population digest; preserve per-feature coverage, zero rate, variability and exact duplicates. Training-only diagnostics do not select or drop features: constant/duplicate columns are retained for this fixed representation, just as in the parent PCA pipeline.

## Fixed comparison

- Preserve source `2053302ba75751be975b9e779f567557b66f4071`, F0 schema, source/cache authentication, September 9/16/23/30 report origins, same 56 historical daily training origins per report origin and decision-time label cutoffs.
- Report truth remains the parent controls' authenticated `2026-10-21T11:59:00Z` observation cutoff, above the minimum maturity cutoffs. No October-origin outcomes are read.
- Baseline branch: the saved weighted PCA99/logistic C=1 from `outputs/pca_followup_20261026/run_fixed_v1`. Baseline ensemble: 0.5499040953014345 combined F1. H2/P021 context: 0.5476344735753119.
- New branch: concatenate the 59 F0 columns and 25 generated columns (84 inputs), signed-log continuous inputs and original binary treatment, train-only StandardScaler, full-SVD PCA retaining 99% variance, no whitening, then the same six independent C=1 logistic classifiers and normalized H2 two-week half-life classifier weights. Preprocessing remains uniformly weighted. No hyperparameter or retained-variance search.
- Primary ensemble: **0.25 rank(H2) + 0.25 rank(new branch) + 0.50 rank(EW7)**, with unchanged persistence-positive counts and UserID tie break.
- Diagnostic: new branch's standalone P021 (0.50 branch rank + 0.50 EW7 rank). Compare both to their respective saved F0-only controls.
- Refit the F0-only PCA+weighted-LR control at the first report origin and require exact parent-probability parity before fitting the augmented branch. Authenticate all other origin training/input hashes against parent fit metadata. The scope is 24 candidate classifiers plus 6 control-parity classifiers.

## Verification, outputs and interpretation

Before real fits, run synthetic tests with hand/independent-oracle count checks, cutoff equality and lower boundaries, future-event/endpoint perturbations, noncohort/unresolved exclusions, self-exclusion, namespaced event identities, empty/singleton behavior and requested user order. Reuse preprocessing/model, policy, and metric tests from the parent experiments. Freeze code/config/plan and hashes before generating candidate features for the real-data experiment.

Rebuild every report feature block from source for independent cache parity, and verify training-only scaler/PCA against recomputed statistics/transforms. Save models, keyed raw probabilities, final rank scores and decisions, feature caches/manifests, source and cutoff audits, expression inventory and training-only feature diagnostics. Reload all new models and recompute policies/counts; compare weekly and pooled scores against the frozen official evaluator. Verify input and artifact hashes.

Report combined/macro/micro F1, per-week and per-target counts/F1, branch-ranking AP, generated dimensions/coverage, runtime, and paired stable-UserID 1,000-draw bootstrap deltas for the primary ensemble against both the existing ensemble and H2. No selection-adjusted inference or unseen confirmation is claimed on these previously inspected weeks.

For a follow-up candidate, require primary pooled gain >=0.001 against the existing ensemble, wins in at least 3/4 weeks against it, and no weekly delta below -0.010. Report the full result even if this screen fails; stop after this one feature group. Keep code/artifacts under new `experiments/dfs_pca_20261026`, `outputs/dfs_pca_20261026` and `reports/dfs_pca_20261026` directories. Parent code, parent artifacts and submission behavior are not changed.
