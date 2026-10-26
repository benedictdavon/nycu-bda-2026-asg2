# One constrained DFS-style group in the PCA ensemble

Implements `docs/exp/DFS_PCA_PLAN_20261026.md` using the M08/M09 and R15/R16 templates in `docs/FEATURE_ENGINEERING.md`.

This is an explicit, whitelisted synthesis grammar implemented with pandas, rather than a Featuretools package run. For the three audited direct-event types (push, issue comment, review comment), it generates self-excluded neighbor COUNT in 7/30 days and NUM_UNIQUE of other cohort actors in 7/30 days, then stacks MEAN/MAX/population-STD across the focal user's distinct 30-day exposure repositories. There are 24 synthesized statistics plus repository-exposure support. It uses the existing validated relational source adapter and checks visibility of both endpoints at every cutoff. It does not infer full GitHub repository traffic from cohort data.

The augmented branch has 84 raw inputs: existing F0 plus the 25 generated features. It retains train-only signed-log/StandardScaler/PCA99 preprocessing, C=1 logistic classifiers and H2 recency weights in the classifier loss. The fixed primary ensemble remains 25% H2 rank, 25% PCA-branch rank, 50% EW7 rank, with persistence-positive prediction counts. The primary control is the existing 0.549904 F0 ensemble; H2 alone and the standalone branch are additional comparisons. No feature or hyperparameter search is performed.

From the project root:

```powershell
$experimentPython = 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe'
& $experimentPython -B -m experiments.dfs_pca_20261026.run
# Optional: --output outputs/dfs_pca_20261026/<new-run-name>

& $experimentPython -B -m experiments.dfs_pca_20261026.verify `
    --run-dir outputs/dfs_pca_20261026/run_fixed_v1
```

Existing run directories are never overwritten. Source and parent hashes are authenticated, then configuration, feature expressions and code are frozen before new real-data feature generation. The fresh F0 control must reproduce parent probabilities exactly before fitting the augmented branch.

The run saves every generated cutoff cache and audit, source audit, fitted models, fit metadata, training-only feature diagnostics and keyed probabilities/scores/decisions. The verifier rebuilds report features from source, independently traces five user examples, checks training-only means/variances and PCA covariance/orthogonality/variance retention, replays all new models, reconstructs both policies and matches the official scorer. Reports are written under `reports/dfs_pca_20261026/<run-name>/`; only `completion.json` with status complete confirms the full workflow succeeded.

All four September report origins were inspected in previous experiments. Any uplift and paired-user bootstrap interval remain exploratory. This run does not evaluate an October-origin target week or change submission policy.

Focused tests (apply the same Windows BLAS settings as the runners):

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:MKL_ENABLE_INSTRUCTIONS='SSE4_2'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
& $experimentPython -B -m pytest tests/test_dfs_pca.py tests/test_relational_features.py `
    tests/test_pca_followup.py tests/test_pca_fisher.py tests/test_phase_b_rules.py tests/test_metrics.py -q
```

Method references: [Featuretools Deep Feature Synthesis](https://docs.featuretools.com/en/stable/getting_started/afe.html), [DFS API and whitelisting options](https://docs.featuretools.com/en/stable/generated/featuretools.dfs.html). Local self-exclusion and population-STD conventions are explicit in the expression inventory and tests.
