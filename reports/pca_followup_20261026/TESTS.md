# Focused test execution

Executed October 26, 2026 with `C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe`, the same runtime as the experiment.

```text
python -B -m pytest tests/test_pca_followup.py tests/test_pca_fisher.py tests/test_phase_b_rules.py tests/test_metrics.py -q --basetemp outputs/pca_followup_test_tmp_20261026
.................................. [100%]
34 passed in 3.04s
```

Windows BLAS environment: OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4, MKL_ENABLE_INSTRUCTIONS=SSE4_2, MKL_THREADING_LAYER=SEQUENTIAL.

Coverage includes fixed-rank blend endpoints, unchanged EW7 contribution, absence of intermediate reranking, rejection of invalid constituent/weight inputs, preservation of prediction counts and stable key alignment, uniform-weight equivalence to the parent PCA classifier, nonuniform classifier-loss weighting, exact preprocessing reuse, constant/singular features, model serialization replay, official-metric conventions, and paired-user bootstrap behavior. Real-source and saved-artifact checks are recorded separately in each completed run's verification report.
