# Focused tests

Executed October 26, 2026 in `C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe`:

```text
python -B -m pytest tests/test_dfs_pca.py tests/test_relational_features.py tests/test_pca_followup.py tests/test_pca_fisher.py tests/test_phase_b_rules.py tests/test_metrics.py -q --basetemp outputs/dfs_pca_test_tmp_20261026
45 passed, 7 warnings in 6.47s
```

The seven warnings concern pandas' future dtype-inference behavior when the synthetic fixture concatenates an empty table with its first row. They do not originate in the production feature builder.

Tests cover an independent Python-loop aggregation oracle and hand counts; focal-user exclusion; event/actor/repository cutoff equality; inclusive lower boundaries; future/noncohort/unresolved rows; namespaced atomic identities; nanosecond/microsecond timestamp conversion; empty exposure and singleton STD; requested user order; preprocessing leakage and serialization; model weighting; rank policy endpoints/counts; official metric conventions and stable-user bootstrap semantics.

Windows BLAS settings matched the experiment: OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4, MKL_ENABLE_INSTRUCTIONS=SSE4_2, MKL_THREADING_LAYER=SEQUENTIAL. Real-source, model-replay and official-scoring checks are recorded by the completed run's verifier.
