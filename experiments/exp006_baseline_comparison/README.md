# EXP-006: Historical baseline comparison

Compare zero, prior-week persistence, logistic regression, and histogram gradient
boosting on EXP-005 historical datasets. `protocol.py` owns the fixed experiment
configuration and time-slice policy; shared models, dataset loading, and scoring
live in `src/baseline_models.py`, `src/dataset_io.py`, and `src/metrics.py`.

```powershell
python -m experiments.exp006_baseline_comparison.train --dry-run
python -m experiments.exp006_baseline_comparison.train --config configs/baseline_comparison.json --models zero,persistence,logistic,hgb
```

The default input is the latest valid build under
`experiments/exp005_temporal_dataset/runs/`. Pin an exact build using `--exp005-run`.
Model runs default to this experiment's `runs/` folder. Existing outputs are protected
from overwrite. Original CLI input option names are retained for compatibility.

The fixed calibration cutoffs are Aug 3/10/17 (Taiwan); Aug 24 is a maturity gap;
primary cutoffs are Aug 31/Sep 7/Sep 14. The October holdout stays sealed.
The measured four-model run's best historical eligible-proxy combined F1 is
persistence **0.454677**; this is not official November performance.

After generating actual final inference features and freezing the model policy:

```powershell
python -m experiments.exp006_baseline_comparison.predict --model-run experiments/exp006_baseline_comparison/runs/run-build-506d860693df-6040f635d4-fc2acd6fe5b2 --inference-run PATH_TO_INFERENCE_RUN --model persistence --test-input assignment2/evaluation/test_input.csv --output outputs/baseline_predictions.csv
```

`PATH_TO_INFERENCE_RUN` must be replaced with the actual generated inference directory.
A changed instructor SHA requires intentional `--allow-new-snapshot` use.
The submission must ultimately use the required `output_{studentID}.csv` filename.

Use `src.model_io.load_model_bundle` to load trusted archived bundles; it resolves
classes stored under the original module name without modifying the saved models.
See `docs/REPOSITORY_STRUCTURE.md` for the migration map.
