# Repository structure and naming

Experiment IDs identify research milestones. Descriptive folder suffixes state their purpose.

- Shared implementations: `src/<responsibility>.py`.
- Executable experiments: `experiments/expNNN_descriptive_topic/`.
- Experiment commands: descriptive files such as `build_dataset.py`, `train.py`, `predict.py`.
- Experiment outputs: the experiment's `runs/` subfolder.
- Configs: `configs/<purpose>.json`; controlled variants: `<purpose>_<variant>.json`.
- Tests: descriptive names reflecting behavior, rather than an experiment number alone.

## Migration map (October 14, 2026)

| Original path | Current path |
|---|---|
| `src/build_dataset.py` | Shared functions: `src/dataset_builder.py`; CLI: `experiments/exp005_temporal_dataset/build_dataset.py` |
| `src/train_baselines.py` | `experiments/exp006_baseline_comparison/train.py` |
| `src/predict_exp006.py` | `experiments/exp006_baseline_comparison/predict.py` |
| `src/exp006_io.py` | Shared loading: `src/dataset_io.py`; experiment configuration/splits: `experiments/exp006_baseline_comparison/protocol.py` |
| `src/exp006_models.py` | `src/baseline_models.py` |
| `src/exp006_scoring.py` | `src/metrics.py` |
| `configs/exp005.json` | `configs/temporal_dataset.json` |
| `configs/exp006.json` | `configs/baseline_comparison.json` |
| `experiments/datasets/exp005/` | `experiments/exp005_temporal_dataset/runs/` |
| `experiments/models/exp006/` | `experiments/exp006_baseline_comparison/runs/` |
| `tests/test_exp006_training.py` | `tests/test_baseline_training.py` |
| `tests/test_exp006_scoring.py` | `tests/test_metrics.py` |

Run commands from the project root using the new module paths shown in README.md.
Original commands are preserved in historical experiment log entries as records of
what was actually executed; those old module commands are not current entry points.

## Scientific artifact compatibility

The relocation preserves run folder names, all file contents, model parameters,
configuration values, metrics, and hashes. Historical manifests deliberately retain
their recorded absolute paths and schema keys. `src.paths.resolve_run_path` translates
original dataset/model directory sequences when the original path no longer exists.
An existing path takes precedence; unrelated paths are never rewritten.

Archived model files contain the old Python class path `src.exp006_models.ModelBundle`.
`src.model_io.load_model_bundle` registers the renamed module only when loading a
trusted local bundle. No archived model is rewritten or retrained. New models store
`src.baseline_models.ModelBundle`.

The instructor clone, historical labels, features, thresholds, holdout policy, and
experiment status are unaffected by this structural change. Future experiments,
including EXP-007, still require the agreed approval and evidence workflow.
