# EXP-005: Temporal dataset construction

This workflow builds point-in-time historical features and future-event labels,
checks the calibrated instructor example, and creates inference features.
Shared implementations live in `src/dataset_builder.py`, `src/features.py`,
`src/labels.py`, `src/data_loader.py`, and `src/temporal_splits.py`.

Run from the personal project root in the project Conda environment:

```powershell
python -m experiments.exp005_temporal_dataset.build_dataset --mode sample-check --require-perfect
python -m experiments.exp005_temporal_dataset.build_dataset --mode build --config configs/temporal_dataset.json
```

Historical builds default to `runs/build-<snapshot>-<config-hash>/` beneath this
experiment folder. They refuse overwrite by default and do not export holdout labels.
The calibrated example agreement is label-reconstruction verification, not predictive F1.

Once the official roster and intended instructor snapshot are available:

```powershell
python -m experiments.exp005_temporal_dataset.build_dataset --mode inference --config configs/temporal_dataset.json
```

Inference uses the target cutoff `2026-11-11T16:00:00Z`, the exact official roster,
and available earlier rows. It writes features, not a submission prediction CSV.
Use `--help` for data-directory and output-directory overrides.

Saved run bytes and manifests are unchanged after relocation. Original recorded paths
are explained in `docs/REPOSITORY_STRUCTURE.md`.
