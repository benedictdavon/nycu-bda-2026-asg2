# Configs

- `temporal_dataset.json`: cutoffs, label maturity, lookback windows and the user eligibility rule used to build the historical dataset.
- `baseline_comparison.json`: report schedule, threshold search and model parameters for the first baseline comparison.
- `exp011_optional_requirements.txt`: pinned optional packages, included by `environments/requirements-exp011.txt`.
- `environments/`: extra environments for experiments that aren't part of the final pipeline.

Runs accept `--output` to write somewhere other than the configured `output_dir`. They won't overwrite an existing output folder.
