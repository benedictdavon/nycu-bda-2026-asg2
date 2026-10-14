# Configuration naming

Use descriptive names that state what a configuration controls:

- `temporal_dataset.json`: historical cutoffs, label maturity, lookbacks, and eligibility proxy.
- `baseline_comparison.json`: calibration/report schedule, threshold search, and model parameters.

Controlled variants use `<purpose>_<variant>.json`, for example a future
`baseline_comparison_recent_12_weeks.json`. Create a variant only when an approved
experiment implements that setting. Experiment IDs belong in experiment directories
and provenance, rather than serving as the entire config filename.

Values in the two existing configs are unchanged from `exp005.json` and `exp006.json`.
