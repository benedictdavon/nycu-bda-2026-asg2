# Configuration naming

Use descriptive names that state what a configuration controls:

- `temporal_dataset.json`: historical cutoffs, label maturity, lookbacks, and eligibility proxy.
- `baseline_comparison.json`: calibration/report schedule, threshold search, and model parameters.
- `baseline_diagnostics.json`: frozen EXP-005/EXP-006 runs and input hashes, unchanged features/protocol, diagnostic threshold search and output location.
- `chronological_threshold_transfer.json`: frozen EXP-006 control, report origins, mature-history window policies, and hashes for the EXP-008 replay inputs.
- `hgb_recency.json`: frozen HGB archive, full-history and recent-eight-week controls, report schedule, rolling maturity policies, and pinned EXP-009 inputs.
- `push_combinations.json`: frozen push-score sources, fixed 13-policy candidate set, three-week mature calibration, threshold grid, and pinned EXP-010 inputs.
- `stronger_tabular.json`: EXP-011's six fixed learner configurations, original 59-column order, separate archival/4-hour-gap schedules, earlier selection clock and decision rules.
- `stronger_tabular_dependencies.json`: exact optional CPU-library versions, wheel hashes, license and dependency metadata for the isolated local installation.
- `temporal_feature_ablations.json`: EXP-012's fixed HGB/F0 control, 24 rhythm and 24 state columns, chronology, conditional union/removal rules, calibration policy and execution budget.
- `exp011_optional_requirements.txt`: pinned optional packages included by `configs/environments/requirements-exp011.txt`; the latter also pins XGBoost for a fresh environment.

Controlled variants use `<purpose>_<variant>.json`, for example a future
`baseline_comparison_recent_12_weeks.json`. Create a variant only when an approved
experiment implements that setting. Experiment IDs belong in experiment directories
and provenance, rather than serving as the entire config filename.

Values in the original dataset and baseline-comparison configs are unchanged from `exp005.json` and `exp006.json`. EXP-008–010 configs define isolated replay settings and output destinations; they do not rewrite those production controls or the frozen archive.

Recorded runs can override `output_dir` with `--output`. The plan names each audited
final run explicitly; choose a new direct child of the appropriate experiment's
`runs/` directory for reproduction. The CLI refuses to overwrite existing outputs.
