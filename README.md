# GitHub Activity Prediction

A reproducible machine learning pipeline for **NYCU Big Data Analytics Techniques and Applications, Fall 2026 — Assignment 2**. The project predicts whether a GitHub user will perform each of six activities during the following week, using the instructor-provided relational dataset.

[Experiment plan](docs/EXPERIMENT_PLAN.md) · [Results and methodology](docs/EXPERIMENT_LOG.md) · [Repository structure](docs/REPOSITORY_STRUCTURE.md) · [Official assignment repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2)

## Task

Each target is binary: `1` means at least one occurrence during the prediction week; `0` means none.

| Target | Activity |
|---|---|
| `open_pr` | Open a pull request |
| `open_issue` | Open an issue, excluding PR conversations |
| `issue_comment` | Comment in an issue or PR discussion |
| `pr_review` | Submit a pull request review |
| `pr_review_comment` | Comment on a pull request diff |
| `push` | Push commits to a branch or tag |

The official prediction interval is **November 12–19, 2026, 00:00 Taiwan time** (`Asia/Taipei`). Evaluation uses **combined F1 = (macro-F1 + micro-F1) / 2**. The official test roster is scheduled for October 28; submission is due November 11 at 23:55 Taiwan time.

Only instructor-provided activity data is used. The eight Parquet tables cover users, repositories, pull requests, issues, discussion comments, reviews, review comments, and pushes.

## Implementation

```text
Official snapshot → data audit → historical features and labels
                 → chronological training → threshold calibration
                 → validation → inference and CSV validation
```

The implemented pipeline provides:

- **Point-in-time features:** activity counts, recency, repository breadth, recent changes, and active-week history, using rows visible strictly before each cutoff.
- **Historical labels:** event-time targets with an explicit observation cutoff and configurable label-maturity buffer.
- **Chronological evaluation:** separate training, threshold-calibration, and reporting periods; October holdout labels remain sealed.
- **Baseline models:** all-zero, previous-week persistence, logistic regression, and histogram gradient boosting.
- **Reproducibility:** snapshot identifiers, configuration and input hashes, model bundles, thresholds, and per-target metrics.

Public user identity is `Users.github_user_id`. Internal row IDs are used only within a single snapshot. Later-known attributes are excluded from model features, and historical scores pool confusion counts across user-weeks using the instructor's F1 convention.

## Historical results

EXP-006 reports the following scores on **16,982 proxy-eligible user-week observations** across the August 31, September 7, and September 14 cutoffs. Models fit on March 2–July 20 samples; thresholds were calibrated on August 3, 10, and 17. All cutoff dates are Taiwan time.

| Model | Macro-F1 | Micro-F1 | Combined F1 |
|---|---:|---:|---:|
| Previous-week persistence | **0.451781** | **0.457574** | **0.454677** |
| Histogram gradient boosting | 0.385487 | 0.412047 | 0.398767 |
| Logistic regression | 0.348296 | 0.414767 | 0.381531 |
| All-zero | 0.000000 | 0.000000 | 0.000000 |

These are historical development-validation results, not official November test performance. The eligibility proxy approximates the future test population. The instructor's calibrated example was used to verify label reconstruction; its perfect agreement is not a predictive score.

The evaluated instructor snapshot is `506d860693df60a698b68aa63a8edb4a984c8e1b`. See the [recorded leaderboard](experiments/exp006_baseline_comparison/runs/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/primary_leaderboard.csv) and [experiment log](docs/EXPERIMENT_LOG.md) for provenance, detailed metrics, and limitations. EXP-006 remains Ongoing pending documentation of its detailed audit.

## Repository layout

```text
├── assignment2/                     # Local instructor clone; ignored by Git
├── configs/                         # Descriptive dataset and model configurations
├── docs/                            # Experiment plan, evidence, and structure guide
├── experiments/
│   ├── exp005_temporal_dataset/      # Dataset construction and recorded runs
│   └── exp006_baseline_comparison/   # Training, prediction, protocol, and results
├── notebooks/                       # Data audit and exploratory analysis
├── src/                             # Shared data, feature, model, and scoring modules
├── tests/                           # Temporal safety and pipeline regression checks
├── outputs/                         # Locally generated reproduction and prediction files
└── environment.yaml                 # Conda environment
```

Experiment-specific workflows live in `experiments/expNNN_descriptive_topic/`; reusable code lives in `src/`. Configs are named by purpose: [temporal_dataset.json](configs/temporal_dataset.json) and [baseline_comparison.json](configs/baseline_comparison.json). See the [structure guide](docs/REPOSITORY_STRUCTURE.md) for module responsibilities and archived-path compatibility.

## Setup

Requirements: Git and a Conda-compatible installation. The environment uses Python 3.11, pandas, NumPy, PyArrow, scikit-learn, Matplotlib, JupyterLab, and pytest. The current pipeline runs on CPU.

Clone the project and create its environment using PowerShell:

```powershell
git clone https://github.com/benedictdavon/nycu-bda-2026-asg2.git
cd nycu-bda-2026-asg2

conda env create -f environment.yaml
conda activate nycu-bda-asg2

git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
```

Run subsequent commands from the project root with `nycu-bda-asg2` activated. Skip cloning a repository that is already present. Keep personal code and outputs outside the replaceable `assignment2/` directory. After dependency changes, update the environment with `conda env update -f environment.yaml --prune`.

### Explore the data

```powershell
jupyter lab
```

Run the notebooks in order:

| Notebook | Purpose |
|---|---|
| [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Schema, key integrity, cohort, coverage, and missingness |
| [02_deeper_data_analysis.ipynb](notebooks/02_deeper_data_analysis.ipynb) | Activity regimes, observation delays, persistence, and repository context |
| [03_label_and_split_validation.ipynb](notebooks/03_label_and_split_validation.ipynb) | Historical labels, split integrity, and exploratory baselines |

Select the project environment as the notebook kernel. After a dataset refresh, restart the kernel and rerun the audit.

## Reproduce the pipeline

The commands below write fresh artifacts beneath `outputs/reproduction/`, separate from the recorded experiment results. Repeated execution with the same snapshot and configuration refuses to overwrite an existing run; use a new output base when preserving a previous reproduction.

### Verify label reconstruction and build datasets

```powershell
python -m experiments.exp005_temporal_dataset.build_dataset --mode sample-check --require-perfect

python -m experiments.exp005_temporal_dataset.build_dataset `
  --mode build `
  --config configs/temporal_dataset.json `
  --output-dir outputs/reproduction/datasets
```

The sample check is anchored to the calibrated instructor snapshot. Historical observations may change in later releases; inspect discrepancies rather than treating a changed sample as a prediction failure.

### Validate the split and compare baselines

```powershell
python -m experiments.exp006_baseline_comparison.train `
  --exp005-base outputs/reproduction/datasets `
  --dry-run

python -m experiments.exp006_baseline_comparison.train `
  --exp005-base outputs/reproduction/datasets `
  --config configs/baseline_comparison.json `
  --models zero,persistence,logistic,hgb `
  --output-base outputs/reproduction/models
```

The CLI prints the selected input and generated run directories. Use `--exp005-run` to pin a specific dataset build. Each model run records pooled and per-target metrics, weekly evaluator checks, calibrated thresholds, and fitted bundles.

For additional options, consult the [dataset workflow](experiments/exp005_temporal_dataset/README.md) and [baseline workflow](experiments/exp006_baseline_comparison/README.md), or invoke either command with `--help`.

### Run tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q --basetemp=.pytest_tmp/checks
```

The suite checks feature boundaries, future-row invariance, label construction, split maturity, holdout protection, scorer parity, output schemas, and archived-model compatibility. The live instructor-sample test is opt-in; enable it with `$env:RUN_ASG2_LIVE_SAMPLE = "1"` before running pytest.

## Final inference

Once the official test roster and intended instructor snapshot are available, construct inference features:

```powershell
python -m experiments.exp005_temporal_dataset.build_dataset `
  --mode inference `
  --config configs/temporal_dataset.json `
  --output-dir outputs/reproduction/datasets
```

Then run the predictor with the selected model and the actual directories printed by the earlier commands. Replace the three uppercase path/filename values below:

```powershell
python -m experiments.exp006_baseline_comparison.predict `
  --model-run "MODEL_RUN_DIRECTORY" `
  --inference-run "INFERENCE_RUN_DIRECTORY" `
  --model persistence `
  --output "output_STUDENT_ID.csv"
```

Persistence is the current historical control; final model selection requires the documented evaluation process. A different inference snapshot requires an intentional `--allow-new-snapshot` flag after checking compatibility.

The required CSV header is:

```csv
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

The file must contain exactly one row per official test user, with binary target values. Validate the completed file against the official roster:

```powershell
python assignment2/evaluation/evaluate.py `
  --pred "output_STUDENT_ID.csv" `
  --input assignment2/evaluation/test_input.csv
```

The final supplied coverage is scheduled to end at November 11, 20:00 Taiwan time; the prediction cutoff is November 12, 00:00. Features retain the target cutoff and use only available earlier observations.

## Dataset updates and submission

The instructor replaces the repository snapshot daily at approximately 21:00 Taiwan time through November 11. Follow the [official update instructions](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2#updating-to-the-latest-version): verify the disposable clone, preserve any needed local snapshot, and re-clone rather than pull. Record the new instructor SHA and rerun the data audit.

Git excludes the instructor clone, Parquet datasets, model binaries, per-user historical predictions, caches, and local reference files. Compact experiment metrics, manifests, configurations, and source code are versioned.

The assignment requires a public source repository, `output_{studentID}.csv`, an English slide deck of at most 10 pages, and an 8–12 minute student-recorded presentation. Slides and narration must address the schema/entity graphs, relational modeling, temporal modeling, and complete pipeline. Confirm final upload filenames and requirements against the [course instructions](https://e3p.nycu.edu.tw/course/view.php?id=25898).
