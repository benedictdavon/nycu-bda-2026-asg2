# NYCU Big Data Analytics — Assignment 2

**Course:** Big Data Analytics Techniques and Applications (Fall 2026)  
**Assignment:** Predicting GitHub User Activity  
**Student project:** [benedictdavon/nycu-bda-2026-asg2](https://github.com/benedictdavon/nycu-bda-2026-asg2)  
**Official assignment repository:** [nycubigdata-2026fall/BigData2026-Assignment2](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2)

This is my implementation workspace for Assignment 2. It holds my exploratory analysis, future modeling code, experiments, tests, and submission materials. **The instructor's data and evaluation files belong in a separate, replaceable `assignment2/` directory; they are not part of my Git history.**

> **Project status:** Three executed EDA notebooks, a reusable temporal dataset pipeline (EXP-005), and measured baseline comparisons (EXP-006). EXP-006 remains Ongoing until its detailed audit is documented. EXP-007 is proposed and awaits approval. See `docs/EXPERIMENT_PLAN.md` and `docs/EXPERIMENT_LOG.md` for evidence.

## 1. Assignment at a glance

| Item | Requirement |
|---|---|
| Work | Individual |
| Weight | **15%** of the course grade |
| Deadline | **November 11, 2026, 23:55** (Taiwan time); no late submissions |
| Task | For each official test user, predict whether each of six GitHub activities will occur in the following week |
| Prediction period | **November 12, 00:00 to November 19, 00:00, 2026** (`Asia/Taipei`, UTC+8) |
| Model performance | **7%**; score = `(macro-F1 + micro-F1) / 2` |
| Video and slides | **8%**; 8–12 minute video plus English PDF slides (maximum 10 pages) |
| Code | Complete, reproducible source code and instructions in a **public** GitHub repository; all commits before deadline |
| Data | **Only** the files provided in the instructor's assignment repository may be used |

The six binary outputs, in their **required order**, are:

1. `open_pr` — opens a pull request.
2. `open_issue` — opens a non-PR issue.
3. `issue_comment` — comments in an issue or PR discussion thread.
4. `pr_review` — submits a PR review.
5. `pr_review_comment` — comments on a specific line of a PR diff.
6. `push` — pushes commits to a branch or tag.

Each output is `1` if the user performs that activity **at least once** during the target week and `0` otherwise. This is **multi-label binary classification**, not a count-regression task.

For the complete specification, grading breakdown, question requirements (Q1–Q4), and implementation constraints, see the [official assignment PDF on the E3 course page](https://e3p.nycu.edu.tw/course/view.php?id=25898) and, when added to this project, [`docs/ASSIGNMENT_REFERENCE.md`](docs/ASSIGNMENT_REFERENCE.md). The source PDF takes precedence over the secondary notes.

## 2. Repository layout

```text
Assignment 2/
├── README.md
├── environment.yaml
├── assignment2/                         # Replaceable instructor clone; ignored
├── notebooks/                          # Three executed EDA/label notebooks
├── src/                                # Shared modules, named by responsibility
│   ├── data_loader.py
│   ├── dataset_builder.py
│   ├── dataset_io.py
│   ├── features.py
│   ├── labels.py
│   ├── temporal_splits.py
│   ├── baseline_models.py
│   ├── metrics.py
│   ├── model_io.py                      # Archived model loading compatibility
│   └── paths.py                         # Archived run path compatibility
├── configs/
│   ├── temporal_dataset.json
│   └── baseline_comparison.json
├── experiments/
│   ├── exp005_temporal_dataset/
│   │   ├── build_dataset.py
│   │   ├── README.md
│   │   └── runs/                        # Existing dataset manifests and Parquets
│   ├── exp006_baseline_comparison/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── protocol.py                  # Experiment-specific split/config policy
│   │   ├── README.md
│   │   └── runs/                        # Existing models and measured results
│   └── eda_snapshot_history.jsonl
├── tests/
├── docs/
├── outputs/
└── reports/
```

Use `experiments/expNNN_descriptive_topic/` for each executable experiment and
`src/<responsibility>.py` for shared functionality. Configs use descriptive
`<purpose>.json` names, with `<purpose>_<variant>.json` for later controlled variants.
Each experiment keeps generated artifacts in `runs/`.

The `assignment2/` folder is a separate instructor-owned repository, intentionally
ignored. Historical manifests and logged commands retain their original provenance;
see [the structure and migration guide](docs/REPOSITORY_STRUCTURE.md) for current paths.

## 3. Prerequisites

- Git, available on the command line.
- [Miniconda / Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or another Conda-compatible installation.
- VS Code with the Python and Jupyter extensions, **or** JupyterLab.
- Enough local disk space for the instructor's Parquet dataset. Dataset size increases over time.

The provided environment is CPU-oriented for EDA and initial baselines. GPU/deep-learning frameworks are **not** installed yet; they will be added after choosing an architecture.

## 4. Initial setup (Windows PowerShell)

Run these commands from the root of **this project** (the directory containing `environment.yaml`).

### 4.1 Download the official data and evaluator

If you do not already have the `assignment2/` subfolder:

```powershell
git clone --depth 1 `
  https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git `
  assignment2
```

If `assignment2/` already exists, **do not re-clone into the same path**. Confirm that it is the instructor's clone:

```powershell
git -C .\assignment2 remote -v
Test-Path .\assignment2\data\db\Users.parquet
```

The nested repository's `origin` should point to **`nycubigdata-2026fall/BigData2026-Assignment2`**, whereas the root project's `origin` points to **`benedictdavon/nycu-bda-2026-asg2`**. Do **not** add the instructor's repository as the root project's `upstream` remote or merge their histories.

The official repository provides:

- Eight Parquet tables in `assignment2/data/db/`.
- Dataset explanations in [`data/README.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/README.md).
- Complete column definitions in [`data/db/SCHEMA.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/db/SCHEMA.md).
- Evaluation instructions and sample labels in [`evaluation/`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/tree/main/evaluation).

> **Do not fetch more GitHub user activity from the GitHub API, other GH Archive event types, or other websites.** The assignment explicitly restricts data sources to the official files.

### 4.2 Create the Conda environment

```powershell
conda env create -f environment.yaml
conda activate nycu-bda-asg2
```

Optionally register a named Jupyter kernel:

```powershell
python -m ipykernel install --user `
  --name nycu-bda-asg2 `
  --display-name "Python (NYCU BDA A2)"
```

The environment includes **Python 3.11, pandas, NumPy, PyArrow, Matplotlib, JupyterLab, ipykernel, scikit-learn, and pytest**.

If `environment.yaml` changes later, update the existing environment instead of creating a duplicate:

```powershell
conda env update -f environment.yaml --prune
```

**A daily data update does not require reinstalling the environment.**

### 4.3 Run the EDA notebook

In VS Code, open `notebooks/01_data_audit.ipynb`, select the **Python (NYCU BDA A2)** kernel, and use **Restart Kernel and Run All**.

Alternatively, from the project root:

```powershell
conda activate nycu-bda-asg2
jupyter lab
```

Open `notebooks/01_data_audit.ipynb` in JupyterLab.

The notebook automatically looks for the following layout, regardless of whether Jupyter started from the root directory or `notebooks/`:

```text
<project-root>/assignment2/data/db/*.parquet
```

If the instructor's files are stored elsewhere, set the `ASG2_DATA_DIR` environment variable to the folder containing the eight Parquet files **before starting Jupyter**, or edit `DATA_DIR_OVERRIDE` in the first configuration cell:

```powershell
$env:ASG2_DATA_DIR = "C:\path\to\assignment2\data\db"
jupyter lab
```

### 4.4 What the notebook examines

The notebook **reads current Parquet files every time it runs** and performs:

- File/release fingerprinting (official clone SHA, file sizes, metadata and row counts).
- Data types, schema, primary-key/foreign-key checks, and representative records.
- Cohort versus non-cohort users, bot information, and stable GitHub IDs.
- Time coverage and weekly activity-related trends.
- Activity sparsity and recent-history summaries for preliminary baseline planning.
- Relationship/context checks such as repository breadth.
- Missing values and **point-in-time leakage risks**.
- Comparison with the most recent locally recorded dataset snapshot, when enabled.
- Detection of `evaluation/test_input.csv` after its publication.

**Important:** Some apparent `open_pr`, `open_issue`, and `pr_review` trends use *row-visibility proxies* rather than exact action labels. A PR/issue/review may have been created or submitted before its `first_seen_at`. These descriptive EDA charts **must not be mistaken for verified seven-day ground-truth labels**, final test results, or model accuracy.

## 5. Refresh the official dataset (daily, 21:00 Taiwan time)

The instructor publishes a **replacement snapshot at approximately 21:00 Asia/Taipei each day from October 14 through November 11**. The **final release** is scheduled for **November 11 at 21:00**, including data through **20:00** that evening.

Unlike an ordinary incrementally committed repository, the instructor's repo is **force-pushed with a replacement commit**. Old snapshots are not retained in its Git history. For this reason, the official instructions say **re-clone; do not use `git pull`**.

From the root of your personal repository, after closing any notebook kernels or scripts using the files:

```powershell
# WARNING: This deletes only the ignored OFFICIAL clone.
# Keep all personal code, notebooks, and outputs outside assignment2/.
Remove-Item -Recurse -Force .\assignment2

git clone --depth 1 `
  https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git `
  assignment2

# Identify the version fetched
git -C .\assignment2 rev-parse --short HEAD
```

If you need to preserve a previous release, copy or archive it **outside `assignment2/` before removal**. Ensure any local backup is also ignored by your personal Git repository.

Then:

1. Restart the Jupyter kernel so previously loaded tables are discarded.
2. Run **all** cells of `notebooks/01_data_audit.ipynb`.
3. Review the new release fingerprint and structural diagnostics.
4. Compare table counts and coverage against the previous local release.

By default the notebook writes compact, append-only audit entries to:

```text
experiments/eda_snapshot_history.jsonl
```

This file records release metadata and aggregate statistics, **not** raw user-level data. A matching fingerprint is not written twice. Set `SAVE_RELEASE_HISTORY = False` in the notebook to disable it. **The notebook does not automatically download new data or run at 21:00; re-clone and rerun it yourself.**

## 6. Understanding the data safely

The instructor's tables are `Users`, `Repositories`, `PullRequests`, `Issues`, `IssueComments`, `PRReviews`, `PRReviewComments`, and `Pushes`.

For future model development, these rules are essential:

- The official output `UserID` is **`Users.github_user_id`**, **not** the internal `Users.user_id`. Internal PK/FK row numbers may change across daily releases.
- `Users.is_cohort` identifies the **9,000 users whose own activity is collected**. Other referenced users may appear in the `Users` table with incomplete activity histories.
- Timestamps in the Parquet files are **UTC without an explicit timezone**; interpret them as UTC and convert deliberately when creating Taiwan-time windows.
- For **feature availability** at cutoff `t`, filter each table using its designated **row time** (`first_seen_at` for `Users`, `Repositories`, `PullRequests`, `Issues`, and `PRReviews`; `created_at` for `IssueComments`, `PRReviewComments`, and `Pushes`).
- Special columns known only later—`PullRequests.merged_at`, `PullRequests.merged_by_id`, `PullRequests.closed_at`, `Issues.closed_at`, `Issues.state_reason`, and `PRReviews.dismissed_at`—require additional point-in-time handling or exclusion.
- **Split training, validation, and historical testing chronologically.** Do not construct features with data observed after a historical prediction cutoff.
- The official files do **not** contain a ready-made labeled training/validation/test split. Creating correct historical labels requires care, especially for events with `first_seen_at` versus `created_at`/`submitted_at` semantics.

Read the official [`data/README.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/README.md) and [`SCHEMA.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/db/SCHEMA.md) before implementing historical labels or a relational graph.

## 7. Planned modeling and evaluation workflow

The temporal dataset and baseline stages below are implemented. Relational/graph model extensions remain proposed; this is not an instructor-mandated architecture:

```text
Official replacement Parquet snapshot
                  |
                  v
       Re-runnable EDA + checks
                  |
                  v
       Point-in-time dataset builder
       (historical features and labels)
                  |
                  v
       Chronological train / val / test
                  |
                  v
       Baselines -> relational/temporal models
                  |
                  v
       Per-label F1, macro-F1, micro-F1
                  |
                  v
       Final seven-column prediction CSV
```

The measured baselines are all-zero, previous-week persistence, logistic regression, and histogram gradient boosting. Later experiments may test training recency, additional temporal and relational features, and graph methods. **Report results only after measurement.**

### Run the implemented pipeline

From the personal project root with `nycu-bda-asg2` activated:

```powershell
# Read-only calibrated example-label check (not predictive performance)
python -m experiments.exp005_temporal_dataset.build_dataset --mode sample-check --require-perfect

# Build historical features/labels; holdout labels stay sealed
python -m experiments.exp005_temporal_dataset.build_dataset --mode build --config configs/temporal_dataset.json

# Validate source-run selection and time splits without fitting
python -m experiments.exp006_baseline_comparison.train --dry-run

# Baseline comparison; refuses to overwrite an existing run
python -m experiments.exp006_baseline_comparison.train --config configs/baseline_comparison.json --models zero,persistence,logistic,hgb

# Existing automated checks
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q --basetemp=.pytest_tmp/checks
```

The EXP-005 and EXP-006 folder READMEs describe exact run selection and inference.
Existing runs were relocated without changing their contents. Building/training the
same run again requires an intentional output location or overwrite choice.

### Official prediction CSV

The required header, **including column order**, is:

```csv
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
989,1,0,0,0,0,1
2016,0,0,0,0,0,1
```

These rows illustrate the format only. The final file is `output_{studentID}.csv`, with **exactly one row for each official test user**, no extra/duplicate users, and binary values in every target column. The official `test_input.csv` is scheduled for release on **October 28**.

After implementing prediction generation, verify the final CSV with the course evaluator (from the personal project root):

```powershell
python .\assignment2\evaluation\evaluate.py `
  --pred .\output_YOUR_STUDENT_ID.csv `
  --input .\assignment2\evaluation\test_input.csv
```

For a historical evaluator smoke test using the **instructor's example predictions and labels**:

```powershell
python .\assignment2\evaluation\evaluate.py `
  --pred .\assignment2\evaluation\examples\sample_output.csv `
  --truth .\assignment2\evaluation\examples\sample_truth.csv
```

A format or user-list mismatch prevents scoring and results in **zero for the 7% model-performance component**. This historical sample's score is **not** the official hidden-test score.

## 8. Presentation and submission

In addition to the code and prediction CSV, the assignment requires an **8–12 minute non-AI-generated presentation video**, an **English PDF slide deck of at most 10 pages**, and answers to:

| Question | Weight | Topic |
|---|---:|---|
| Q1 | 1% | Complete schema graph; entity graph for GitHub IDs `22996444` and `28983449` during `2026-09-15 00:00`–`2026-09-17 00:00 UTC` |
| Q2 | 2% | How relationships across tables were modeled more effectively |
| Q3 | 2% | How temporal information was modeled more effectively |
| Q4 | 3% | Framework diagram and complete pipeline |

Use the [course slide template](https://docs.google.com/presentation/d/1TcLh6RZz-5-sjDLmC5tbSEFzOt8RR9M3whmDk-Le4io/edit?usp=sharing). Material intended for grading must appear in **both** the slides and the recorded explanation. The PDF contains **inconsistent `Asg2_...` versus `Asg_...` video/slide filename instructions**; check E3 or ask the TA before uploading.

Submit the prediction file, slides and recording through E3 before the deadline. All code commits must be complete by that time. The instructor requires the GitHub repository to be **public at submission**, even if development begins in a private repository.

## 9. Keep the official data out of personal Git history

At minimum, the root `.gitignore` should contain:

```gitignore
# Instructor-provided repository and temporary copies
/assignment2/
/assignment2_backup/
/assignment2_new/

# Large generated/raw files
*.parquet
*.pt
*.pth
*.ckpt
*.joblib
*.pkl
*.mp4

# Python/Jupyter/secrets
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
.env
```

Check whether the instructor's dataset is ignored:

```powershell
git check-ignore -v assignment2/data/db/Users.parquet
git status --short
```

If files were already committed or staged, adding `.gitignore` alone does not remove them from Git's index. Review the tracked files before your first push. Never remove the instructor's `.git` directory and mistake their full repository for original work.
