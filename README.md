# GitHub Activity Prediction

NYCU Big Data Analytics Techniques and Applications, Fall 2026, Assignment 2.
Student ID: 314561002.

For each test user, the model predicts whether they will do each of six GitHub activities between November 22, 00:00 and November 29, 00:00 Taiwan time (2026-11-21T16:00Z to 2026-11-28T16:00Z). The test users are listed in `submission/test_input.csv` of the [course repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2) (4,465 users).

| Column | 1 means the user will |
|---|---|
| `open_pr` | open a pull request |
| `open_issue` | open an issue |
| `issue_comment` | comment in an issue or PR discussion |
| `pr_review` | submit a PR review |
| `pr_review_comment` | comment on a PR diff |
| `push` | push commits to a branch or tag |

The score is (macro-F1 + micro-F1) / 2 over the six columns.

## Data Source

The data comes from [GH Archive](https://www.gharchive.org/), a public archive of GitHub events. I download the hourly files from `https://data.gharchive.org/` and build the eight tables with the course's `scripts/preprocess.py`, which also filters to the course cohort and anonymizes the data. The model reads only those eight tables. The table format is described in the course [README](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/README.md) and [SCHEMA](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/SCHEMA.md).

## Model

The submitted policy is H2/P021 (`--policy h2_p021`):

1. Build 59 features per user from the 7, 30 and 90 days before the cutoff.
2. Train one histogram gradient boosting classifier per activity on 56 daily training origins from the last 63 days, with sample weights that halve every two weeks.
3. Average each classifier's rank with the rank of an exponentially decayed 7-day activity score (EW7).
4. For each activity, mark as many users positive as were active in the last 7 days (the persistence count). Ties go to the smaller `UserID`.

Time rules, with S = 2026-11-21T16:00Z:

- Features only use rows visible before C = S − 4 h (Nov 21 12:00 UTC).
- Training labels only use events before A = S − 3 h.
- PRs, issues and reviews become visible at `first_seen_at`; comments and pushes at `created_at`. Fields that are only known later (`merged_at`, `closed_at`, etc.) are not used.
- Users are matched across runs by `Users.github_user_id`.

`run.py` also has the variants `h2wpca_p021`, `h2dfs_p021` and `persist7`, which I tested but did not submit.

## Setup

Python 3.11, run from the project root. Everything runs on CPU.

```powershell
git clone https://github.com/benedictdavon/nycu-bda-2026-asg2.git
cd nycu-bda-2026-asg2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-final.txt

git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git data/course
```

On Linux/macOS use `source .venv/bin/activate`. `environment.yaml` is the larger environment I used for the notebooks.

## Download and preprocess

The submission uses every hour from 2026-05-01 00:00 to 2026-11-21 11:00 UTC (4,908 files). In Bash:

```bash
mkdir -p data/raw
for month in 05 06 07 08 09 10; do
  for day in $(seq -w 1 31); do
    wget -nc -P data/raw https://data.gharchive.org/2026-${month}-${day}-{0..23}.json.gz
  done
done
wget -nc -P data/raw https://data.gharchive.org/2026-11-{01..20}-{0..23}.json.gz
wget -nc -P data/raw https://data.gharchive.org/2026-11-21-{0..11}.json.gz
```

Dates that don't exist (for example June 31) return 404, which is fine. Then build the tables:

```powershell
python data/course/scripts/preprocess.py --raw data/raw --out data/final_20261121
```

## Train and predict

```powershell
python -m experiments.final_submission.run --mode submit --origin 2026-11-21T16:00:00Z --official-repo data/course --data-dir data/final_20261121/db --roster data/course/submission/test_input.csv --policy h2_p021 --out delivery/output_314561002.csv --output-root outputs/final_delivery

python data/course/scripts/evaluate.py --pred delivery/output_314561002.csv --input data/course/submission/test_input.csv
```

The run writes its settings, input table hashes and training origins to the `RUN_DIR` it prints.

To do everything in one step (preprocess, train twice, check that both CSVs are identical, run the format check):

```powershell
python scripts/reproduce_submission.py --official-repo data/course --raw data/raw --output-dir delivery/final_20261121 --student-id 314561002
```

The result is `delivery/final_20261121/output_314561002.csv`.

## Tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:MKL_ENABLE_INSTRUCTIONS = "SSE4_2"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python -m pytest -c pytest-final.ini -q --basetemp=.pytest_tmp/final_delivery
```

The tests cover the time cutoffs, labels, splits, ranking and ties, and preprocessing. The MKL settings work around a native library crash I hit with Conda on Windows.

## Results

In weekly backtests from Sep 27 to Nov 15, H2/P021 scored 0.047 to 0.065 combined F1 above 7-day persistence. Earlier comparisons with the PCA, DFS and SVM variants are in [reports/svm_variants_20261026/newer_20261108/SUMMARY.md](reports/svm_variants_20261026/newer_20261108/SUMMARY.md).

## Repository layout

- `experiments/final_submission/`: training and prediction entry point
- `src/`: data loading, features, labels, time splits and metrics
- `experiments/neural_research_common/`: shared population and model settings
- `experiments/pca_fisher_20261026/`, `pca_followup_20261026/`, `dfs_pca_20261026/`: the tested variants
- `scripts/reproduce_submission.py`: one-step rebuild and check
- `tests/`: tests
- `reports/`: experiment results
- `reports/presentation/`: schema graph figures and the script that draws them
- `notebooks/`: early data exploration on the first version of the dataset
