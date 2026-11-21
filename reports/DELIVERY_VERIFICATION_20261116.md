# Reproducibility rehearsal, November 16, 2026

A dry run of the full submission process on a past week, to check that the pipeline gives the same CSV every time.

- Course repository commit: `af9f335301d9badb482dfafa10c4534eabac845a`, 4,465 test users.
- Rehearsal prediction time: `2026-10-31T16:00:00Z` (November 1, 00:00 Taiwan).
- Input: 5,007 hourly GH Archive files from `2026-04-13T12:00Z` up to `2026-11-08T03:00Z`. To save disk space I kept only the course cohort's records from each hour, using the same filter as `preprocess.py`, and logged the SHA-256 of each original and filtered file. The eight tables were then built with the unchanged `preprocess.py`.

## Results

1. `pytest -c pytest-final.ini`: 117 passed (7 pandas deprecation warnings), also from a clean clone.
2. Two separate H2/P021 fits on the same tables produced byte-identical CSVs. A third fit from the clean clone matched too.
3. `evaluate.py`: "Format check: no errors (4465 users)" for every fit.
4. All 57 feature cutoffs passed the time checks, and training labels only used events before the label cutoff.

CSV SHA-256: `51f2726c86fe6ef9b6e73ba484a10ff0605de176dfb6f4a037ec2af8f9de4f8a`

The run records are in `delivery/rehearsal_20261116/`. They stay local and are not in Git, along with the data, fitted models and predictions.

To repeat it, set up the environment from the main README and run `scripts/reproduce_submission.py` with `--rehearsal --origin 2026-10-31T16:00:00Z` on the same input files.
