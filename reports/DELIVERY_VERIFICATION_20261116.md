# Delivery verification — November 16, 2026

Student: 314561002. Primary policy: frozen H2/P021 (`h2_p021`).

## Verified scope

- Revised course checkout: `af9f335301d9badb482dfafa10c4534eabac845a`.
- Official revised roster: 4,465 users.
- Historical rehearsal origin: `2026-10-31T16:00:00Z` (November 1, 00:00 Taiwan).
- Input staging interval: `2026-04-13T12:00:00Z` to `2026-11-08T03:00:00Z`, end exclusive; 5,007 hourly files.
- Input staging files retain the exact cohort/event records selected by the instructor's extraction function. They are smaller than the full public GH Archive files. The retained source receipts map original download SHA-256 values to staged SHA-256 values. The official preprocessing CLI was then run unchanged to build all eight tables.
- This is a historical rehearsal, not the November 22–29 prediction CSV and not evidence of an E3 upload.

## Checks completed

1. CPU verification suite: **117 passed**, with seven pandas deprecation warnings. Repeated from a clean source snapshot with no local experiment outputs or untracked source dependencies.
2. Official preprocessing: rebuilt all eight tables from the retained staged archive files.
3. Two independent full H2 refits on the same rebuilt tables: identical prediction CSV bytes.
4. Third H2 refit from the clean source snapshot: identical CSV bytes again.
5. Official evaluator: **Format check: no errors (4465 users)** for every fit.
6. Temporal audit: all 57 scored/training feature cutoffs passed; training labels used only observations before the decision cutoff.

Shared CSV SHA-256:

```text
51f2726c86fe6ef9b6e73ba484a10ff0605de176dfb6f4a037ec2af8f9de4f8a
```

The local receipt is `delivery/rehearsal_20261116/reproduction.json`. Original-to-staged input provenance is retained in `source_receipts.json` alongside it. Run configurations record all input table hashes, source hashes, 56 training origins, runtime versions, model settings, per-stage timing and official validator output. Generated data, models, per-user predictions and raw archives are excluded from Git.

## Reproduction commands

Use Python 3.11 with `requirements-final.txt`, then run the commands in the main README. `scripts/reproduce_submission.py` rebuilds the official tables, refits twice, validates the CSVs and requires matching hashes. Supply the exact retained raw archive files and pinned course checkout to reproduce an existing receipt.

The selected test command is:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:MKL_ENABLE_INSTRUCTIONS = "SSE4_2"
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python -m pytest -c pytest-final.ini -q --basetemp=.pytest_tmp/final_delivery
```

The broader research suite includes optional neural experiments requiring PyTorch. It is not claimed to pass in the pinned CPU-only environment. Without the documented MKL settings, the installed Windows Conda runtime raised a native library error in an earlier verification attempt; the documented settings and the inference entry point avoid that failure.

## Final delivery still tied to the deadline

The revised deadline is **November 21, 2026, 23:55 Taiwan time**. Predictions concern **November 22–29**, so they must be fixed and submitted before that activity happens; scoring follows the target week.

For the final run, stage the full needed history through `2026-11-21T12:00:00Z` exclusive (20:00 Taiwan), retain the archive-hour list and course recipe, then run:

```powershell
python scripts/reproduce_submission.py --official-repo course_assignment2 --raw raw_data --output-dir delivery/final_20261121 --student-id 314561002
```

Final mode refuses to run before its data cutoff or with post-cutoff raw hours. Its checked CSV will be `delivery/final_20261121/output_314561002.csv`. Retain the matching input archive, all manifests and fitted models for the TA reproduction check. The grade still requires the actual E3 uploads, at most 16 English slides, and the student's own 16–20 minute recorded presentation.
