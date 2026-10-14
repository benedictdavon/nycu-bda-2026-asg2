# Assignment 2 — Experiment & EDA Log

> **Path migration, October 14:** original commands and paths below are historical execution records. Current entry points and relocated artifact paths are listed in [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md). Archived scientific files are unchanged.


**Course:** NYCU · Big Data Analytics Techniques and Applications · Fall 2026  
**Assignment:** Assignment 2 — Predicting GitHub User Activity  
**Log status (updated 2026-10-14):** EXP-001 through EXP-005 completed with recorded evidence; **EXP-006 ongoing, with real training and primary historical validation completed for zero, persistence, logistic regression and histogram gradient boosting.** Best reported primary proxy-eligible pooled score: **persistence 0.454677**; full per-target diagnostics and threshold audit remain outstanding. **No untouched October holdout or official November test score.**  
**Data access:** Official instructor-supplied GitHub relational dataset only  
**Authoring date:** 2026-10-14 (Asia/Taipei)  
**Initial observed instructor snapshot:** `506d860693df60a698b68aa63a8edb4a984c8e1b`  
**Primary evidence:** Executed `01_data_audit(1).ipynb`, `02_deeper_data_analysis(1).ipynb`, and `03_label_and_split_validation(5).ipynb`; subsequent **user-supplied diagnostic outputs in this conversation** for EXP-004. All used the same instructor SHA unless explicitly marked otherwise.

> **Purpose.** This is the single persistent, auditable record for the assignment and for coding-agent handoff. The original EDA-01/02 measurements below are preserved from the supplied log. EDA-03 results are verified against saved executed-notebook outputs; EXP-004 follow-on diagnostics are supported by the exact tables printed and supplied in this conversation (not necessarily saved into the notebook file). A **non-trained previous-week heuristic** was scored on reconstructed historical labels, and **sample-ground-truth reconstruction** was verified separately. Later EXP-006 fitted logistic regression and histogram gradient boosting and compared them with two heuristics on a different historical validation protocol; see §15 for these measured results. No result here is an official November prediction score, and no untouched October historical holdout has been evaluated. Do not silently rewrite historical entries; append corrections and add dated current-state summaries.

## Reader's map

**Current state: read [§0](#0-current-status-and-interpretation-guide), [§14](#14-exp-005-reusable-temporal-dataset-generator-real-execution), and [§15](#15-exp-006-historical-baseline-modeling-first-real-results) first; §§10–11 explain the earlier label investigations.** Sections 1–13 preserve earlier dated evidence and may contain prior-status statements superseded by the new EXP-005/006 entries.

1. [Executive findings and decisions](#1-executive-findings-and-decisions)
2. [Assignment contract and definitions](#2-assignment-contract-and-definitions)
3. [Evidence provenance and experiment protocol](#3-evidence-provenance-and-experiment-protocol)
4. [EDA-01: Initial data audit](#4-eda-01-initial-data-audit)
5. [EDA-02: Deeper follow-up](#5-eda-02-deeper-follow-up)
6. [Cross-study synthesis](#6-cross-study-synthesis-and-hypotheses)
7. [Engineering decisions and experiment backlog](#7-engineering-decisions-and-experiment-backlog)
8. [Release management, open issues, and append-only template](#8-release-management-open-issues-and-append-only-template)
9. [References and traceability](#9-references-and-traceability)
10. [EDA-03: Historical labels, temporal splits and heuristic baselines](#10-eda-03-historical-labels-temporal-splits-and-rule-based-baselines)
11. [EXP-004: Official example truth reconciliation — exact agreement](#11-exp-004-official-example-label-reconstruction-and-observation-cutoff-reconciliation)
12. [Updated engineering decisions, limitations and next steps](#12-updated-decisions-open-questions-and-implementation-checklist)
13. [Reproducibility addendum and evidence ledger](#13-reproducibility-addendum-and-evidence-ledger)
14. [EXP-005: Reusable temporal dataset generator, real execution](#14-exp-005-reusable-temporal-dataset-generator-real-execution)
15. [EXP-006: Historical baseline modeling, first real results](#15-exp-006-historical-baseline-modeling-first-real-results)

---

## 0. Current status and interpretation guide

**Status last updated:** 2026-10-14 Asia/Taipei. **Official instructor data SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`. **Daily release comparison:** still only a **single distinct SHA** across the three notebook analyses; no demonstrated day-over-day data change. **Test user roster:** not yet released at the time of these experiments (official schedule: October 28).

### 0.1 Results that supersede the earlier EDA-only summary

| Study | Outcome | Exact scope / significance |
|---|---|---|
| **EDA-01** | 2,552,788 rows, 9,000 cohort users, 0 PK/FK integrity violations | One-snapshot structural/coverage audit; descriptive only. |
| **EDA-02** | Large time-varying activity regimes; prior-week persistence and repository breadth correlate with next-week activity | Correlation/EDA, **not** fitted prediction improvements or causal claims. |
| **EDA-03** | 31 chronological 7-day windows: **22 training, 5 validation, 4 sealed historical holdout**; **279,000 user-week rows** (243,000 with evaluated candidate labels, 36,000 sealed) | A cut-off-based **candidate-label** and split construction exercise; see §10. |
| **EDA-03 rule baseline** | Eligible-proxy mean weekly combined F1 **0.5462 training → 0.3504 validation** using the previous-week recurrence rule; all-zero scored 0 in the evaluated weeks | Uses instructor's `evaluation/evaluate.py` against **historically reconstructed candidate labels**. Not a fitted ML model, not prospective official test performance, and not a pooled score. |
| **EXP-004 initial comparison** | With Sept 27 **00:00 UTC** week start and event-time labels, **188** mismatched user–target decisions versus instructor `sample_truth.csv`; combined F1 **0.978894** | First comparison had the **wrong likely example-window clock boundary**, not proof that event-time labeling was incorrect. |
| **EXP-004 boundary diagnosis** | Shift +13 h: sample window Sept 27 **13:00 UTC** to Oct 4 **13:00 UTC** = Sept 27 **21:00 Taiwan** to Oct 4 **21:00 Taiwan**; push mismatch **97 → 0** | Perfect push agreement at this calibrated example-week clock boundary. |
| **EXP-004 six-target agreement before as-of filter** | **28,708 / 28,710** decisions correct; only **two `open_issue` false positives**; macro **0.999501**, micro **0.999776**, combined **0.999639** | Five targets match exactly; two issues created inside example week but only first observed Oct 9–10. |
| **EXP-004 as-of reconciliation** | Event-time labels with **Oct 9, 20:00 or 21:00 Taiwan** candidate observation cutoff: **28,710 / 28,710** labels match; macro, micro and combined F1 **all 1.000000** | **Exact in-sample reproduction of the 4,785-user official example truth**, after calibrating week boundary and cutoff to the example. This **does not prove** the TA's generation code or generalize to other weeks. |
| **EXP-005 real dataset build** | **14 passed / 1 skipped** tests; perfect official sample regression; **189,000 training, 81,000 validation, 9,000 sealed holdout feature rows** | User-executed Windows Conda/Parquet build, documented in §14; not predictive model performance. |
| **EXP-006 learned baseline comparison** | On **16,982 proxy-eligible user-weeks** (Aug 31, Sep 7, Sep 14): persistence **0.454677**, HGB **0.398767**, logistic **0.381531**, zero **0.000000** combined pooled F1 | User-executed real training; §15 documents exact leaderboard, timing, split, and remaining per-target verification work. **No independent holdout or November score**. |

### 0.2 Three quantities that must not be confused

1. **Observed behavior association:** conditional recurrence rates or repository-breadth group rates (EDA-02). No predictions were fitted or evaluated for those tables.
2. **Historical prediction performance:** score of the simple previous-week rule **versus candidate historical labels** on weeks split by time (EDA-03); validation mean combined F1 **0.3504** for the eligibility proxy. This is an actual heuristic benchmark under that configuration, but **not** the instructor's November test score.
3. **Historical label reconstruction agreement:** score of reconstructed labels **versus the supplied example truth itself** (EXP-004), reaching **1.000000** with calibrated boundaries/cutoff. This measures the reconstruction, **not predictive capability**.
4. **Historical fitted-model / heuristic validation:** EXP-006 compares persistence, logistic and HGB on a **different** three-week reporting slice with **pooled** TP/FP/FN scores and 16,982 proxy-eligible user-weeks. Best reported score **0.454677**, which is **not directly comparable** to EDA-03's mean-per-week **0.3504** because the window and aggregation protocols differ.

### 0.3 Temporal semantics established by the investigation

- **Feature time `t`:** include a record in model input only when the **official row-visibility timestamp is strictly before `t`**; exclude or correctly mask six explicitly late-known fields. A historical model cannot inspect an issue created in September if the supplied data did not observe that issue until October.
- **Event-label window `[t, t+7d)`:** look for the original action timestamp (`created_at` / `submitted_at`), not `first_seen_at`, as the primary retrospective label candidate. Filter `Issues.is_pull_request=False` for `open_issue`, and map internal actors to `Users.github_user_id`.
- **Historical label-as-of `s`:** for entity-derived targets with independent first-observation times, a released dataset can only support a label once a relevant entity becomes observable. An **explicit, configurable observation cutoff** is needed for reproducibility; do **not** universally equate the observed example's approximately five-day maturation with the official label policy.
- **Direct-event tables (`Pushes`, `IssueComments`, `PRReviewComments`):** Notebook 03 sets both `event_utc` and `seen_utc` to the same stored `created_at`. This is **not a separate ingestion/first-seen measurement**; a label-as-of filter cannot reconstruct prior snapshot completeness for these three tables.
- **Example vs official target:** the inferred Sept 27 21:00 Taiwan example window must **never** replace the explicitly official Nov 12 00:00–Nov 19 00:00 Taiwan prediction interval.

### 0.4 Confidence and restrictions

| Claim | Status |
|---|---|
| The selected rule reconstructs all six **sample** labels at Oct 9 candidate cutoffs | **Verified numerically for the supplied 4,785 examples.** |
| Original action timestamp is better than first observation for **this sample** | **Verified comparison.** |
| The course's sample truth was generated from the Oct 9 release | **Well-supported hypothesis, not directly confirmed by the TA's generation code.** |
| Exact collection watermark was 20:00 rather than 21:00 Taiwan | **Undetermined**: both produce exactly the same labels here. |
| The official later test uses the same example week clock boundary or five-day lag | **Not established; official November window is separately specified.** |
| A model can achieve F1 = 1 | **Not supported**: 1.000000 is ground-truth reconstruction, not prediction. |
| The previous-week heuristic's 0.3504 remains the definitive baseline after labeling policy changes | **Not established**: rerun after finalizing historical label/split settings. |

> **Reader instruction:** Original §§1–13 retain earlier evidence and dated assumptions. Text such as “no fitted models yet” or “EXP-005 planned” describes an earlier state and is superseded by the current-state summary and **§§14–15**. Earlier measured numbers remain valid **only for their stated experiment protocols**.

---

## 1. Executive findings and decisions

### 1.1 Confirmed observations (initial snapshot)

| Topic | Measured result | Meaning / required restraint |
|---|---|---|
| Scale | **2,552,788 rows** across 8 tables; **9,000 cohort users** / 45,590 Users rows | Reference users are also present; only cohort histories are collected comprehensively. |
| Structural integrity | **0 null PKs; 0 duplicate PKs; 0 orphaned non-null FKs** across **20** declared FK relationships | Within-snapshot PK/FK joins appear structurally sound, but observation/label correctness is separate. |
| Dominant table | Pushes: **1,661,256** rows | Event frequencies differ greatly across target types; entity table rows are not uniformly emitted activity events. |
| Regime shift | Issue-comment median weekly active users **996 → 161 → 470** (Early / Dip / Recovery); review-comment **605 → 70 → 311**; push **3,639 → 3,246 → 2,394** | Sharp and target-specific time variation. The recovery label describes the regime's timing, **not a recovery in every activity**. |
| Recent-week persistence | Median lift for next-week recurrence: **push 4.985×**, **issue_comment 14.793×**, **pr_review_comment 21.243×** | Strong observational motivation for recency features, **not a trained-model performance result**. |
| Repository breadth | Median next-week push rate by prior 30-day distinct pushed repos: **0 → 5.26%; 1 → 31.28%; 2–3 → 46.36%; 4+ → 70.65%** | Promising relational indicator, but confounded by push volume and prior activeness. |
| Label ambiguity | For cohort-attributed non-PR Issues, **10.82%** of paired rows were first seen over seven days after `created_at`; full Issues table: **20.99%** over seven days | First observation cannot safely stand in for creation event; date/actor proxies require semantic verification. |
| Historical testing population | ≥5 observation records / 90-day **proxy eligibility** fell from **95.18% (Jul 20)** to **50.36% (Oct 5)** | Final selection differs from all 9,000 users; proxy does not reproduce TA's official Oct 28 roster. |
| Late-known fields | At Oct 1 cutoff, among otherwise visible rows: future `merged_at` **8**, PR `closed_at` **8**, Issue `closed_at` **25**, review `dismissed_at` **0** | Historical features must mask/exclude future-known values even when their row exists. |
| Release comparison | Both EDA runs used the **same course SHA**; **0 row-count differences** in all 8 tables | **No day-over-day update effect measured yet**. |

### 1.2 Historical limitations after EDA-01/02, before Notebook 03 (superseded where noted)

- **No official test score**, no macro-F1 or micro-F1, and no trained predictive baseline.
- **No verified six-target historical ground truth:** `open_pr`, `open_issue`, and `pr_review` are visualized as **first-observation proxies** in some experiments, not proven original events.
- **No causal explanation** for summer activity decline or September shifts.
- **No demonstrated performance advantage** for GNN/RDL, graph features, repository breadth, or particular lookback lengths.
- **No comparison across distinct daily instructor dataset releases.** The history log has only one SHA.

### 1.3 Priorities recorded after EDA-02 (historical; see updated §12)

1. **P0: Validate target label semantics and observation delays** for all six types before assembling historical supervised datasets.
2. **P0: Construct a point-in-time-safe, documented temporal dataset builder** with cutoff-based feature masking, checks for delayed observation, and stable user identity.
3. **P1: Investigate the time-regime shift** using equal-duration cohorts / direct events; distinguish possible data-collection changes from behavior without assuming cause.
4. **P1: Train simple, strong time-respecting baselines** before adding graph architectures.
5. **P2: Add and ablate temporal and relational features** on identical historical splits.
6. **P2: Confirm actual Oct 28 test roster and behavior**, replace eligibility proxy for final analysis, track each data refresh SHA.

---

## 2. Assignment contract and definitions

### 2.1 Goal, timeline, evaluation

Predict for each official `UserID` whether each of six actions occurs on **any GitHub repository** during **2026-11-12 00:00 to 2026-11-19 00:00 Asia/Taipei**, equivalent to **2026-11-11 16:00 to 2026-11-18 16:00 UTC**. This is six simultaneous binary labels per user, not six event counts.

| Output field | Semantics | Corresponding source / attribution used by EDA |
|---|---|---|
| `open_pr` | Opens new PR | `PullRequests.author_id`; visibility proxy `first_seen_at`; possible original time `created_at` |
| `open_issue` | Opens a new *non-PR* issue | `Issues.author_id`, `is_pull_request=False`; visibility proxy `first_seen_at`; possible original `created_at` |
| `issue_comment` | Comment on issue or PR discussion | `IssueComments.author_id`, `created_at` (direct event) |
| `pr_review` | Submit PR review | `PRReviews.reviewer_id`; visibility proxy `first_seen_at`; possible original `submitted_at` |
| `pr_review_comment` | Comment on PR diff code | `PRReviewComments.author_id`, `created_at` (direct event) |
| `push` | Push to branch/tag | `Pushes.user_id`, `created_at` (direct event) |

**Critical distinction:** The target labels in the instructor's evaluation refer to *activities*. A PullRequests/Issues/PRReviews entity appearing in the snapshot does not itself establish that a fresh matching GitHub action occurred at `first_seen_at`. The direct-event rows have clearer semantics, though they still require careful cohort, time, and integrity handling.

**Official score:** `Score = (macro-F1 + micro-F1)/2`, computed over six activities. **Prediction performance = 7%** of course grade; questions and video/slides = **8%**, total assignment **15%**. Deadline **2026-11-11 23:55 Taiwan**, no late submissions. `test_input.csv` scheduled for **2026-10-28**; CSV must exactly match its GitHub user IDs and exact seven-column header. Invalid output yields zero for the performance component.

**Data restriction:** Use **only the instructor-provided eight-table data**. No extra GH Archive datasets, GitHub API queries, web scraping, or alternative event logs. It is permitted to create historical train/validation/test samples from the supplied tables.

### 2.2 Timestamp and cutoff rules

- All Parquet timestamps are stored as **timezone-naive UTC**; explicitly attach `UTC` before converting to `Asia/Taipei` for weekly comparisons.
- Feature visibility times: `first_seen_at` for `Users`, `Repositories`, `PullRequests`, `Issues`, `PRReviews`; `created_at` for `IssueComments`, `PRReviewComments`, `Pushes`.
- Historical feature sample at cutoff **t**: include *only rows with official visibility timestamp `< t`*. Candidate labels must describe the future `[t, t + 7 days)` without contaminating past features.
- Six late-known columns: `PullRequests.merged_at`, `merged_by_id`, `closed_at`; `Issues.closed_at`, `state_reason`; `PRReviews.dismissed_at`. A row's visibility does **not** mean the late-known fields were already known. Mask them or exclude them if availability cannot be justified.
- Internal row-number PKs/FKs may be **renumbered every daily refresh**. `Users.github_user_id` is the stable ID for cross-release matching and final CSV `UserID`.
- Some old PR/issue records may enter a later release because of more recent events; **completed weeks** in a current snapshot are not proof of final historical coverage.

```text
                         Snapshot available today
     historical event time ------------------------------> now
               |                    |
               |                    +-- first_seen_at: when row became observable
               +-- created_at / submitted_at: when entity/event originally occurred

   For historical feature cutoff t:
       INCLUDE a row only when its official ROW TIME < t.
       MASK any subsequently disclosed attributes at t.
       LEARN label y only from events in [t, t+7d), using verified target semantics.
       NEVER use today's full entity state as if it were known at historical t.
```

### 2.3 Schema / relational design relevance

All analyses use the 8-table relational model. `Users` contains **cohort** and **non-cohort referenced** actors. One-sided historical activity data exist only for the 9,000 cohort accounts. The 20 FK edges connect PRs, issues, comments, reviews and pushes to users/repos and other entities (including a self-FK on review comments). Within-snapshot PK-to-FK joins are valid, but relational feature *usefulness* and target-label correctness are hypotheses to test, not automatic consequences of integrity.

---

## 3. Evidence provenance and experiment protocol

### 3.1 Executed input artifacts and controlled configuration

| Evidence | Local artifact in project | Executed version reviewed | Scope |
|---|---|---|---|
| EDA-01 | `notebooks/01_data_audit.ipynb` | Uploaded executed copy `01_data_audit(1).ipynb` (32 cells, 14 code cells) | Data audit and first descriptive results |
| EDA-02 | `notebooks/02_deeper_data_analysis.ipynb` | Uploaded executed copy `02_deeper_data_analysis(1).ipynb` (28 cells, 12 code cells) | Follow-up and hypothesis diagnostics |
| Instructor | `assignment2/data/db/*.parquet` | SHA `506d860693df60a698b68aa63a8edb4a984c8e1b` | Both runs; 8 tables |
| Official schema | `assignment2/data/db/SCHEMA.md` | Repository documentation | Column/PK/FK/timestamp semantics |
| Official evaluator | `assignment2/evaluation/evaluate.py` | Repository documentation | CSV validation and eventual F1 score |

- Local environment reported **pandas 2.3.3**, **NumPy 2.4.6**, and PyArrow installed.
- Local instructor files were present at `Assignment 2/assignment2/data/db/` on a Windows workstation.
- Notebook 01 used an **audit cutoff = 2026-10-01 00:00 Asia/Taipei** and lookback windows of **7 / 30 / 90 days**.
- Notebook 02 used **45 complete Monday-to-Monday Taiwan weeks** (based on most recent recorded event timestamp), **12 recent weekly cutoffs** for selection/persistence/repository analyses, **8 recent complete weeks** for first-seen vs candidate-event-time comparisons, **90-day** selection proxy, minimum **5** observation rows, and **30-day** repository lookback.
- Notebook 02's descriptive regimes were predeclared as **Early** (`week start < 2026-06-01`), **Dip** (`2026-06-01 ≤ week start < 2026-09-01`), **Recovery** (`week start ≥ 2026-09-01`). The 2026-08-31 start belongs to *Dip*.
- No model fit or train/validation/test label set was generated in these two notebooks.

### 3.2 Numeric interpretation conventions

- `observed_rows` = number of rows for a signal in one snapshot; for entity tables these are **entity observations**, not always original action events.
- `unique cohort users` = unique internal user FK mapped to `Users.is_cohort=True` *within the current snapshot*.
- `weekly active users` = at least one qualifying row in `[Monday 00:00, following Monday 00:00)` Taiwan time.
- A **median across weeks** is not a pooled percentage. `median(lift_week)` is not necessarily `median(active_rate)/median(inactive_rate)`.
- `first_seen_at` visibility lag = `first_seen_at − created_at` (or `− submitted_at`); missing original time is excluded from paired-lag denominator and reported separately.
- **Jaccard overlap** for user sets A, B = `|A ∩ B| / |A ∪ B|`; it depends on the length of the periods used to define the sets.
- Percentages are rounded as the notebooks printed them; rounding may cause slight discrepancies if recomputed from displayed rounded values.
- The `file_MB` labels use different conversion conventions across the notebooks (decimal MB in 01 versus MiB in 02); **do not interpret their numeric differences as actual file changes**.

---

## 4. EDA-01: Initial data audit

**Experiment ID:** `EDA-01`  
**Method:** One-snapshot exploratory/structural audit  
**Primary artifact:** executed `01_data_audit(1).ipynb`; notebook sections 1–12 / code cells 1–14  
**Output state:** successful; no recorded execution exceptions  
**Snapshot:** `506d860693df60a698b68aa63a8edb4a984c8e1b`  
**Interpretation level:** descriptive, *not model validation*.

### 4.1 EDA-01.A — Ingestion, volume, file metadata

Loaded all eight Parquet files from the **local official clone**; no collection from outside the supplied dataset. Below are the recorded exact table sizes, Parquet column counts, Parquet row groups, approximate file sizes and local filesystem modification times. **`file_MB` in this notebook uses decimal MB.**

| table            | rows    | columns   | row_groups   | file_MB   | file_modified_local     |
|:-----------------|:--------|:----------|:-------------|:----------|:------------------------|
| Users            | 45590   | 6         | 1            | 1.26      | 2026-10-14 10:21:49 CST |
| Repositories     | 109081  | 7         | 1            | 4.33      | 2026-10-14 10:21:49 CST |
| PullRequests     | 301518  | 18        | 1            | 26.75     | 2026-10-14 10:21:49 CST |
| Issues           | 172007  | 20        | 1            | 66.4      | 2026-10-14 10:21:49 CST |
| IssueComments    | 106204  | 10        | 1            | 18.88     | 2026-10-14 10:21:48 CST |
| PRReviews        | 90710   | 12        | 1            | 7.37      | 2026-10-14 10:21:49 CST |
| PRReviewComments | 66422   | 17        | 1            | 21.06     | 2026-10-14 10:21:49 CST |
| Pushes           | 1661256 | 10        | 2            | 42.84     | 2026-10-14 10:21:49 CST |

**Measured:** 2,552,788 total table rows across 8 distinct tables, with **100 combined columns counted across tables**. Pushes contain approximately **65.08%** of total *rows across all tables* (1,661,256 / 2,552,788). This cross-table proportion is **not** the probability that a user pushes, and the denominator mixes distinct record meanings.

The local file modification times were **2026-10-14 around 10:21 (as printed by Windows)**; the latest event was on **Oct 13**, so this was **not** the subsequently scheduled Oct 14 21:00 release. The reported modification time is not the instructor's upload time.

**Assessment:** Eight files loaded successfully, no file missing. No claim of steady state or longitudinal daily changes from one download.

### 4.2 EDA-01.B — Primary/foreign-key integrity

Primary-key check results (complete audit):

| table            | PK         | rows    | PK_null   | PK_duplicates   |
|:-----------------|:-----------|:--------|:----------|:----------------|
| Users            | user_id    | 45590   | 0         | 0               |
| Repositories     | repo_id    | 109081  | 0         | 0               |
| PullRequests     | pr_id      | 301518  | 0         | 0               |
| Issues           | issue_id   | 172007  | 0         | 0               |
| IssueComments    | comment_id | 106204  | 0         | 0               |
| PRReviews        | review_id  | 90710   | 0         | 0               |
| PRReviewComments | comment_id | 66422   | 0         | 0               |
| Pushes           | push_id    | 1661256 | 0         | 0               |

**Interpretation:** All eight primary-key columns contained no missing values and no duplicates. The non-null FK integrity audit across **20** edges is below (nulls may be intentional). This is a within-release structural result; no cross-release identity stability is asserted.

| table            | foreign_key            | references                  | non_null   | null_pct   | orphan_references   |
|:-----------------|:-----------------------|:----------------------------|:-----------|:-----------|:--------------------|
| PullRequests     | repo_id                | Repositories.repo_id        | 301518     | 0.0        | 0                   |
| PullRequests     | author_id              | Users.user_id               | 135077     | 55.2       | 0                   |
| PullRequests     | merged_by_id           | Users.user_id               | 105179     | 65.12      | 0                   |
| Issues           | repo_id                | Repositories.repo_id        | 172007     | 0.0        | 0                   |
| Issues           | author_id              | Users.user_id               | 172007     | 0.0        | 0                   |
| Issues           | pull_request_id        | PullRequests.pr_id          | 15443      | 91.02      | 0                   |
| Issues           | assignee_id            | Users.user_id               | 39952      | 76.77      | 0                   |
| IssueComments    | issue_id               | Issues.issue_id             | 106204     | 0.0        | 0                   |
| IssueComments    | repo_id                | Repositories.repo_id        | 106204     | 0.0        | 0                   |
| IssueComments    | author_id              | Users.user_id               | 106204     | 0.0        | 0                   |
| PRReviews        | pr_id                  | PullRequests.pr_id          | 90710      | 0.0        | 0                   |
| PRReviews        | repo_id                | Repositories.repo_id        | 90710      | 0.0        | 0                   |
| PRReviews        | reviewer_id            | Users.user_id               | 90710      | 0.0        | 0                   |
| PRReviewComments | pr_id                  | PullRequests.pr_id          | 66422      | 0.0        | 0                   |
| PRReviewComments | repo_id                | Repositories.repo_id        | 66422      | 0.0        | 0                   |
| PRReviewComments | author_id              | Users.user_id               | 66422      | 0.0        | 0                   |
| PRReviewComments | pull_request_review_id | PRReviews.review_id         | 25235      | 62.01      | 0                   |
| PRReviewComments | in_reply_to_id         | PRReviewComments.comment_id | 1361       | 97.95      | 0                   |
| Pushes           | user_id                | Users.user_id               | 1661256    | 0.0        | 0                   |
| Pushes           | repo_id                | Repositories.repo_id        | 1661256    | 0.0        | 0                   |

**Observed:** The aggregate orphan FK count was **0**. This does not imply every optional FK exists, that every entity has a known author, or that all entity attributes were observable at earlier historical cutoffs.

### 4.3 EDA-01.C — User population and identity

- Users table **45,590** rows, of which **9,000** have `is_cohort=True` and **36,590** are other referenced accounts.
- **155** bot accounts across all Users rows; **0** bots in the cohort.
- **0** duplicated `github_user_id` and **0** duplicated `user_id` in this snapshot.
- Cross-tabulation: **36,435** non-cohort non-bots, **155** non-cohort bots, **9,000** cohort non-bots.
- The cohort's own activity is collected for all six types; non-cohort actors appear only indirectly through cohort-observed events. The cohort is the appropriate set for complete-history activity analysis.

### 4.4 EDA-01.D — Time coverage and freshness

All eight *official row-time* columns were non-null in this snapshot. Times are recorded as naive UTC in Parquet. Latest timestamps by source are not identical because event types occur with different frequencies.

| table            | row_time      | non_null_rows   | null_pct   | min_UTC                   | max_UTC                   | max_Taiwan                |
|:-----------------|:--------------|:----------------|:-----------|:--------------------------|:--------------------------|:--------------------------|
| Users            | first_seen_at | 45590           | 0.0        | 2025-12-01 00:01:12+00:00 | 2026-10-13 12:51:56+00:00 | 2026-10-13 20:51:56+08:00 |
| Repositories     | first_seen_at | 109081          | 0.0        | 2025-12-01 00:01:12+00:00 | 2026-10-13 12:56:36+00:00 | 2026-10-13 20:56:36+08:00 |
| PullRequests     | first_seen_at | 301518          | 0.0        | 2025-12-01 00:03:36+00:00 | 2026-10-13 12:47:07+00:00 | 2026-10-13 20:47:07+08:00 |
| Issues           | first_seen_at | 172007          | 0.0        | 2025-12-01 00:01:55+00:00 | 2026-10-13 12:51:56+00:00 | 2026-10-13 20:51:56+08:00 |
| IssueComments    | created_at    | 106204          | 0.0        | 2025-12-01 00:01:55+00:00 | 2026-10-13 12:38:07+00:00 | 2026-10-13 20:38:07+08:00 |
| PRReviews        | first_seen_at | 90710           | 0.0        | 2025-12-01 00:26:48+00:00 | 2026-10-13 12:41:43+00:00 | 2026-10-13 20:41:43+08:00 |
| PRReviewComments | created_at    | 66422           | 0.0        | 2025-12-01 00:30:42+00:00 | 2026-10-13 11:32:36+00:00 | 2026-10-13 19:32:36+08:00 |
| Pushes           | created_at    | 1661256         | 0.0        | 2025-12-01 00:01:12+00:00 | 2026-10-13 12:59:29+00:00 | 2026-10-13 20:59:29+08:00 |

**Implication:** Snapshot-level minimum and maximum timestamps can diagnose stale input and incomplete time coverage. The final observed source event was **2026-10-13 20:59:29 Taiwan** in `Pushes`; it is *not* proof that all other activity types were collected up to that second. Notebook 01's weekly chart included **46 displayed week buckets**, including a **partial final week**; Notebook 02 later restricted its regime comparison to **45 complete** weeks.

### 4.5 EDA-01.E — Observation signals and weekly activity coverage

Six analysis signals were built with cohort attribution. Only `issue_comment`, `pr_review_comment` and `push` are direct-event rows. `open_pr`, `open_issue`, and `pr_review` use entity `first_seen_at` and must carry a **proxy marker** in interpretations.

| activity          | weeks   | observed_rows   | max_weekly_unique_users   |
|:------------------|:--------|:----------------|:--------------------------|
| issue_comment     | 46      | 106204          | 1157                      |
| open_issue        | 46      | 83543           | 1076                      |
| open_pr           | 46      | 130043          | 2652                      |
| pr_review         | 46      | 89538           | 955                       |
| pr_review_comment | 46      | 66422           | 689                       |
| push              | 46      | 1661256         | 7356                      |

**Interpretation:** Push is much more frequent than all other sources; maximum weekly distinct active user participation for push **7,356**, versus review comments **689**. These maxima need not occur in the same week. The descriptive charts also revealed a mid-year activity change; the precise regime analysis was deferred to EDA-02. Do not interpret an `open_issue` proxy row as proof an issue was opened in that week.

### 4.6 EDA-01.F — Recent historical sparsity at Oct 1 cutoff

**Feature-side cutoff:** `2026-10-01 00:00 Asia/Taipei`, i.e. **2026-09-30 16:00 UTC**. For 7, 30, 90 trailing days, count how many of the 9,000 cohort users had ≥1 qualifying *observation* row. The table below also records total observation rows, median observations among active users, and maximum observations for one user. **This is trailing historical activity, not future-label prevalence.**

| activity_proxy    | window_days   | cohort_users   | positive_users   | positive_users_pct   | total_rows   | median_rows_among_active   | max_rows_for_one_user   |
|:------------------|:--------------|:---------------|:-----------------|:---------------------|:-------------|:---------------------------|:------------------------|
| open_pr           | 7             | 9000           | 478              | 5.31                 | 853          | 1.0                        | 24                      |
| open_pr           | 30            | 9000           | 2107             | 23.41                | 12517        | 2.0                        | 149                     |
| open_pr           | 90            | 9000           | 2394             | 26.6                 | 14317        | 2.0                        | 155                     |
| open_issue        | 7             | 9000           | 297              | 3.3                  | 1162         | 1.0                        | 129                     |
| open_issue        | 30            | 9000           | 1077             | 11.97                | 10259        | 2.0                        | 566                     |
| open_issue        | 90            | 9000           | 1274             | 14.16                | 12383        | 2.0                        | 613                     |
| issue_comment     | 7             | 9000           | 348              | 3.87                 | 855          | 1.0                        | 64                      |
| issue_comment     | 30            | 9000           | 1616             | 17.96                | 12095        | 2.0                        | 413                     |
| issue_comment     | 90            | 9000           | 1821             | 20.23                | 13751        | 2.0                        | 424                     |
| pr_review         | 7             | 9000           | 343              | 3.81                 | 634          | 1.0                        | 24                      |
| pr_review         | 30            | 9000           | 1267             | 14.08                | 9362         | 3.0                        | 304                     |
| pr_review         | 90            | 9000           | 1431             | 15.9                 | 10473        | 3.0                        | 314                     |
| pr_review_comment | 7             | 9000           | 233              | 2.59                 | 477          | 1.0                        | 36                      |
| pr_review_comment | 30            | 9000           | 951              | 10.57                | 8281         | 3.0                        | 498                     |
| pr_review_comment | 90            | 9000           | 1065             | 11.83                | 9074         | 3.0                        | 510                     |
| push              | 7             | 9000           | 2627             | 29.19                | 19280        | 3.0                        | 536                     |
| push              | 30            | 9000           | 3881             | 43.12                | 45628        | 4.0                        | 1000                    |
| push              | 90            | 9000           | 6221             | 69.12                | 194949       | 10.0                       | 2849                    |

**Strong observations:** In the preceding 7 days, **2,627 (29.19%)** users pushed, **233 (2.59%)** had a PR-review-comment event, and **348 (3.87%)** made an issue comment. First-observation proxies: **478 (5.31%)** for PR, **297 (3.30%)** for issue, **343 (3.81%)** for review. In the preceding 90 days, **6,221 (69.12%)** users pushed at least once.

**Skew:** One user could have as many as **536** pushes in 7 days, **1,000** in 30 days, **2,849** in 90 days (as printed); the median among active pushers was **3**, **4**, **10** respectively. Large per-user heterogeneity is important for count transformations, robust scalers and outlier-aware baselines.

**Risk:** An always-zero model may have high accuracy on rare signals but often poor F1, especially recall of positives. The denominator for the official test should not be assumed to be the full 9,000 cohort.

### 4.7 EDA-01.G — Relational repository diversity

At the Oct 1 cutoff with 30-day lookback, per-cohort-user number of distinct repos with qualifying rows:

| signal             | cohort_mean_unique_repos   | cohort_median_unique_repos   | users_touched_multiple_repos   | max_unique_repos_single_user   |
|:-------------------|:---------------------------|:-----------------------------|:-------------------------------|:-------------------------------|
| push               | 1.066                      | 0.0                          | 1903                           | 53                             |
| pr_review_observed | 0.239                      | 0.0                          | 414                            | 25                             |

The **median is zero** for both signals across the full cohort, so means reflect a smaller set of engaged users. The maximum of **53** distinct repos for pushing and **25** for reviews shows heterogeneous relational exposure. **No prospective outcome was evaluated in EDA-01**; EDA-02 subsequently tested the association with next-week push probability.

### 4.8 EDA-01.H — Missingness audit

The executed notebook audited selected structural/target-relevant fields (not every single string/text column). Below is the **full saved missingness table for those audited fields**, including zeros. `missing_pct` uses each table's row count as denominator.

| table            | column                 | missing_n   | missing_pct   | rows    |
|:-----------------|:-----------------------|:------------|:--------------|:--------|
| IssueComments    | comment_id             | 0           | 0.0           | 106204  |
| IssueComments    | issue_id               | 0           | 0.0           | 106204  |
| IssueComments    | repo_id                | 0           | 0.0           | 106204  |
| IssueComments    | author_id              | 0           | 0.0           | 106204  |
| IssueComments    | created_at             | 0           | 0.0           | 106204  |
| Issues           | pull_request_id        | 156564      | 91.02         | 172007  |
| Issues           | assignee_id            | 132055      | 76.77         | 172007  |
| Issues           | state_reason           | 117250      | 68.17         | 172007  |
| Issues           | closed_at              | 103198      | 60.0          | 172007  |
| Issues           | issue_id               | 0           | 0.0           | 172007  |
| Issues           | repo_id                | 0           | 0.0           | 172007  |
| Issues           | author_id              | 0           | 0.0           | 172007  |
| Issues           | is_pull_request        | 0           | 0.0           | 172007  |
| Issues           | created_at             | 0           | 0.0           | 172007  |
| Issues           | first_seen_at          | 0           | 0.0           | 172007  |
| PRReviewComments | in_reply_to_id         | 65061       | 97.95         | 66422   |
| PRReviewComments | pull_request_review_id | 41187       | 62.01         | 66422   |
| PRReviewComments | comment_id             | 0           | 0.0           | 66422   |
| PRReviewComments | pr_id                  | 0           | 0.0           | 66422   |
| PRReviewComments | repo_id                | 0           | 0.0           | 66422   |
| PRReviewComments | author_id              | 0           | 0.0           | 66422   |
| PRReviewComments | created_at             | 0           | 0.0           | 66422   |
| PRReviews        | dismissed_at           | 89350       | 98.5          | 90710   |
| PRReviews        | review_id              | 0           | 0.0           | 90710   |
| PRReviews        | pr_id                  | 0           | 0.0           | 90710   |
| PRReviews        | repo_id                | 0           | 0.0           | 90710   |
| PRReviews        | reviewer_id            | 0           | 0.0           | 90710   |
| PRReviews        | submitted_at           | 0           | 0.0           | 90710   |
| PRReviews        | first_seen_at          | 0           | 0.0           | 90710   |
| PullRequests     | merged_by_id           | 196339      | 65.12         | 301518  |
| PullRequests     | merged_at              | 196339      | 65.12         | 301518  |
| PullRequests     | closed_at              | 189148      | 62.73         | 301518  |
| PullRequests     | author_id              | 166441      | 55.2          | 301518  |
| PullRequests     | created_at             | 166441      | 55.2          | 301518  |
| PullRequests     | pr_id                  | 0           | 0.0           | 301518  |
| PullRequests     | repo_id                | 0           | 0.0           | 301518  |
| PullRequests     | first_seen_at          | 0           | 0.0           | 301518  |
| Pushes           | push_id                | 0           | 0.0           | 1661256 |
| Pushes           | user_id                | 0           | 0.0           | 1661256 |
| Pushes           | repo_id                | 0           | 0.0           | 1661256 |
| Pushes           | created_at             | 0           | 0.0           | 1661256 |
| Pushes           | is_new_ref             | 0           | 0.0           | 1661256 |
| Pushes           | ref_type               | 0           | 0.0           | 1661256 |
| Repositories     | org_id                 | 77379       | 70.94         | 109081  |
| Repositories     | repo_id                | 0           | 0.0           | 109081  |
| Repositories     | github_repo_id         | 0           | 0.0           | 109081  |
| Repositories     | first_seen_at          | 0           | 0.0           | 109081  |
| Users            | user_id                | 0           | 0.0           | 45590   |
| Users            | github_user_id         | 0           | 0.0           | 45590   |
| Users            | is_cohort              | 0           | 0.0           | 45590   |
| Users            | is_bot                 | 0           | 0.0           | 45590   |
| Users            | first_seen_at          | 0           | 0.0           | 45590   |

**Important nonzero missingness:**

- `PullRequests.author_id` and `PullRequests.created_at`: **166,441/301,518 = 55.20%** each. The identity and creation time for many PR rows are unavailable; missingness cannot be fixed by blindly imputing user IDs.
- `PullRequests.merged_at` / `merged_by_id`: **65.12%** missing; `closed_at` **62.73%** missing. Values also have temporal availability restrictions.
- `Issues.pull_request_id`: **91.02%** missing because many Issues are genuine issues and/or no matching PR row exists; not automatically an error.
- `Issues.assignee_id`: **76.77%** missing; `state_reason` **68.17%**; `closed_at` **60.00%**.
- `PRReviews.dismissed_at`: **98.50%** missing, often because dismissal is rare, not because of corrupt data.
- `PRReviewComments.in_reply_to_id`: **97.95%** missing; `pull_request_review_id`: **62.01%** missing. These are optional relations.
- `Repositories.org_id`: **70.94%** missing; many repositories are not owned by an organization.

**Conclusion:** Treat optional structural nulls, unresolved entity attribution, and values known only later as **different categories**. Missingness-by-column alone does not define leakage safety.

### 4.9 EDA-01.I — Candidate creation/submission time versus first observation

The notebook compared `first_seen_at` to `created_at` for PRs/issues and `submitted_at` for reviews, **all relevant rows** (PR including missing event-time rows; Issues includes both issues and PR-conversation rows here). Lag percentages use only paired non-null timestamps.

| table        | candidate_event_time   | records   | paired_timestamps   | event_time_missing_pct   | first_seen_later_gt_24h_pct   | first_seen_later_gt_7d_pct   | negative_lag_pct   | median_lag_hours   |
|:-------------|:-----------------------|:----------|:--------------------|:-------------------------|:------------------------------|:-----------------------------|:-------------------|:-------------------|
| PullRequests | created_at             | 301518    | 135077              | 55.2                     | 4.66                          | 2.27                         | 0.0                | 0.0                |
| Issues       | created_at             | 172007    | 172007              | 0.0                      | 34.71                         | 20.99                        | 0.0                | 2.12               |
| PRReviews    | submitted_at           | 90710     | 90710               | 0.0                      | 0.45                          | 0.13                         | 0.0                | 0.0                |

**Important denominator distinction:** The **20.99%** Issues >7-day lag above applies to **all Issues rows**, including PR conversations, whereas EDA-02 separately analyzes **non-PR issues** and the **cohort-author subset**. Do not mix these numbers or interpret them as a percentage of incorrect official labels.

**Risk:** `first_seen_at` is an observation time. Old records can surface late, including issues first discovered months or years after opening. Direct activity time and visibility time are different dimensions; the same entity row must be filtered by visibility when forming historical *features*.

### 4.10 EDA-01.J — Point-in-time cutoff / late-known field audit

For `2026-10-01 00:00 Asia/Taipei`, the notebook counted rows eligible by the official row-visibility timestamp (and rows excluded because they first appeared later):

| table            | total_rows   | rows_visible_before_cutoff   | rows_not_yet_visible   | missing_time   |
|:-----------------|:-------------|:-----------------------------|:-----------------------|:---------------|
| Users            | 45590        | 45221                        | 369                    | 0              |
| Repositories     | 109081       | 107216                       | 1865                   | 0              |
| PullRequests     | 301518       | 296037                       | 5481                   | 0              |
| Issues           | 172007       | 168481                       | 3526                   | 0              |
| IssueComments    | 106204       | 104581                       | 1623                   | 0              |
| PRReviews        | 90710        | 89354                        | 1356                   | 0              |
| PRReviewComments | 66422        | 65512                        | 910                    | 0              |
| Pushes           | 1661256      | 1633385                      | 27871                  | 0              |

The following table shows **already-visible** rows with late-known fields filled in today's snapshot. `would_leak_if_used_as_is` counts those whose timestamped outcome occurred *after* historical cutoff t. This particular audit only tests timestamped late-known fields, not `merged_by_id`/`state_reason` whose temporal availability requires associated event logic.

| table        | late_known_field   | rows_visible_at_cutoff   | currently_filled_in_visible_rows   | would_leak_if_used_as_is   |
|:-------------|:-------------------|:-------------------------|:-----------------------------------|:---------------------------|
| PullRequests | merged_at          | 296037                   | 103550                             | 8                          |
| PullRequests | closed_at          | 296037                   | 110690                             | 8                          |
| Issues       | closed_at          | 168481                   | 67647                              | 25                         |
| PRReviews    | dismissed_at       | 89354                    | 1339                               | 0                          |

**Conclusion:** If historical features copy the current `merged_at`, `closed_at`, etc. without cutoff masking, future information leaks even after the table's row-time filter is applied. Conservative initial models should exclude all six late-known fields, then add only with verified time-aware conditions.

### 4.11 EDA-01.K — Official test list and baseline snapshot history

- `assignment2/evaluation/test_input.csv` was **absent**, expected because the published assignment scheduled its release on **Oct 28**.
- Notebook 01 created the **first** `experiments/eda_snapshot_history.jsonl` entry, holding release fingerprint/commit and table row counts; there was **no prior local snapshot** available for day-over-day drift.
- The instructor snapshot was not modified by the notebook. A future run after **21:00 Taiwan** requires a newly cloned official repository because instructor releases replace Git history.

### 4.12 EDA-01 decisions

**Accepted:** Within-snapshot FK joins are structurally feasible; user IDs must be joined internally within a release then exported as stable `github_user_id`; recent activity and repository breadth are reasonable candidate features.

**Not accepted / pending:** Original event labels based on entity `first_seen_at`; choosing a temporal split; making causal claims about the mid-year dip; choosing a final GNN. These questions motivated EDA-02.

---

## 5. EDA-02: Deeper follow-up

**Experiment ID:** `EDA-02`  
**Method:** Extended one-snapshot observation analyses; no trained model  
**Primary artifact:** executed `02_deeper_data_analysis(1).ipynb`; sections 1–12 / code cells 1–12  
**Output state:** successful; no recorded execution exceptions  
**Snapshot SHA:** **same** as EDA-01, `506d860693df60a698b68aa63a8edb4a984c8e1b`  
**Available event coverage:** through **2026-10-13 20:59:29 Taiwan** (maximum among eight tables).  
**Comparison scope:** **45** complete Taiwan Monday-to-Monday weeks through week starting **2026-10-05**; 12 recent cutoffs **2026-07-20 through 2026-10-05**; 8 most recent complete weeks **2026-08-17 through 2026-10-05**.

### 5.1 EDA-02.A — Reproducibility / current-release recheck

- Same **2,552,788** total records and **9,000 cohort users**.
- **36,590** additional referenced user rows, **0 cohort bots**.
- `open_pr` entity-observation signal has **130,043** cohort-attributed rows and **7,456** unique cohort users over the dataset; `open_issue` proxy **83,543 / 4,059**; `pr_review` proxy **89,538 / 3,483**.
- Direct-event signals: `issue_comment` **106,204 / 4,712** unique cohort users; `pr_review_comment` **66,422 / 2,726**; `push` **1,661,256 / 8,932**.
- **Difference vs EDA-01: zero** in row counts and instructor SHA. EDA-02 excludes partial calendar weeks from time-regime summaries; this explains **45** vs initial chart's **46** week buckets.

Raw observation-index count table:

| signal            | rows_attributed_to_cohort   | unique_cohort_users   |
|:------------------|:----------------------------|:----------------------|
| open_pr*          | 130043                      | 7456                  |
| open_issue*       | 83543                       | 4059                  |
| issue_comment     | 106204                      | 4712                  |
| pr_review*        | 89538                       | 3483                  |
| pr_review_comment | 66422                       | 2726                  |
| push              | 1661256                     | 8932                  |

**Caution:** The number of unique people seen anywhere over ~10 months is not the fraction active in a particular past or future week. Also, a PR entity first observed is not identical to a new-PR action.

### 5.2 EDA-02.B — Activity regimes: participation and per-active intensity

**Goal:** Determine whether the mid-year decline arises from fewer participants, fewer events per participant, or both. Windows are grouped by **start of completed Taiwan Monday weeks**, zeros included when reindexing weeks, but `rows_per_active_user` is undefined in a zero-user week.

**Periods:** 26 completed Early weeks (Dec–May), 14 Dip weeks (Jun–Aug), 5 Recovery-labeled weeks (Sep–Oct); note duration asymmetry and late summer week classification. Every number below is a **median over the weeks in that period**, not a sum and not the median user's event count.

| signal            | regime          | weeks   | median_weekly_users   | median_weekly_rows   | median_rows_per_active   |
|:------------------|:----------------|:--------|:----------------------|:---------------------|:-------------------------|
| issue_comment     | Dip (Jun–Aug)   | 14      | 161.0                 | 300.5                | 1.64                     |
| issue_comment     | Early (Dec–May) | 26      | 996.0                 | 3378.5               | 3.76                     |
| issue_comment     | Recovery (Sep+) | 5       | 470.0                 | 1392.0               | 2.96                     |
| open_issue        | Dip (Jun–Aug)   | 14      | 129.0                 | 338.5                | 2.65                     |
| open_issue        | Early (Dec–May) | 26      | 605.0                 | 2493.0               | 4.11                     |
| open_issue        | Recovery (Sep+) | 5       | 386.0                 | 1815.0               | 4.7                      |
| open_pr           | Dip (Jun–Aug)   | 14      | 193.5                 | 295.5                | 1.46                     |
| open_pr           | Early (Dec–May) | 26      | 1287.5                | 4258.5               | 3.22                     |
| open_pr           | Recovery (Sep+) | 5       | 635.0                 | 1341.0               | 2.11                     |
| pr_review         | Dip (Jun–Aug)   | 14      | 142.5                 | 219.0                | 1.39                     |
| pr_review         | Early (Dec–May) | 26      | 866.5                 | 2961.5               | 3.65                     |
| pr_review         | Recovery (Sep+) | 5       | 439.0                 | 1041.0               | 2.37                     |
| pr_review_comment | Dip (Jun–Aug)   | 14      | 70.0                  | 94.5                 | 1.36                     |
| pr_review_comment | Early (Dec–May) | 26      | 605.0                 | 2274.0               | 3.79                     |
| pr_review_comment | Recovery (Sep+) | 5       | 311.0                 | 836.0                | 2.69                     |
| push              | Dip (Jun–Aug)   | 14      | 3246.0                | 23905.0              | 7.32                     |
| push              | Early (Dec–May) | 26      | 3639.0                | 44842.5              | 11.66                    |
| push              | Recovery (Sep+) | 5       | 2394.0                | 13280.0              | 5.55                     |

**Primary direct-event interpretations:**

| Signal | Early weekly active median | Dip | Recovery-labeled | Dip / Early | Early median rows/active | Dip rows/active | Recovery rows/active |
|---|---:|---:|---:|---:|---:|---:|---:|
| Push | 3,639 | 3,246 | 2,394 | **0.89×** | 11.66 | 7.32 | 5.55 |
| Issue comments | 996 | 161 | 470 | **0.16×** | 3.76 | 1.64 | 2.96 |
| PR review comments | 605 | 70 | 311 | **0.12×** | 3.79 | 1.36 | 2.69 |

- **Both extensive and intensive margins changed**: users per week fell, and records per active user generally fell.
- Issue-comment participation partially recovered after Aug but remained below the Early median (**470 vs 996**).
- PR-review-comment participation likewise partially recovered (**311 vs 605**).
- **Push did not recover** in the final five weeks by these medians (**2,394** vs Dip **3,246**, Early **3,639**).
- First-seen PR/issue/review proxies show analogous broad patterns but are not verified emitted-event rates.
- A possibility is **changing real behavior**, a **collection/ETL pipeline artifact**, or **both**. The snapshot alone cannot choose among these explanations.

**Potential experiment:** Equal-length sliding windows and a direct-event coverage audit, including cohort-specific denominators and per-user rates, before interpreting this as seasonality.

### 5.3 EDA-02.C — Active-user turnover between periods

For each direct-event signal, EDA-02 forms a set of cohort users with at least one recorded event in each entire period, then calculates Jaccard similarity plus entry/exit counts. These are *whole-period sets*, not independent weekly samples.

| signal            | pair           | earlier_active   | later_active   | both   | jaccard   | new_in_later   | missing_from_later   |
|:------------------|:---------------|:-----------------|:---------------|:-------|:----------|:---------------|:---------------------|
| push              | Early → Dip    | 8914             | 7438           | 7421   | 0.831     | 17             | 1493                 |
| push              | Dip → Recovery | 7438             | 4364           | 4217   | 0.556     | 147            | 3221                 |
| issue_comment     | Early → Dip    | 4412             | 1768           | 1560   | 0.338     | 208            | 2852                 |
| issue_comment     | Dip → Recovery | 1768             | 1509           | 973    | 0.422     | 536            | 795                  |
| pr_review_comment | Early → Dip    | 2520             | 908            | 781    | 0.295     | 127            | 1739                 |
| pr_review_comment | Dip → Recovery | 908              | 927            | 536    | 0.413     | 391            | 372                  |

**Interpretation:** Push early-to-dip sets overlap strongly (Jaccard **0.831**), while the commenting sets overlap less (**0.338** issue comments; **0.295** review comments). Dip-to-Recovery overlaps are **0.556**, **0.422**, **0.413** respectively.

- Push Early→Dip: **7,421** users in both periods; **17** appear in Dip but not Early; **1,493** Early users absent from Dip.
- Issue comments Early→Dip: **1,560** in both, **208** Dip-only, **2,852** Early-only.
- Review comments Early→Dip: **781** in both, **127** Dip-only, **1,739** Early-only.
- These describe changing *membership of active sets* and cannot disentangle user churn from unequal observation durations. Early spans 26 weeks, Dip 14, Recovery 5. Recompute on equal-length intervals for a fairer churn analysis.

### 5.4 EDA-02.D — Original event timestamps versus first discovery, all rows and cohort actors

EDA-02 improves EDA-01 by excluding PR-conversation issues from the `Issues` comparison, separating **all non-PR issue rows** from **cohort-author rows**, and displaying percentile lag magnitudes. Here, **delay is `first_seen_at - candidate_event_time`**; `candidate_event_time` means `PullRequests.created_at`, `Issues.created_at`, or `PRReviews.submitted_at`.

- **All PullRequests** retain rows with missing original timestamps and IDs for the missingness percentages, but timestamp-lag percentages are computed *only for paired timestamps*.
- **Non-PR Issues** are selected with `is_pull_request=False`; hence the all-rows count below is **125,508**, not the 172,007 Issues rows in EDA-01's broader check.
- **Cohort-actor** excludes non-cohort or unattributed actors, which substantially changes the denominator and lag distribution.

| table        | scope        | rows   | actor_null_pct   | event_time_missing_pct   | median_delay_days   | p90_delay_days   | p99_delay_days   | delay_gt_1d_pct_of_paired   | delay_gt_7d_pct_of_paired   |
|:-------------|:-------------|:-------|:-----------------|:-------------------------|:--------------------|:-----------------|:-----------------|:----------------------------|:----------------------------|
| PullRequests | all_rows     | 301518 | 55.2             | 55.2                     | 0.0                 | 0.0              | 25.32            | 4.66                        | 2.27                        |
| PullRequests | cohort_actor | 130043 | 0.0              | 0.0                      | 0.0                 | 0.0              | 8.81             | 2.64                        | 1.16                        |
| Issues       | all_rows     | 125508 | 0.0              | 0.0                      | 0.03                | 80.15            | 1870.86          | 33.27                       | 21.63                       |
| Issues       | cohort_actor | 83543  | 0.0              | 0.0                      | 0.0                 | 8.58             | 618.12           | 18.48                       | 10.82                       |
| PRReviews    | all_rows     | 90710  | 0.0              | 0.0                      | 0.0                 | 0.0              | 0.09             | 0.45                        | 0.13                        |
| PRReviews    | cohort_actor | 89538  | 0.0              | 0.0                      | 0.0                 | 0.0              | 0.0              | 0.13                        | 0.03                        |

**Key comparisons:**

- PullRequests: **55.20%** actor and creation timestamp missing across all **301,518** PR rows; among **130,043** cohort-author rows, these fields are **0% missing by construction** (selection uses known author). **1.16%** of paired cohort PR rows were first discovered over 7 days late, versus **2.27%** among paired PRs overall.
- Non-PR Issues: **125,508** total, **83,543** cohort-author rows. Cohort >7-day lag **10.82%**, compared with **21.63%** of all non-PR issue rows. Non-PR Issues `p99_delay_days` **1,870.86** all rows, **618.12** cohort rows: an extreme long tail. These are measured on candidate event timestamps with later first discovery, not arbitrary random sampling.
- PRReviews: **90,710** total, **89,538** cohort-reviewer rows; cohort >7-day lag **0.03%**, all rows **0.13%**. Candidate submission dates and first observation generally align much more closely here, but not perfectly.

**Distribution of lag bins (% of paired *cohort actor* rows):**

| table / lag_bin   | Issues   | PRReviews   | PullRequests   |
|:------------------|:---------|:------------|:---------------|
| 0–1h              | 65.14    | 84.98       | 3.23           |
| 1h–1d             | 10.71    | 0.32        | 2.14           |
| 1–7d              | 7.66     | 0.1         | 1.47           |
| 7–30d             | 4.69     | 0.02        | 0.76           |
| >30d              | 6.12     | 0.01        | 0.41           |
| ≤0d               | 5.67     | 14.57       | 92.0           |

Interpretation note: The category `≤0d` includes exactly zero lag; it does **not** indicate that a large percentage were observed before creation. EDA-01 separately found **0% negative lag** for paired timestamps. Very large `≤0d` fractions for PR indicate simultaneous event/discovery observations.

### 5.5 EDA-02.E — Per-week proxy-versus-candidate event-time disagreement

**Question:** For each user and each of the **8 completed Taiwan calendar weeks starting Aug 17–Oct 5**, do first-seen and original candidate event timestamps put the user in the *same* positive set?

For each table and week, two distinct sets are formed:

- **Proxy set:** cohort actors of rows with `first_seen_at ∈ [W, W+7d)`.
- **Candidate set:** cohort actors of rows with `created_at` / `submitted_at ∈ [W, W+7d)`.
- `proxy_only = proxy \ candidate`; `event_only = candidate \ proxy`; `intersection` is overlap; `jaccard = intersection / union`.

One person can occur in many weeks, so pooled counts are **user–week memberships**, not unique users over all 8 weeks. **Neither set has been independently verified as the instructor's official emitted-event ground truth.**

**Full per-week results (all 24 comparisons):**

| source       | week_Taiwan   | proxy_users   | candidate_users   | intersection   | proxy_only   | event_only   | jaccard   |
|:-------------|:--------------|:--------------|:------------------|:---------------|:-------------|:-------------|:----------|
| PullRequests | 2026-08-17    | 154           | 161               | 154            | 0            | 7            | 0.957     |
| PullRequests | 2026-08-24    | 146           | 165               | 146            | 0            | 19           | 0.885     |
| PullRequests | 2026-08-31    | 1055          | 1051              | 1039           | 16           | 12           | 0.974     |
| PullRequests | 2026-09-07    | 1213          | 1201              | 1193           | 20           | 8            | 0.977     |
| PullRequests | 2026-09-14    | 907           | 887               | 884            | 23           | 3            | 0.971     |
| PullRequests | 2026-09-21    | 457           | 452               | 448            | 9            | 4            | 0.972     |
| PullRequests | 2026-09-28    | 635           | 628               | 628            | 7            | 0            | 0.989     |
| PullRequests | 2026-10-05    | 448           | 440               | 440            | 8            | 0            | 0.982     |
| Issues       | 2026-08-17    | 111           | 158               | 96             | 15           | 62           | 0.555     |
| Issues       | 2026-08-24    | 116           | 167               | 102            | 14           | 65           | 0.564     |
| Issues       | 2026-08-31    | 489           | 457               | 437            | 52           | 20           | 0.859     |
| Issues       | 2026-09-07    | 572           | 509               | 498            | 74           | 11           | 0.854     |
| Issues       | 2026-09-14    | 443           | 409               | 395            | 48           | 14           | 0.864     |
| Issues       | 2026-09-21    | 277           | 270               | 251            | 26           | 19           | 0.848     |
| Issues       | 2026-09-28    | 386           | 354               | 343            | 43           | 11           | 0.864     |
| Issues       | 2026-10-05    | 280           | 250               | 249            | 31           | 1            | 0.886     |
| PRReviews    | 2026-08-17    | 113           | 113               | 113            | 0            | 0            | 1.0       |
| PRReviews    | 2026-08-24    | 108           | 108               | 108            | 0            | 0            | 1.0       |
| PRReviews    | 2026-08-31    | 576           | 576               | 576            | 0            | 0            | 1.0       |
| PRReviews    | 2026-09-07    | 757           | 757               | 756            | 1            | 1            | 0.997     |
| PRReviews    | 2026-09-14    | 641           | 640               | 640            | 1            | 0            | 0.998     |
| PRReviews    | 2026-09-21    | 316           | 317               | 316            | 0            | 1            | 0.997     |
| PRReviews    | 2026-09-28    | 439           | 438               | 438            | 1            | 0            | 0.998     |
| PRReviews    | 2026-10-05    | 298           | 298               | 298            | 0            | 0            | 1.0       |

**Eight-week pooled totals:**

| source       | proxy_users   | candidate_users   | intersection   | proxy_only   | event_only   |
|:-------------|:--------------|:------------------|:---------------|:-------------|:-------------|
| Issues       | 2674          | 2574              | 2371           | 303          | 203          |
| PRReviews    | 3248          | 3247              | 3245           | 3            | 2            |
| PullRequests | 5015          | 4985              | 4932           | 83           | 53           |

- **PullRequests:** **83 proxy-only** + **53 candidate-only** = **136 discordant user-week memberships**. **4,932** intersection memberships.
- **Non-PR Issues:** **303 proxy-only** + **203 candidate-only** = **506 discordant user-week memberships**. **2,371** intersection memberships. Relative to the other two types, this is a substantial disagreement to investigate before supervised label construction.
- **PRReviews:** **3 proxy-only** + **2 candidate-only** = **5 discordant user-week memberships**. **3,245** intersection memberships.
- `proxy_only` and `event_only` identify *timing-definition disagreements*, **not false positives and false negatives versus verified original GH Archive event labels**.

**Retroactive discovery risk.** EDA-02 also counted cohort rows whose candidate original event time lies in a week but whose row was **first seen only at or after that week's end**. Such rows would not yet have been present in a snapshot taken during the target week. The full results follow.

| table        | week_Taiwan   | cohort_rows_with_candidate_time_in_week   | not_first_seen_until_after_week   |
|:-------------|:--------------|:------------------------------------------|:----------------------------------|
| PullRequests | 2026-08-17    | 186                                       | 8                                 |
| PullRequests | 2026-08-24    | 189                                       | 27                                |
| PullRequests | 2026-08-31    | 4377                                      | 59                                |
| PullRequests | 2026-09-07    | 4462                                      | 35                                |
| PullRequests | 2026-09-14    | 2396                                      | 10                                |
| PullRequests | 2026-09-21    | 779                                       | 6                                 |
| PullRequests | 2026-09-28    | 1328                                      | 4                                 |
| PullRequests | 2026-10-05    | 802                                       | 1                                 |
| Issues       | 2026-08-17    | 336                                       | 171                               |
| Issues       | 2026-08-24    | 451                                       | 259                               |
| Issues       | 2026-08-31    | 2668                                      | 239                               |
| Issues       | 2026-09-07    | 3063                                      | 130                               |
| Issues       | 2026-09-14    | 2131                                      | 103                               |
| Issues       | 2026-09-21    | 1041                                      | 147                               |
| Issues       | 2026-09-28    | 1549                                      | 61                                |
| Issues       | 2026-10-05    | 830                                       | 24                                |
| PRReviews    | 2026-08-17    | 141                                       | 0                                 |
| PRReviews    | 2026-08-24    | 138                                       | 2                                 |
| PRReviews    | 2026-08-31    | 2910                                      | 2                                 |
| PRReviews    | 2026-09-07    | 3397                                      | 2                                 |
| PRReviews    | 2026-09-14    | 2125                                      | 0                                 |
| PRReviews    | 2026-09-21    | 576                                       | 1                                 |
| PRReviews    | 2026-09-28    | 1040                                      | 0                                 |
| PRReviews    | 2026-10-05    | 568                                       | 0                                 |

Notable examples: non-PR Issues in week **2026-08-17** had **171 of 336** candidate-week rows only discovered after the week; week **2026-08-24** had **259 of 451**. Later weeks show lower but nonzero counts. This makes historical labels from today's reconstructed entity tables potentially different from what an earlier real-time release showed. The notebook does **not** resolve exactly how the official November future labels are generated.

### 5.6 EDA-02.F — Approximate historical test-population eligibility

The instructor's evaluation README describes an Oct 28 test list from cohort users having at least **5 activity records in the preceding 90 days** as of that official release. EDA-02 **approximates** this by counting six activity-related observation tables over 90 days before each of 12 cutoffs. This mixes direct-event rows and first-seen proxies, so it is **not the exact official eligibility construction**. All cutoff times are Monday 00:00 Taiwan.

**Full 12-week population trajectory:**

| week_Taiwan   | all_cohort   | eligible_proxy_users   | eligible_pct   | record_count_p50   | record_count_p90   |
|:--------------|:-------------|:-----------------------|:---------------|:-------------------|:-------------------|
| 2026-07-20    | 9000         | 8566                   | 95.18          | 39.0               | 202.0              |
| 2026-07-27    | 9000         | 8532                   | 94.8           | 36.0               | 191.0              |
| 2026-08-03    | 9000         | 8489                   | 94.32          | 34.0               | 178.0              |
| 2026-08-10    | 9000         | 8390                   | 93.22          | 30.0               | 165.0              |
| 2026-08-17    | 9000         | 7803                   | 86.7           | 25.0               | 149.0              |
| 2026-08-24    | 9000         | 6792                   | 75.47          | 18.0               | 126.0              |
| 2026-08-31    | 9000         | 6030                   | 67.0           | 13.0               | 108.0              |
| 2026-09-07    | 9000         | 5656                   | 62.84          | 11.0               | 95.0               |
| 2026-09-14    | 9000         | 5296                   | 58.84          | 9.0                | 84.0               |
| 2026-09-21    | 9000         | 5056                   | 56.18          | 7.0                | 76.0               |
| 2026-09-28    | 9000         | 4782                   | 53.13          | 6.0                | 67.0               |
| 2026-10-05    | 9000         | 4532                   | 50.36          | 5.0                | 60.0               |

**Measured:** Proxy eligibility decreases from **8,566 / 9,000 = 95.18%** (Jul 20 cutoff) to **4,532 / 9,000 = 50.36%** (Oct 5 cutoff). Median eligible share over 12 weekly cutoffs **71.23%**. Median cohort 90-day *observation record count* declines from **39** to **5**; 90th percentile declines **202** to **60**.

**Interpretation:** The rolling-window count may decline even with partial recovery in a few weekly activities because formerly busy weeks fall out of the 90-day lookback. This is not proof of official test roster size, nor of a causal decline in real GitHub usage.

**Actual next-week direct-event occurrence, segmented by proxy eligibility:** This table is the median across 12 weeks of each week's positive percentage, not one pooled test result.

| signal            | segment        | weeks   | median_positive_pct   | mean_users   |
|:------------------|:---------------|:--------|:----------------------|:-------------|
| issue_comment     | all cohort     | 12      | 2.75                  | 9000.0       |
| issue_comment     | eligible proxy | 12      | 4.35                  | 6660.33      |
| issue_comment     | other cohort   | 12      | 0.23                  | 2339.67      |
| pr_review_comment | all cohort     | 12      | 1.44                  | 9000.0       |
| pr_review_comment | eligible proxy | 12      | 2.22                  | 6660.33      |
| pr_review_comment | other cohort   | 12      | 0.04                  | 2339.67      |
| push              | all cohort     | 12      | 26.24                 | 9000.0       |
| push              | eligible proxy | 12      | 32.53                 | 6660.33      |
| push              | other cohort   | 12      | 2.94                  | 2339.67      |

- **Push:** eligible proxy **32.53%** vs other cohort **2.94%** median future-week positive rates; all-cohort **26.24%**.
- **Issue comments:** eligible **4.35%** vs other **0.23%**; all **2.75%**.
- **PR review comments:** eligible **2.22%** vs other **0.04%**; all **1.44%**.

**Implication:** All-cohort validation scores/prevalence may fail to approximate final test difficulty. Keep both all-cohort and historically eligibility-restricted slices, clearly labeled; use real official test users when available. The observed association is descriptive and can partly reflect eligibility directly encoding prior behavior.

### 5.7 EDA-02.G — Week-to-week behavioral persistence (direct events)

Use the **12 most recent fully completed Monday-week cutoffs**, with each prior week strictly before its respective future week. For each signal and cutoff, form prior-week active set and next-week active set over the same fixed 9,000 cohort users. Then calculate:

```text
P(next event | prior event)     = count(prior_active AND next_active) / count(prior_active)
P(next event | no prior event)  = count(prior_inactive AND next_active) / count(prior_inactive)
Lift per week                  = P(next | prior) / P(next | no prior)
```

This is measured only for **push, issue_comment, pr_review_comment**, where direct event-row timestamps are available. Medians/minima/maxima are calculated **over 12 week-level estimates**, not by pooling users across weeks.

**Full reported median/min/max summary:**

| signal            | p_future_if_prior_active / median   | p_future_if_prior_active / min   | p_future_if_prior_active / max   | p_future_if_prior_inactive / median   | p_future_if_prior_inactive / min   | p_future_if_prior_inactive / max   | lift / median   | lift / min   | lift / max   |
|:------------------|:------------------------------------|:---------------------------------|:---------------------------------|:--------------------------------------|:-----------------------------------|:-----------------------------------|:----------------|:-------------|:-------------|
| issue_comment     | 0.277                               | 0.139                            | 0.648                            | 0.019                                 | 0.006                              | 0.083                              | 14.793          | 7.816        | 36.277       |
| pr_review_comment | 0.244                               | 0.08                             | 0.571                            | 0.01                                  | 0.004                              | 0.04                               | 21.243          | 14.386       | 31.645       |
| push              | 0.65                                | 0.256                            | 0.7                              | 0.131                                 | 0.032                              | 0.174                              | 4.985           | 4.016        | 8.104        |

**Headline (rounded conditional rate medians):**

| Signal | Median P(next | active previous week) | Median P(next | inactive previous week) | Median weekly lift |
|---|---:|---:|---:|
| Issue comment | 27.7% | 1.9% | **14.793×** |
| PR review comment | 24.4% | 1.0% | **21.243×** |
| Push | 65.0% | 13.1% | **4.985×** |

- **Strong predictive association:** Recent participation relates to next-week participation in each direct-event signal.
- Large lifts for commenting may be driven partly by extremely small inactive-group base rates; compare precision/recall under actual trained models and account for rare positives.
- **Not an intervention or causal effect** and **not F1/accuracy**. The observations alone cannot select a predictive decision threshold or guarantee out-of-time generalization.
- Candidate baseline: prior-7-day flag; counts in prior 7/30/90 days; active weeks in recent horizon; days since last activity; activity diversity; bursts and decays. Compare under time-purged validation.

### 5.8 EDA-02.H — Repository diversity versus future push activity

For each of 12 cutoffs: compute **distinct `repo_id` values of pushes during prior 30 days**, grouped into `0`, `1`, `2–3`, `4+`; then measure **following-week push occurrence**. All 9,000 cohort users remain in the denominator of some group each week. These are **median weekly** rates, so activity group sizes may vary by cutoff.

| prior_30d_unique_repos   | mean_users_per_week   | median_future_push_pct   | min_future_push_pct   | max_future_push_pct   |
|:-------------------------|:----------------------|:-------------------------|:----------------------|:----------------------|
| 0                        | 4565.75               | 5.26                     | 1.24                  | 10.54                 |
| 1                        | 2099.58               | 31.28                    | 7.29                  | 47.5                  |
| 2–3                      | 1456.08               | 46.36                    | 14.87                 | 67.56                 |
| 4+                       | 878.58                | 70.65                    | 31.36                 | 90.27                 |

**Summary:** 0 previously pushed repositories → **5.26%** median next-week push occurrence; 1 → **31.28%**; 2–3 → **46.36%**; 4+ → **70.65%**. The 4+ group averaged **878.58** cohort users per week across cutoffs; group 0 averaged **4,565.75**.

**Serious confounder:** More distinct repos implies at least some push activity and correlates with total push count, last-push recency, and engagement. **This is a relationship-feature hypothesis**, not evidence of incremental relational model performance. Control for prior push counts/recency before making improvement claims.

### 5.9 EDA-02.I — Daily release comparison and runtime status

The notebook read `experiments/eda_snapshot_history.jsonl` (written optionally by EDA-01), without writing any new snapshot entry. One snapshot entry existed and matched the current official commit SHA.

**Row-count comparison against that single logged snapshot:**

| table            | last_logged_rows   | current_rows   | difference   |
|:-----------------|:-------------------|:---------------|:-------------|
| Users            | 45590              | 45590          | 0            |
| Repositories     | 109081             | 109081         | 0            |
| PullRequests     | 301518             | 301518         | 0            |
| Issues           | 172007             | 172007         | 0            |
| IssueComments    | 106204             | 106204         | 0            |
| PRReviews        | 90710              | 90710          | 0            |
| PRReviewComments | 66422              | 66422          | 0            |
| Pushes           | 1661256            | 1661256        | 0            |

**Outcome:** eight zeros. This was **not** an experiment comparing two daily releases. `test_input.csv` was still absent, as expected before its scheduled Oct 28 release. To measure dataset drift across releases, first preserve the previous SHA plus row/time counts, then re-clone and execute again on a *new SHA*; new filesystem mtimes alone do not establish new upstream data.

### 5.10 EDA-02 documented decision and limits

1. **Regime shift:** real change in recorded participation and intensity, with unknown cause; no justified seasonal/ETL conclusion.
2. **Label integrity:** issues are clearly timing-sensitive; all first-observation proxies must remain labeled as proxies; current source tables may not provide fully reconstructable exact official action labels.
3. **Selection:** official users are a subset selected for recent activity; use proxies only for sensitivity analysis, never present them as final roster.
4. **Predictability:** prior activity and repository breadth have strong observational associations with next-week activity; no model score measured.
5. **Next milestone:** create and test a six-label specification before any trained-model comparisons.

---

## 6. Cross-study synthesis and hypotheses

### 6.1 Evidence ledger: statements an agent may safely reuse

| Claim | Evidence | Scope/confidence | Inference boundary |
|---|---|---|---|
| Eight-table PK/FK structure valid | EDA-01 PK/FK audit | **Measured** for SHA `506d860...` | Does not certify label validity. |
| Cohort is exactly 9,000 and bots are outside it | EDA-01 cohort cross-tab, EDA-02 sanity check | **Measured** for snapshot | Future release must be rechecked. |
| Push rows strongly dominate the database | EDA-01 count table | **Measured** count | Different table rows have different meanings. |
| Mid-year recorded comments dramatically declined | EDA-02 regime medians | **Measured descriptive trend** | Cause unverified; periods unequal length. |
| Push activity in recovery-labeled period lower than early period | EDA-02 regime medians | **Measured** | Do not state blanket September recovery for push. |
| Issue entity visibility often delayed | EDA-01 global and EDA-02 non-PR/cohort lags | **Measured** | Lag does not quantify official-label error. |
| First-seen and candidate-event weekly labels disagree | EDA-02 eight-week sets | **Measured** | Neither candidate is independently proven official truth. |
| Historically eligible activity-selected users more active | EDA-02 proxy selection | **Measured conditional rates** | Proxy may not equal official selection. |
| Prior-week activity associated with subsequent occurrence | EDA-02 direct-event persistence | **Measured association** | Does not equal classifier F1 or causality. |
| More repositories previously pushed to associated with pushing next week | EDA-02 relational breadth study | **Measured association** | Confounded by prior push volume/engagement. |
| Recent time or relation features *improve* a fitted model | None yet | **Untested hypothesis** | Requires controlled ablations. |
| GNN/RDL > feature-engineered baseline | None yet | **Untested hypothesis** | Do not claim on slides. |
| Dataset changed after daily refresh | None yet | **Untested** | Same SHA in both EDA runs. |

### 6.2 Working hypotheses for testing (NOT conclusions)

**H1 — Recency matters.** Last-week flags, days-since-last-event, and 7/30/90-day counts yield better out-of-time F1 than static whole-history counts because of demonstrated behavioral persistence.

**H2 — Temporal weighting helps under regime shifts.** More recent training examples or features may generalize better than naive uniform all-history aggregation. Counter-hypothesis: longer history helps rare targets; evaluate both.

**H3 — Relation breadth has independent value.** Repository diversity and collaborative neighborhoods may improve F1 beyond matched prior user-event volume; first run count-matched analyses and model ablations.

**H4 — Label construction must distinguish visibility and action occurrence.** Candidate event time should be evaluated and official evaluation semantics investigated using only permitted materials. If exact action labels cannot be recovered, mark them uncertain and analyze downstream sensitivity.

**H5 — Selection-aware validation better estimates official behavior.** An eligible-only historical sample may resemble the test roster more closely than all-cohort evaluation, but must be verified against the Oct 28 roster and protocol.

**H6 — Multiple lookback horizons complement one another.** Compare 7 vs 7+30 vs 7+30+90-day features on identical chronological partitions; rare class results may require longer horizons.

### 6.3 Confounders and recurrent analytical traps

1. **Cohort vs referenced users:** Do not regard 45,590 Users rows as 45,590 complete longitudinally tracked users.
2. **Different denominators:** Full Issues (172,007), non-PR Issues (125,508), and cohort-attributed non-PR Issues (83,543) must not be conflated.
3. **Missing PR authors:** Selecting known cohort actors removes records with unknown authors by construction; it does not mean the missing-author problem has been solved globally.
4. **Different timestamps:** `first_seen_at` is the official feature-side availability timestamp for certain entities; `created_at/submitted_at` is candidate target event time; neither role can be swapped casually.
5. **Retroactive inclusion:** Modern reconstructed Parquet rows may reveal older events discovered only later. Time-safe feature construction cannot use these rows before first observation; historical targets need their own validation/maturity policy.
6. **Imbalance:** Accuracy-only metrics misleading for sparse activities; record per-label confusion counts, prevalence, precision, recall, F1, and combined official score once labels are trustworthy.
7. **Weekly median:** A median across period-level weekly rates is not overall pooled event probability. Report aggregation method each time.
8. **Jaccard with unequal periods:** 26 vs 14 vs 5-week set overlaps are descriptive, not normalized churn rates.
9. **Repository breadth confounding:** 0 prior repos means no prior pushes by definition; compare with recency/count baselines and within-volume strata.
10. **Partial weeks:** Week containing the latest event cannot be treated as complete; a historical week may have late-arriving records even after Monday passes.
11. **Automatic daily dataset refresh:** EDA notebooks read local clone; they do not automatically download; rerun on a newly cloned release after 21:00.
12. **Source versions:** Never merge or compare release-level row-number PKs without resolving stable GitHub IDs.
13. **Official test selection:** Current ≥5-in-90-day six-table criterion is a convenience **proxy**, not guaranteed to match the TA's selection logic.

### 6.4 Data processing dependency diagram

```text
  Official course GitHub repo [one force-replaced dataset snapshot per day]
                               |
                       manual re-clone
                               v
              assignment2/data/db/*.parquet
                               |
                               v
                 Snapshot SHA + schema audit
                               |
             +-----------------+--------------------+
             |                 |                    |
             v                 v                    v
     Direct event tables  Entity observation    Cohort identity
     (push, comments)     tables (PRs, etc.)   Users.is_cohort
             |                 |                    |
             |                 v                    |
             |         Distinguish first_seen       |
             |         from candidate event time   |
             |                 |                    |
             +-----------------+--------------------+
                               |
                        Historical cutoff t
                               |
                  +------------+-------------+
                  |                          |
          historical features         future 7-day labels
          row time < t                [t, t+7d), valid semantics
          late-known masked           maturity checked
                  |                          |
                  +------------+-------------+
                               |
                  chronological train / val / test
                               |
                  baseline -> ablations -> scoring
```

---

## 7. Engineering decisions and experiment backlog

### 7.1 Original decision register as of EDA-02 (historical; superseded by §12)

| ID | Status | Decision | Rationale / evidence | What would reverse the decision |
|---|---|---|---|---|
| D-01 | Accepted | Keep official `assignment2/` clone isolated and ignored by personal Git | Instructor force-pushes daily; daily re-clone safe when code elsewhere | None, barring official repo policy change |
| D-02 | Accepted | Record instructor **full SHA** for every analytical run | Prevent comparison of identical vs different releases | None |
| D-03 | Accepted | Use `github_user_id` as durable external identity; FK row numbers only within release | IDs can change across releases | None |
| D-04 | Accepted | Use row visibility timestamps for historical features | Official schema and temporal leakage concerns | None |
| D-05 | Accepted | Remove late-known fields from first baseline feature set | Future values identified in Oct 1 audit | Reintroduce only with time-safe audited logic |
| D-06 | Accepted | Begin modeling with simple historical counts/recency and a transparent baseline | Strong persistence; smaller engineering/reproducibility burden | Empirical out-of-time comparison |
| D-07 | Pending | Choose exact historical labels for open-PR, non-PR issue, review | Proxy/candidate timing disagreement remains | Proven original-event provenance in allowed official data |
| D-08 | Pending | Choose time splits, embargo/label-maturity rules | Shifting distribution, delayed discovery | Backtesting maturity audit and temporal coverage checks |
| D-09 | Pending | Decide feature weighting / lookback range | Temporal shift and sparse positives | Time-based validation results |
| D-10 | Pending | Choose relational graph / GNN complexity | Relational breadth association only, no marginal gain evidence | Controlled holdout ablation |
| D-11 | Pending | Confirm official test population and its positive prevalence | test_input absent at current snapshot | Oct 28 official release / future truth |
| D-12 | Pending | Verify dataset collection-regime artifact vs behavior | Observation trend cannot isolate cause | Allowed data-only diagnostics, release comparisons |

### 7.2 Original follow-up backlog after EDA-02 (some now completed; see §12)

#### EXP-LABEL-01 — Target semantics and historical-ground-truth feasibility

**Goal:** Define how to determine each of the six target activities and whether the supplied data supports exact labels. Include explicit exclusion of PR-conversation rows from `open_issue`. Distinguish entity creation, observation and action emission.

**Plan:**
1. Read official `data/README.md`, `SCHEMA.md`, evaluation docs and relevant allowed source information; do **not** collect supplemental GitHub events.
2. Inspect candidate actor/time fields, missingness, duplicate events and late-discovery patterns per target.
3. Compare visibility-week and original-candidate-week memberships for several past complete weeks and report mismatches by target, user and lag stratum.
4. For any target not provably reconstructable from allowed data, define a labeled **proxy** and quantify sensitivity instead of claiming exact truth.
5. Write a source-specific target definition table and unit tests.

**Success criterion:** Label rules are reproducible and their uncertainty is explicit. Do **not** invent inferred PR opening events for missing author/time rows.

#### EXP-SPLIT-01 — Historical temporal generator and leakage tests

**Goal:** Build `(stable_github_user_id, prediction_cutoff, features, six_labels)` samples using a known dataset SHA and future 7-day windows. Separate features from labels in construction and inputs.

**Initial historical approach:** Chronological training on older cutoffs; recent later weeks for validation; last untouched mature weeks for holdout; no random row-level splitting. Consider purge boundaries between sample label windows and subsequent validation feature selection; do not let future-labeled examples enter training ahead of a validation cutoff.

**Acceptance tests:**
- No record with official row time `>= t` enters feature construction for sample cutoff t.
- Six late-known columns excluded or demonstrably masked at every t.
- All target windows exactly `[t, t+7d)` Asia/Taipei with no overlap mistakes.
- User IDs not duplicated within one cutoff; stable GitHub ID used for exported samples.
- Same snapshot and configuration yield identical feature/label outputs.
- Missing/unknown event attribution remains explicit and does not silently become negative labels.
- A past snapshot that receives retroactive rows on a later release is detectable via SHA and validation checks.
- Historical candidate labels are not presented as verified official truth unless supported.

#### EXP-BASE-01 — Simple temporal baseline

**Models to consider:** per-label base rate, previous-week occurrence heuristic, regularized logistic regression, gradient-boosted trees (e.g. LightGBM/CatBoost if installed later). No model chosen or run yet.

**Common features (all built before cutoff):** prior 7/30/90-day counts; binary activity presence; days since last activity; active-week count; count trend; event diversity; activity density/dispersion. Use all six channels carefully, distinguishing verified direct events and observation proxies. Start with safe feature columns only.

**Metric protocol:** For each target collect TP/FP/FN, precision/recall/F1, support/prevalence and class-specific thresholds; compute official macro-F1, micro-F1, and their mean from held-out predictions. Record validation and holdout **separately**. Threshold tuning only on validation, not the final untouched holdout.

**Baseline comparison table (not filled — no results yet):**

| Experiment | Features | Historical train dates | Val dates | Test dates | macro-F1 | micro-F1 | Score | Status |
|---|---|---|---|---|---:|---:|---:|---|
| B0 | Majority / base rate | TBD | TBD | TBD | — | — | — | Not run |
| B1 | Last-week flag | TBD | TBD | TBD | — | — | — | Not run |
| B2 | 7/30-day counts+recency | TBD | TBD | TBD | — | — | — | Not run |
| B3 | 7/30/90-day + trends | TBD | TBD | TBD | — | — | — | Not run |

#### EXP-TIME-01 — Temporal ablations

Test constant all-history vs 7 vs 7+30 vs 7+30+90-day aggregation; time-decayed counts, rolling slope/trends, and down-weighting older training weeks. **Keep cutoff schedule, cohort selection, labels, model capacity, random seeds and thresholds protocol controlled**. Analyze period-specific and per-class F1.

#### EXP-REL-01 — Relational feature ablation

Compare user-event baseline with +unique repos, +repository activity totals, +distinct collaboration partners, +multi-hop relation aggregates, always computed point-in-time. Compare repository-breadth groups **conditional on the same prior push count** to test whether prior breadth adds information beyond usage. Avoid future entity joins.

#### EXP-RDL-01 — Optional relational GNN comparison

Only after the baselines and labels work: graph construction using the eight PK/FK table relations, typed node/edge features, and cutoff-respecting temporal neighborhood sampling. Test GNN/RDL on **identical** time splits and compute costs/runtime; do not assume lecture examples imply superior performance on this dataset.

#### EXP-DRIFT-01 — Release-to-release drift

After each new **21:00 Taiwan** course release: safely re-clone, record commit, per-file size/hash/row count/time min-max, cohort consistency, global feature distributions, and counts of rows whose candidate original date is in prior weeks but whose visibility date is later. Compare by *stable* GitHub IDs across releases if needed. Do not equate new local file mtimes with real upstream changes.

### 7.3 Suggested repository artifacts as experiments mature

```text
Assignment 2/
├── README.md
├── environment.yaml
├── EXPERIMENT_LOG.md           # THIS persistent, append-only log
├── assignment2/               # ignored replaceable official clone
│   ├── data/db/*.parquet
│   └── evaluation/evaluate.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_deeper_data_analysis.ipynb
│   └── 03_label_and_split_validation.ipynb      # PLANNED
├── src/
│   ├── data.py                                   # PLANNED
│   ├── temporal_dataset.py                       # PLANNED
│   ├── features.py                               # PLANNED
│   ├── train.py                                  # PLANNED
│   └── predict.py                                # PLANNED
├── tests/
│   ├── test_labels.py                            # PLANNED
│   └── test_leakage.py                           # PLANNED
└── experiments/
    ├── eda_snapshot_history.jsonl
    ├── metrics/                                  # OPTIONAL, future
    └── configs/                                  # OPTIONAL, future
```

---

## 8. Release management, open issues, and append-only template

### 8.1 Rules for updating this log

1. **Preserve existing historical EDA numbers** for snapshot `506d860...`; never rewrite them to match a later instructor release.
2. Add a **new dated release entry** for each distinct instructor SHA, not for every rerun of identical data.
3. Add a new **experiment entry** whenever the code/configuration/feature definition/split changes materially, even if data SHA stays the same.
4. Keep exact **denominator**, timestamp timezone, week inclusivity, cohort/eligibility definition, table/column semantics, metrics and sample counts with each new result.
5. If a result changes because a bug was discovered, preserve the original result as **invalidated** and append the correction + cause, code commit, data SHA, and re-run metrics.
6. Never fabricate unobserved results, numerical gains, missing labels or official evaluation outcomes. Use `TBD`, `Not run`, or `Unknown` where appropriate.
7. If an agent prepares slides, only use clearly measured results, preserve caveats, and do not turn hypotheses into claims of improvement.
8. For code reproducibility, record personal-project commit SHA **separately** from the instructor dataset SHA.

### 8.2 Upcoming monitoring milestones

| Milestone | Date (Taiwan time) | Action |
|---|---|---|
| Instructor daily refresh | Every day **21:00**, Oct 14–Nov 11 | Re-clone official dataset when necessary; compare full SHA before logging drift |
| Official testing user IDs | **Oct 28** | Read `evaluation/test_input.csv`, validate all stable IDs, characterize cohort selection without external collection |
| Final dataset | **Nov 11 21:00** | Includes observations through **Nov 11 20:00**; rerun already-tested end-to-end pipeline |
| Assignment submission | **Nov 11 23:55** | Public reproducible GitHub code + valid prediction CSV + PDF slides + MP4 |
| Target future window | **Nov 12–Nov 19** | Final, unobserved at submission time; exact official targets are not provided in historical EDA |

### 8.3 Unresolved questions / research constraints

- **Q1 — What do the original event/action types guarantee?** The supplied entity-oriented tables may lack sufficient provenance to exactly reconstruct `open_pr`, `open_issue`, and `pr_review` emission events for every historical week. Do not assume completeness.
- **Q2 — Why the June–August dip?** A common upstream collection issue, event-type selective loss, sampling, timing of cohort selection, seasonality, and real behavior can create similar descriptive plots. No explanation verified.
- **Q3 — How stable is the snapshot reconstruction?** A later release may retroactively include older entities; until a second distinct SHA is analyzed, quantitative release drift is unknown.
- **Q4 — Does the activity-selection proxy match official test IDs?** Cannot confirm until Oct 28; proxy uses observation counts and differs from potentially more precise TA event accounting.
- **Q5 — Which targets benefit from particular modeling features?** All relationships discovered are descriptive. No supervised F1 estimates yet.
- **Q6 — Is temporal generalization robust across regime boundaries?** Not tested; require multiple later held-out weeks and optionally rolling origin comparisons.
- **Q7 — Are June–August low-count windows representative?** Unknown; use period-specific metrics and sensitivity analysis rather than silently dropping them.

### 8.4 Append-only new release entry template

Copy this for each newly confirmed **distinct instructor SHA**:

```markdown
### RELEASE-YYYYMMDD — Instructor dataset refresh

- **Observed at (Taiwan):** YYYY-MM-DD HH:MM
- **Instructor Git SHA:** full SHA
- **Previous compared SHA:** full SHA
- **Local project Git SHA:** full SHA
- **Source files:** assignment2/data/db/*.parquet
- **Rows per table (before -> after):** ...
- **Latest observation in UTC / Taiwan:** ...
- **Schema/PK/FK status:** ...
- **Stable user population check:** ...
- **New rows / retroactive changes / moved row IDs:** ...
- **Warnings / quality checks:** ...
- **Changed EDA conclusions, if any:** ...
- **Reproduce:** conda env, notebook/script commands, code commit, input SHA
```

### 8.5 Append-only new model/analysis experiment template

```markdown
### EXP-YYYYMMDD-NN — Descriptive experiment title

- **Question / hypothesis:** ...
- **Status:** Planned | Running | Completed | Failed | Invalidated
- **Dataset:** instructor SHA, latest available event time, source files
- **Code:** personal-project commit SHA / script and notebook path
- **Environment / seed:** versions, execution hardware if relevant, seed(s)
- **Cohort / eligibility:** exact membership rule, unique users
- **Prediction cutoffs / horizon:** dates, timezone, inclusion rules
- **Label definitions / caveats:** one definition per target, missingness handling
- **Train / validation / historical test:** chronological ranges, sample counts
- **Features / model / hyperparameters:** fully specified
- **Comparator:** baseline and precise controlled differences
- **Primary metrics:** per-label TP/FP/FN, F1; macro-F1, micro-F1, combined score
- **Additional metrics:** precision, recall, prevalence, calibration, runtime, memory
- **Ablation/statistical uncertainty:** exact procedure and outcomes
- **Observed results (table):** ...
- **Interpretation supported by data:** ...
- **Alternative explanations / limitations:** ...
- **Decision:** keep/change/drop and why
- **Next follow-up:** ...
```

### 8.6 Submission and research integrity safeguards

- Public GitHub code must include data ingestion, features, training/inference and a reproducible `README.md`; keep instructor dataset clone ignored.
- Final CSV header **exactly** `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`; one row per official test user; six binary 0/1 fields; check with `assignment2/evaluation/evaluate.py`.
- Video **8–12 minutes**; slides English, at most **10 pages**; answer Q1 schema+entity graphs, Q2 relations, Q3 time, Q4 pipeline. Any experimental result claimed should appear in both slides and video and match the code/log.
- The assignment allows AI help with reasoning but forbids an AI-generated submission video and penalizes hallucinated findings; measurements in this log must not be embellished.
- Note the assignment PDF contains an internal naming inconsistency (`Asg_` vs `Asg2_`) for slide/video files; verify with E3/TA before submission.

---

## 9. References and traceability

### 9.1 Primary evidence (these have actually been inspected)

1. **Executed EDA-01:** `01_data_audit(1).ipynb`, 32 cells; contains stored tables, stdout, and plotted figures; use original project name `notebooks/01_data_audit.ipynb` when reproducibly rerunning.
2. **Executed EDA-02:** `02_deeper_data_analysis(1).ipynb`, 28 cells; includes expanded regime, lag, eligibility, persistence and relation-breadth tables; use project name `notebooks/02_deeper_data_analysis.ipynb` for reruns.
3. **Assignment PDF:** `Big Data Course - Assignment 2.pdf` (10 pages), particularly pages 3–9 for tasks, data rules, scoring and grading.
4. **Official data README:** https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/README.md
5. **Official schema:** https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/db/SCHEMA.md
6. **Official evaluation README:** https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/README.md
7. **Official evaluator implementation:** https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/evaluate.py
8. **Assignment reference:** `BigData_Assignment2_Agent_Reference.md` (created previously; explanatory secondary reference, not substituted for actual experiments).
9. **Course Lecture 5:** `BigData_115_lec05_RelationalData-1 （Update).pdf` and `BigData_115_lec05_RelationalData-2.pdf`, conceptual background for schema/entity graphs, RDL, GNNs, time-dependent sampling and relation modeling; no direct result in these notebooks depends on implementing them.

### 9.2 Notebook section → log mapping

| Original notebook | Executed cell index (zero-based) | Logged as |
|---|---|---|
| 01 | 3, 5, 7 | EDA-01 environment, ingestion, schema (§4.1) |
| 01 | 9, 11, 13 | EDA-01 integrity, user cohort, coverage (§4.2–4.4) |
| 01 | 15, 17, 19 | EDA-01 signals, trailing windows, repo breadth (§4.5–4.7) |
| 01 | 21, 23, 25 | EDA-01 missingness, lag, cutoff audit (§4.8–4.10) |
| 01 | 27, 29 | EDA-01 test-list and history status (§4.11) |
| 02 | 4, 6, 8 | EDA-02 setup, coverage and observation index (§5.1) |
| 02 | 10, 12 | EDA-02 periods and cohort turnover (§5.2–5.3) |
| 02 | 14, 16 | EDA-02 visibility lag and weekly proxy disagreement (§5.4–5.5) |
| 02 | 18, 20, 22 | EDA-02 eligibility, persistence, repository breadth (§5.6–5.8) |
| 02 | 24, 26 | EDA-02 release comparison and generated summary (§5.9–5.10) |

---

**End of the original EDA-01/02-only source log. The following sections are appended results and updated engineering decisions; the original numerical observations above are preserved under instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`.**

---

## 10. EDA-03: Historical labels, temporal splits and rule-based baselines

**Experiment ID:** `EDA-03`  
**Executed artifact:** `notebooks/03_label_and_split_validation.ipynb`, supplied executed copy `03_label_and_split_validation(5).ipynb` (33 total cells, original 15 notebook code cells plus 1 added confusion-matrix cell; saved output present).  
**Experiment date:** 2026-10-14 Asia/Taipei.  
**Status:** Completed exploratory protocol validation; historical holdout future labels sealed.  
**Instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`.  
**Scoring implementation:** Exact local `assignment2/evaluation/evaluate.py`, dynamically imported as module `official`; `official.scores()` and `official.f1_per_event()` used.  
**No external GitHub activity collected. No fitted/learned classifier trained.**

### 10.1 Purpose, scope, and relation to EDA-01/02

EDA-01 audited the eight tables and showed important lag and imbalance. EDA-02 documented temporal-regime changes, previous-week persistence and differences between original creation times and first observation. EDA-03 was designed to turn these descriptive findings into **auditable historical examples and baseline scores** without losing point-in-time constraints.

The goals were: (i) instantiate cohort-attributed six-target event streams; (ii) compare the two historical label time definitions; (iii) build strict non-overlapping calendar splits; (iv) construct a 90-day activity-count eligibility proxy; (v) evaluate two non-trained baselines with the **instructor's evaluator**; (vi) keep a proposed holdout sealed; (vii) check temporal integrity. An optional sample-truth comparison motivated EXP-004.

This experiment produces **user–cutoff samples**, not a final feature-rich model-ready training matrix, and no fit of logistic regression, trees, neural networks, or GNNs was performed.

### 10.2 Exact environment and configuration

Saved notebook configuration:

```python
TZ = 'Asia/Taipei'
HORIZON_DAYS = 7
FEATURE_LOOKBACK_DAYS = 90
ELIGIBLE_MIN_RECORDS = 5
LABEL_MATURATION_DAYS = 7
VALID_START_TW = '2026-08-03'
HOLDOUT_START_TW = '2026-09-07'
LATEST_CUTOFF_TW = None
EVALUATE_HOLDOUT = False
RUN_SAMPLE_TRUTH_CHECK = True  # enabled in executed (5) copy
EXPORT_LOCAL_FILES = False
```

- `LABEL_MATURATION_DAYS=7` is a **study scheduling buffer**, **not** a guarantee that all historical entity rows were collected within seven days.
- `FEATURE_LOOKBACK_DAYS=90` applies to the cutoff calendar and activity-based eligibility proxy; the simple `previous_week` heuristic uses the preceding **7 days** only.
- `EVALUATE_HOLDOUT=False` causes six future event labels **and** six seen-time labels for holdout windows to be stored as nullable/uncomputed, rather than just hiding their scores.
- `RUN_SAMPLE_TRUTH_CHECK=True` reads the official example truth for diagnostic calibration. This is distinct from opening the notebook's own historical-holdout labels, **but it compromises independence of overlapping September calendar periods for label-policy selection**.
- Rerunning against a different instructor release must generate a **new experiment entry and identify the new SHA**. This run shares the same SHA as EDA-01/02.

#### Source mapping: actor, row visibility, candidate event timestamp

| Target | Table | Actor FK (join to `Users.user_id`) | Historical row-visibility time | Candidate label event time | Additional condition |
|---|---|---|---|---|---|
| `open_pr` | `PullRequests` | `author_id` | `first_seen_at` | `created_at` | actor maps to cohort |
| `open_issue` | `Issues` | `author_id` | `first_seen_at` | `created_at` | `is_pull_request=False`; actor maps to cohort |
| `issue_comment` | `IssueComments` | `author_id` | `created_at` | `created_at` | actor maps to cohort |
| `pr_review` | `PRReviews` | `reviewer_id` | `first_seen_at` | `submitted_at` | actor maps to cohort |
| `pr_review_comment` | `PRReviewComments` | `author_id` | `created_at` | `created_at` | actor maps to cohort |
| `push` | `Pushes` | `user_id` | `created_at` | `created_at` | actor maps to cohort |

The code builds a stable GitHub actor map from `Users.user_id` to `Users.github_user_id`, then saves rows with `UserID`, `seen_utc`, and `event_utc`. All six labels are **binary user-week existence indicators**, not event counts. Rows are selected using `[start, end)` and a source-specific timestamp. Null/unattributed/non-cohort actors are not claimed to represent complete histories.

A key nuance: the code initially creates an `event_time` candidate **without applying a historical observation-as-of cutoff** to its labels; that is why EXP-004's later cutoff analysis must be recorded as a separate, calibrated variant rather than quietly assumed to have been used in EDA-03's baseline metrics.

### 10.3 Official scoring implementation and conventions

The official module contains `EVENT_COLS` in exactly this order:

```text
open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push
```

For one target with TP, FP and FN:

\[
F1_k = \frac{2 TP_k}{2 TP_k + FP_k + FN_k}.
\]

The instructor's `_f1` function returns **1.0 when `TP+FP+FN == 0`** (both truth and prediction have no positives). For all other cases the fraction above applies. The notebook verified this convention with a smoke test.

\[
\text{macro-F1}=\frac{1}{6}\sum_{k=1}^6 F1_k,
\quad
\text{micro-F1}=\frac{2 \sum_k TP_k}{2\sum_kTP_k+\sum_kFP_k+\sum_kFN_k},
\quad
\text{Score}=\frac{\text{macro-F1}+\text{micro-F1}}{2}.
\]

The official implementation is the authority; avoid silently substituting sklearn defaults. User row ordering in predictions is reconciled by `UserID`. Separate `read_predictions()` enforces exact seven-column header, binary values, nonnegative-integer formatting, no duplicates, and expected test user ID set.

**Aggregation warning:** The EDA-03 baseline tables report **mean/median/min/max over weekly official scores**. For example, the reported validation mean **0.3504** is an **arithmetic mean of five individually computed weekly combined scores**, not the score of one pooled 37,504-user-week array or the final official test submission.

### 10.4 Cohort and observation coverage checks

- `Users` rows: **45,590**.
- Cohort (`is_cohort=True`): **9,000**; referenced other users **36,590**.
- Six cohort-attributed source row counts: `open_pr` **130,043**, `open_issue` **83,543**, `issue_comment` **106,204**, `pr_review` **89,538**, `pr_review_comment` **66,422**, `push` **1,661,256**.
- Observed cohort signal time range: **2025-12-01 00:01:12 UTC** through **2026-10-13 12:59:29 UTC** (**2026-10-13 20:59:29 Taiwan**).
- These maxima indicate records observed in the current snapshot; the latest event timestamp is **not** an independently established ingestion completeness or publication watermark.
- Cohort actor timestamp values were present for the six constructed activity streams; that is conditional on actor mapping and does not cure global PR author missingness.

### 10.5 Calendar design, maturity gate, and split boundaries

**Window start:** Monday 00:00 Asia/Taipei; seven-day future horizon; feature history lookback 90 days. The first cutoff must follow the start of available data by at least 90 days; the latest cutoff must have a future seven-day window **plus** configured seven-day buffer before the latest observed timestamp.

The notebook generated **31 consecutive complete-week cutoffs**:

| Split | Cutoff weeks | First cutoff (Taiwan) | Last cutoff (Taiwan) | User-week rows | Proxy-eligible user-weeks |
|---|---:|---|---|---:|---:|
| Training | 22 | 2026-03-02 00:00 | 2026-07-27 00:00 | **198,000** | **158,269** |
| Validation | 5 | 2026-08-03 00:00 | 2026-08-31 00:00 | **45,000** | **37,504** |
| Historical holdout | 4 | 2026-09-07 00:00 | 2026-09-28 00:00 | **36,000** | **20,790** |
| **All** | **31** | 2026-03-02 | 2026-09-28 | **279,000** | **216,563** |

- Each cutoff has **9,000** distinct cohort users, one row per `(UserID, cutoff)` pair.
- Labeled training plus validation examples: **243,000 user-weeks**, and **6 × 243,000 = 1,458,000** candidate binary event-label cells evaluated in the disagreement audit.
- Holdout rows **36,000** exist only with **sealed** future labels; eligibility and previous-week feature proxies were computed on those rows, but their ground-truth outcomes were not.
- Training prediction windows are `[cutoff, cutoff+7d)` and finish at or before the validation cutoff. Validation weeks finish at or before the holdout cutoff. No random row-based splitting was used.
- Maximum allowable cutoff given the 7+7 day boundary: **2026-09-29 20:59:29 Taiwan**; the last aligned Monday cutoff is Sep 28.
- **These Monday midnight historical cutoffs are not the same as the later-discovered official example's Sunday 21:00 Taiwan boundary.** The calendar was deliberately exploratory; the official November target has yet another explicit midnight boundary.

### 10.6 Proxy eligibility and shifting sample population

The notebook defines `eligible_proxy = (number of rows observed during the previous 90 days across the six constructed activity-related streams >= 5)`, where history rows require `seen_utc < cutoff`. This is **not** the TA's exact Oct 28 user selection algorithm, although it follows the published broad five-record/90-day principle.

Selected full user counts from the executed notebook:

| Week cutoff, Taiwan | Split | Proxy eligible (of 9,000) | Share |
|---|---|---:|---:|
| 2026-07-13 | Training | 8,595 | 95.50% |
| 2026-07-20 | Training | 8,566 | 95.18% |
| 2026-07-27 | Training | 8,532 | 94.80% |
| 2026-08-03 | Validation | 8,489 | 94.32% |
| 2026-08-10 | Validation | 8,390 | 93.22% |
| 2026-08-17 | Validation | 7,803 | 86.70% |
| 2026-08-24 | Validation | 6,792 | 75.47% |
| 2026-08-31 | Validation | 6,030 | 67.00% |
| 2026-09-07 | Holdout, labels sealed | 5,656 | 62.84% |
| 2026-09-14 | Holdout, labels sealed | 5,296 | 58.84% |
| 2026-09-21 | Holdout, labels sealed | 5,056 | 56.18% |
| 2026-09-28 | Holdout, labels sealed | 4,782 | 53.13% |

Interpretation: substantial shift in the **population satisfying the approximate activity filter** as older busy weeks leave the rolling window. User-week rows and unique users are different quantities: the 158,269 training proxy-eligible samples are **not 158,269 different GitHub accounts**. It remains possible that official selection differs in how it counts actions versus entity observations.

### 10.7 Candidate event-time versus first-seen labels: all development windows

The notebook compared binary membership by activity at every cutoff in **22 training + five validation = 27** windows (243,000 user-weeks per target). The two label variants used exactly the same users, cutoffs, and seven-day horizons; only the timestamp field differs.

| Target | Event-time positives | First-seen positives | Event-only | Seen-only | Total discordant user-weeks | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| `open_pr` | 25,172 | 25,243 | 349 | 420 | **769** | 0.9700 |
| `open_issue` | 10,992 | 11,429 | 933 | 1,370 | **2,303** | 0.8137 |
| `issue_comment` | 15,522 | 15,522 | 0 | 0 | **0** | 1.0000 |
| `pr_review` | 13,230 | 13,230 | 9 | 9 | **18** | 0.9986 |
| `pr_review_comment` | 8,692 | 8,692 | 0 | 0 | **0** | 1.0000 |
| `push` | 107,223 | 107,223 | 0 | 0 | **0** | 1.0000 |

Definitions: `event-only` means positive in the original event-time candidate but negative in the first-seen candidate; `seen-only` is the reverse. **Discordance is between label-generating methods**, not an error count versus the official truth. The direct-event label variants agree exactly *by construction*, since source `event_utc == seen_utc` for those streams.

**Interpretation:** Non-PR issue opening is markedly more sensitive to timestamp selection than other targets. Thus the initial EDA recommendation to use `first_seen_at` only for features, not naively as an opening time, was justified. The subsequent official sample comparison independently supports event-time labels when the correct window and observation policy are used (EXP-004).

### 10.8 Training/validation candidate-label prevalence and period shift

These are prevalence percentages across **proxy-eligible user-week samples** using **event-time candidate** future labels. Denominators are **158,269 training user-weeks** and **37,504 validation user-weeks**, not whole-cohort user counts and not verified final official November prevalence.

| Target | Train positive % | Validation positive % | Change, percentage points |
|---|---:|---:|---:|
| `push` | **57.671** | **23.227** | **−34.444** |
| `open_pr` | 13.610 | 4.458 | −9.152 |
| `issue_comment` | 8.578 | 3.456 | −5.122 |
| `pr_review` | 7.506 | 2.720 | −4.786 |
| `open_issue` | 5.816 | 2.736 | −3.080 |
| `pr_review_comment` | 4.931 | 1.856 | −3.075 |

**Measured implication:** The simple target distribution is substantially different between historical training and validation. The large push change is particularly significant. Because there is no fitted estimator yet, do not explain the score decline below as traditional learned-model overfitting. Plausible contributors include behavior changes, collection effects, historical user selection changes, and changes in persistence under different prevalence regimes. **No causal explanation has been established.**

For completeness, the corresponding event-time positive rates among the **proxy-ineligible** segments were:

| Target | Training, ineligible (%) | Validation, ineligible (%) |
|---|---:|---:|
| `open_pr` | 4.775 | 0.840 |
| `open_issue` | 1.847 | 0.360 |
| `issue_comment` | 1.515 | 0.627 |
| `pr_review` | 0.783 | 0.267 |
| `pr_review_comment` | 0.451 | 0.160 |
| `push` | 17.895 | 1.681 |

Those rates are conditional on the historical proxy group, not the official Oct 28 roster. The eligible/ineligible contrast reinforces the need to evaluate both the full cohort and the selected population under the same split.

### 10.9 Historical rule-based baselines and all recorded metric summaries

**Baseline A: `all_zero`.** For every included user and every target, output zero.  
**Baseline B: `previous_week`.** For every user and target, output one iff at least one corresponding **observable row** existed in the preceding seven days `[t−7d, t)`; otherwise zero. No parameter estimation or threshold tuning. The preceding seven-day feature uses the official **visibility time** (`seen_utc`), while the future reconstructed truth uses the candidate **event time** (`event_utc`).

Both baselines were scored **separately at every weekly cutoff** with official `official.scores()` and then summarized by mean, median, minimum and maximum across weeks. Two populations were considered: the full 9,000 cohort users and the eligibility proxy subset. There are **22 scored training weeks** and **five scored validation weeks**. The historical holdout was not scored.

#### Complete saved per-week score summary (four decimal places)

| Split | Population | Rule | Macro mean | Macro median | Macro min | Macro max | Micro mean | Micro median | Micro min | Micro max | Combined mean | Combined median | Combined min | Combined max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | All cohort | All zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Train | All cohort | Previous week | 0.4368 | 0.4442 | 0.2477 | 0.5988 | 0.6299 | 0.6302 | 0.5762 | 0.6652 | **0.5334** | 0.5323 | 0.4359 | 0.6285 |
| Train | Eligible proxy | All zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Train | Eligible proxy | Previous week | 0.4475 | 0.4738 | 0.2483 | 0.6140 | 0.6449 | 0.6495 | 0.5958 | 0.6777 | **0.5462** | 0.5539 | 0.4369 | 0.6459 |
| Validation | All cohort | All zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Validation | All cohort | Previous week | 0.2410 | 0.2420 | 0.2138 | 0.2722 | 0.4527 | 0.4814 | 0.2424 | 0.5481 | **0.3469** | 0.3530 | 0.2281 | 0.4096 |
| Validation | Eligible proxy | All zero | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Validation | Eligible proxy | Previous week | **0.2434** | 0.2424 | 0.2192 | 0.2731 | **0.4575** | 0.4860 | 0.2493 | 0.5498 | **0.3504** | 0.3560 | 0.2342 | 0.4112 |

A direct train-to-validation comparison for the **same eligible-proxy previous-week rule**:

- Training mean macro **0.4475**, validation **0.2434** (decrease **0.2041** absolute).
- Training mean micro **0.6449**, validation **0.4575** (decrease **0.1874** absolute).
- Training mean combined **0.5462**, validation **0.3504** (decrease **0.1958** absolute, about **35.8% relative** to 0.5462).
- The rule itself has **no fitted parameters**: this result supports **out-of-time degradation of the same heuristic** under changing populations/distributions, not learned-model overfitting.
- Across the evaluated weeks, `all_zero` obtained 0 under the instructor's functions; that implies each evaluated weekly/population truth contained positives across the targets (otherwise the `F1(empty,empty)=1` convention could make its macro nonzero).

#### Median per-label validation F1 (eligible-proxy population)

| Target | All zero | Previous week |
|---|---:|---:|
| `open_pr` | 0.0000 | **0.1463** |
| `open_issue` | 0.0000 | **0.2575** |
| `issue_comment` | 0.0000 | **0.1820** |
| `pr_review` | 0.0000 | **0.1735** |
| `pr_review_comment` | 0.0000 | **0.1629** |
| `push` | 0.0000 | **0.5641** |

**Interpretation:** Push is the strongest channel for the previous-week rule on these validation weeks; the rare opening/comment/review targets remain difficult. This is sufficient motivation to test per-target lookback windows, counts, trends, and thresholds **on validation only**, while preserving untouched future holdout windows. No claim has been made that a learned model can reach or beat any specific score.

### 10.10 What the integrity audit did and did not establish

The notebook reported **10/10 PASS** for:

| Test key in notebook | Result | Precisely what was checked |
|---|---|---|
| `all_users_are_cohort` | PASS | Every generated UserID belongs to the cohort mapping. |
| `unique_user_cutoff` | PASS | No duplicate `(UserID, cutoff_tw)` rows. |
| `six_binary_labels` | PASS | Evaluated partition label values are 0 or 1. |
| `no_null_labels` | PASS | Evaluated partition labels are not null. |
| `holdout_labels_sealed` | PASS | Holdout event labels are null with `EVALUATE_HOLDOUT=False`. |
| `chronological_splits` | PASS | Training label windows end before validation starts; validation ends before holdout. |
| `all_label_windows_mature_by_buffer` | PASS | Cutoff+horizon+configured 7-day buffer precedes latest observed timestamp. |
| `feature_sources_use_seen_time` | PASS | Configured feature-time fields are official row-time candidates (`first_seen_at` or `created_at`). |
| `no_future_known_columns_read` | PASS | Selected columns exclude `merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`. |
| `past_only_last_week_flags` | PASS | Last-week flags exactly match sets of rows with `seen_utc` in `[t−7d, t)`. |

**Limits:** The audit is necessary, but cannot prove historical information completeness, perfect activity semantics, point-in-time safety of **future unimplemented feature code**, correct exact test eligibility selection, or that a seven-day buffer makes all labels mature. The label equality issue was addressed separately in EXP-004 on one instructor example.

### 10.11 EDA-03 decisions, outstanding limits, and handoff

**Accepted:** official scorer integrated, 90-day history and 7-day horizon modeled explicitly, event-time and seen-time candidate labels audited, chronological split implemented, holdout sealed, cohort IDs stable within snapshot, non-trained heuristic benchmark recorded. Use the results as a **diagnostic baseline only** pending calibrated labeling policy.

**Must revisit:** historical labels generated before EXP-004 did **not** apply a record-observation `label_as_of` cutoff. The Sept 27 sample has now been used for label-policy calibration, overlapping a proposed Sept historical holdout region; **do not describe all September calendar weeks as untouched independently validated**. Final training/validation/holdout policy should be updated before fitting a learned model.

---

## 11. EXP-004: Official example-label reconstruction and observation-cutoff reconciliation

**Experiment ID:** `EXP-004` / `EXP-LABEL-01` (completed on the **official example only**)  
**Date:** 2026-10-14 Asia/Taipei  
**Input:** instructor `assignment2/evaluation/examples/sample_truth.csv`, saved executed Notebook 03 outputs plus successive diagnostic tables supplied in this conversation.  
**Users:** **4,785**.  
**Targets:** six binary labels per user, **28,710 user-target binary decisions**.  
**Dataset:** same course SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`.  
**Score implementation:** `official.scores()` and `official.f1_per_event()` loaded from instructor `evaluation/evaluate.py`; confusion matrices are direct comparisons against the official truth.  
**Status:** **Exact reconstruction (28,710 / 28,710) achieved for one calibrated example week and a candidate Oct 9 observation cutoff.** This is not prospective predictive validation.  
**Data restrictions respected:** no extra GitHub API, scraping, external GH Archive, or unsupported data collection.

### 11.1 Initial question and ground-truth profile

The objective was to test whether candidate binary historical activity labels could be reconstructed from the allowed relational tables to match instructor-provided answers, and to distinguish errors caused by event-time semantics, prediction-week boundaries, late discovery, or the snapshot used to generate official example truth.

Official example is described in `evaluation/README.md` as the **week of 9/27**, without a precise clock-time boundary in that text. The sample file contains exactly **4,785** users and header `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`.

| Target | Ground-truth positive users | Ground-truth negative users | Positive prevalence |
|---|---:|---:|---:|
| `open_pr` | 592 | 4,193 | 12.372% |
| `open_issue` | 333 | 4,452 | 6.959% |
| `issue_comment` | 450 | 4,335 | 9.404% |
| `pr_review` | 423 | 4,362 | 8.840% |
| `pr_review_comment` | 304 | 4,481 | 6.353% |
| `push` | 2,362 | 2,423 | 49.363% |
| **Total binary entries** | **4,464** | **24,246** | **15.548% across all cells** |

All user IDs refer to **GitHub** `Users.github_user_id`, not an internal mutable `Users.user_id` index. No sample-truth labels are available for the official November window at submission time.

### 11.2 Step A — Initial four-way comparison (two midnight boundaries × two time definitions)

The first diagnostic tested the seven-day window starting **Sept 27 at 00:00 Taiwan** versus **Sept 27 at 00:00 UTC**, each with `event_time` and `seen_time` labels, without an additional entity-observation cutoff.

| Window start | Candidate label timestamp | Macro-F1 | Micro-F1 | Combined F1 | Total mismatches |
|---|---|---:|---:|---:|---:|
| Sep 27 **00:00 Taiwan** | Event time | 0.963868 | 0.961534 | **0.962701** | **343** |
| Sep 27 **00:00 Taiwan** | First seen | 0.952420 | 0.955987 | **0.954204** | **394** |
| Sep 27 **00:00 UTC** | Event time | 0.978851 | 0.978938 | **0.978894** | **188** |
| Sep 27 **00:00 UTC** | First seen | 0.966765 | 0.972991 | **0.969878** | **242** |

**The full per-target mismatch counts from the saved executed notebook** (fixed source of truth; row times/actors equal aside from label timestamp):

| Boundary / variant | `open_pr` | `open_issue` | `issue_comment` | `pr_review` | `pr_review_comment` | `push` | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taiwan 00 / event | 48 | 26 | 36 | 19 | 19 | 195 | **343** |
| Taiwan 00 / seen | 54 | 70 | 36 | 20 | 19 | 195 | **394** |
| UTC 00 / event | 33 | 20 | 18 | 11 | 9 | 97 | **188** |
| UTC 00 / seen | 40 | 66 | 18 | 12 | 9 | 97 | **242** |

The initial UTC-midnight event-time candidate had the strongest combined agreement of those four, **but still made 188 binary decisions different from truth**. This was the starting point for focused boundary reconciliation.

#### Confusion matrix for initial 00:00 UTC + event-time candidate

| Target | TP | FP | FN | TN | Disagreements | Reconstructed positives | Official positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| `open_pr` | 565 | 6 | 27 | 4,187 | 33 | 571 | 592 |
| `open_issue` | 322 | 9 | 11 | 4,443 | 20 | 331 | 333 |
| `issue_comment` | 440 | 8 | 10 | 4,327 | 18 | 448 | 450 |
| `pr_review` | 415 | 3 | 8 | 4,359 | 11 | 418 | 423 |
| `pr_review_comment` | 298 | 3 | 6 | 4,478 | 9 | 301 | 304 |
| `push` | 2,329 | 64 | 33 | 2,359 | **97** | 2,393 | 2,362 |
| **Sum** | **4,369** | **93** | **95** | **24,153** | **188** | **4,462** | **4,464** |

Interpretation at that point: promising event-time semantics but possible clock-boundary misalignment. The sample truth was not yet proven to use UTC midnight. The 97 push mismatches became the diagnostic lever.

### 11.3 Step B — Push mismatch inspection and 13-hour window shift

Initial push confusion: **64 false positives, 33 false negatives**. Inspection of mismatching users showed pushes just after Sept 27 UTC midnight and just after Oct 4 UTC midnight. For example:

- `UserID 9306`, official push **0**, reconstructed **1**, pushes **2026-09-27 12:38:31 UTC** and **12:39:59 UTC**. These precede **13:00 UTC**, so they disappear when the candidate window starts then.
- `UserID 377938`, official push **0**, reconstructed **1**, event **2026-09-27 07:09:07 UTC**, also before a 13:00 UTC start.
- Error-position diagnostic for the wrong UTC-midnight week: **all 64 false-positive users had zero push events in the middle five days or last 24 hours**; false negatives had zero recorded pushes anywhere inside that wrongly aligned week. The output showed 102 push events from false-positive users in the first 24 hours, and 66 push events for false-negative users in the 24 hours just after the end. These are **event counts, not numbers of distinct users**.

A diagnostic sweep moved the entire seven-day window by each integer hour **−24 to +24** relative to Sept 27 **00:00 UTC**:

| Relative start shift | Push TP | Push FP | Push FN | Disagreements | Push F1 |
|---:|---:|---:|---:|---:|---:|
| **+13 hours** | **2,362** | **0** | **0** | **0** | **1.000000** |
| +14 hours | 2,359 | 2 | 3 | 5 | 0.998941 |
| +12 hours | 2,356 | 6 | 6 | 12 | 0.997460 |
| +15 hours | 2,346 | 6 | 16 | 22 | 0.995333 |
| +11 hours | 2,352 | 13 | 10 | 23 | 0.995134 |
| +10 hours | 2,349 | 17 | 13 | 30 | 0.993655 |
| +16 hours | 2,342 | 11 | 20 | 31 | 0.993425 |
| +9 hours | 2,347 | 24 | 15 | 39 | 0.991760 |
| +17 hours | 2,336 | 15 | 26 | 41 | 0.991301 |
| +8 hours | 2,346 | 31 | 16 | 47 | 0.990082 |

**Finding:** For sample-truth `push`, the calibrated example window is exactly reproduced by:

```text
Event-time start = 2026-09-27 13:00:00 UTC = 2026-09-27 21:00 Taiwan
Event-time end   = 2026-10-04 13:00:00 UTC = 2026-10-04 21:00 Taiwan
Interval         = [start, end) = exactly seven days.
```

The 21:00 Taiwan alignment coincides with the announced **daily repository release time**, but **the causal reason for choosing that boundary remains inferred**, not documented explicitly in the example README. The selected boundary was found by examining the example answers, so using this same example afterward is **in-sample calibration**.

**Technical caveat:** Notebook 03 defines `Pushes.event_utc` and `Pushes.seen_utc` using the same `Pushes.created_at`; equality of these two columns does **not** prove an independent observation/ingestion timestamp.

### 11.4 Step C — Recompare all six targets at the calibrated 21:00 Taiwan week boundary

Using `start=2026-09-27 13:00 UTC`, `end=2026-10-04 13:00 UTC`, and **event-time** labels without an observation cutoff:

| Target | F1 | Ground truth positives | Reconstructed positives | Mismatches | Nature |
|---|---:|---:|---:|---:|---|
| `open_pr` | **1.000000** | 592 | 592 | **0** | Exact |
| `open_issue` | **0.997006** | 333 | 335 | **2** | 2 false positives, 0 false negatives |
| `issue_comment` | **1.000000** | 450 | 450 | **0** | Exact |
| `pr_review` | **1.000000** | 423 | 423 | **0** | Exact |
| `pr_review_comment` | **1.000000** | 304 | 304 | **0** | Exact |
| `push` | **1.000000** | 2,362 | 2,362 | **0** | Exact |

Overall: **2** mismatches / **28,710** cells; macro **0.999501**, micro **0.999776**, combined **0.999639**; binary agreement **28,708/28,710**. The only remaining issue is `open_issue`, not activity-window alignment for the other five targets.

For comparison, the same **21:00 Taiwan** window with `seen_time` labels gave **58** mismatches, combined **0.990226**:

| Seen-time target | Mismatches | Reconstructed positives |
|---|---:|---:|
| `open_pr` | 7 | 599 |
| `open_issue` | 50 | 367 |
| `issue_comment` | 0 | 450 |
| `pr_review` | 1 | 424 |
| `pr_review_comment` | 0 | 304 |
| `push` | 0 | 2,362 |

**Decision:** Use original event timestamps as the provisional label definition when analyzing this sample. Continue to use first-observation timestamps for **feature visibility**, not as substitutes for issue/PR opening time.

### 11.5 Step D — Identify the exact two `open_issue` false positives

The user-level mismatch list was:

| GitHub UserID | Instructor `open_issue` | Reconstructed without as-of filter | Error |
|---|---:|---:|---|
| **101449496** | 0 | 1 | False positive |
| **112989862** | 0 | 1 | False positive |

Original `Issues.parquet` record inspection mapped actors through the current snapshot's `Users` and selected **non-PR issues created inside the example window**:

| GitHub UserID | Stable `github_issue_id` | Internal `repo_id` (snapshot-local!) | Repo issue number | Issue `created_at` UTC | Issue `first_seen_at` UTC | Observation delay | First seen after window? | `created_via_app` |
|---|---:|---:|---:|---|---|---:|---|---|
| 101449496 | **5608729534** | 102827 | 241 | **2026-09-28 03:08:24** | **2026-10-10 05:11:20** | **290.048889 h** (~12.09 days) | Yes | Null |
| 112989862 | **5619458654** | 37821 | 344 | **2026-09-28 18:38:47** | **2026-10-09 17:36:51** | **262.967778 h** (~10.96 days) | Yes | Null |

The **event creation time** for both lies within `[Sep 27 13:00, Oct 4 13:00)` UTC, but the records were **not observed in this dataset until Oct 9 or Oct 10**. A reconstruction from the later October 14 snapshot therefore counts them as positive retrospectively; a snapshot taken before their first observation would not contain them.

**Established:** delayed observation explains why those records can be counted by event time in today's snapshot but may have been absent when the instructor made the example truth. **Not established:** that the TA intentionally excluded the issues, or the exact process/time at which sample labels were frozen.

### 11.6 Step E — Observation cutoff sensitivity: prediction-week end versus later

A further diagnostic constructed event-time labels inside the calibrated example window, but retained only entities for which `seen_utc <= label_as_of` (as implemented by existing Notebook 03 event streams). Results:

| Candidate label observation cutoff | Total FP | Total FN | Total mismatches | Macro-F1 | Micro-F1 | Combined F1 |
|---|---:|---:|---:|---:|---:|---:|
| Example week end, **Oct 4 13:00 UTC** | 0 | 8 | **8** | 0.997974 | 0.999103 | **0.998538** |
| Week end + 7 days, **Oct 11 13:00 UTC** | 2 | 0 | **2** | 0.999501 | 0.999776 | **0.999639** |
| Week end + 14 days, **Oct 18 13:00 UTC** | 2 | 0 | **2** | 0.999501 | 0.999776 | **0.999639** |
| No observation cutoff (full current snapshot) | 2 | 0 | **2** | 0.999501 | 0.999776 | **0.999639** |

Every mismatch remained on `open_issue`; all other targets were exact under these tested cutoffs. **A simple “labels visible by prediction-week end” rule does not reproduce truth**: it loses eight official positive issue creators who became visible after week end. Conversely, a seven-day lag after week end includes those eight **and** the two later false positives.

This experiment established the need to distinguish **event window end** from **historical label-as-of**; it did not yet identify the exact label-as-of cutoff.

### 11.7 Step F — Separating interval for all ten delayed issue users

Select `Issues` with creation time inside the example week, first observed after week end but by week end +7 days, and whose author appears in sample truth. The ten newly visible user records have the following earliest first-observed timestamps and official labels:

| UserID | First observed UTC | Issue rows | Official `open_issue` |
|---:|---|---:|---:|
| 104238077 | 2026-10-04 23:03:10 | 1 | 1 |
| 108928239 | 2026-10-05 03:49:25 | 1 | 1 |
| 21228832 | 2026-10-06 18:19:56 | 1 | 1 |
| 121954209 | 2026-10-07 20:09:13 | 1 | 1 |
| 13179623 | 2026-10-08 00:57:24 | 1 | 1 |
| 40018640 | 2026-10-08 08:30:16 | 1 | 1 |
| 17217484 | 2026-10-08 09:42:07 | 1 | 1 |
| 117448747 | **2026-10-09 05:46:50** | 1 | **1** (last of eight positives) |
| 112989862 | **2026-10-09 17:36:51** | 1 | **0** (first of two negatives) |
| 101449496 | 2026-10-10 05:11:20 | 1 | 0 |

A **single separating cutoff** is possible because:

```text
Last desired positive first-seen = 2026-10-09 05:46:50 UTC
First undesired negative first-seen = 2026-10-09 17:36:51 UTC

Any s satisfying:
    2026-10-09 05:46:50 UTC <= s < 2026-10-09 17:36:51 UTC
includes all eight positive delayed users and excludes both negative delayed users.
```

This **~11h50m separation interval** contains both **Oct 9 12:00 UTC (20:00 Taiwan)** and **Oct 9 13:00 UTC (21:00 Taiwan)**, the instructor's reported data through-time and scheduled publication time respectively. Therefore, an **October 9 release/snapshot hypothesis** is especially plausible, but the observed separation alone does **not** identify exact publication/processing logic.

### 11.8 Step G — Final adjacent-cutoff verification across all six targets

Using the **event-time** target rule and fixed calibrated example week start **Sept 27 13:00 UTC**, the following four `label_as_of` cutoffs were tested in the user-supplied execution outputs:

| Candidate observation cutoff | Macro-F1 | Micro-F1 | Combined F1 | Total mismatches | `open_pr` | `open_issue` | `issue_comment` | `pr_review` | `pr_review_comment` | `push` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Oct 8 **20:00 Taiwan** (Oct 8 12:00 UTC) | 0.999749 | 0.999888 | 0.999819 | **1** | 0 | 1 | 0 | 0 | 0 | 0 |
| **Oct 9 20:00 Taiwan** (Oct 9 12:00 UTC) | **1.000000** | **1.000000** | **1.000000** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| **Oct 9 21:00 Taiwan** (Oct 9 13:00 UTC) | **1.000000** | **1.000000** | **1.000000** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| Oct 10 **20:00 Taiwan** (Oct 10 12:00 UTC) | 0.999501 | 0.999776 | 0.999639 | **2** | 0 | 2 | 0 | 0 | 0 | 0 |

**Measured final result:** With either of the two Oct 9 observation cutoffs, **every binary example label agrees exactly**:

```text
Users:                           4,785
Targets per user:                6
Total user–target comparisons:  28,710
Matching labels:                28,710
Disagreements (FP + FN):        0
Macro-F1 (official):            1.000000
Micro-F1 (official):            1.000000
Combined score (official):      1.000000
```

The Oct 8 test misses one issue-positive user because their row was not yet observed, whereas the Oct 10 test admits both false-positive issues. This produces the 1 → 0 → 0 → 2 mismatch sequence, exactly as the late-observation hypothesis predicts for the printed records.

**Conservative wording:** “We achieved exact reconstruction of the **supplied example truth** using a calibrated Sept 27 21:00 Taiwan event window and an Oct 9 as-of cutoff on the supplied snapshot.” **Do not write:** “We proved the TA used Oct 9 20:00” or “our model achieved perfect F1.”

### 11.9 Reconciliation of inference, scope and limitations

**Verified numerical observations**

1. `sample_truth.csv` contains 4,785 distinct users, 6 targets, 28,710 decision cells.
2. At Sept 27 00:00 UTC event-time boundary: 188 mismatches, 0.978894 combined agreement.
3. Moving the event window **+13 hours** yields exactly correct `push` labels (2,362 TP, 0 FP, 0 FN).
4. The same 21:00 Taiwan window yields correct labels for five targets, with two late-observed `open_issue` false positives.
5. Those two issue rows were created during Sept 27–Oct 4 but first observed Oct 9 17:36:51 and Oct 10 05:11:20 UTC.
6. The eight late-observed official issue positives were all first seen earlier than either of the two false-positive issues.
7. Both **Oct 9 20:00** and **Oct 9 21:00 Taiwan** as-of cutoffs give **zero mismatches for all six activities**.
8. Scoring was computed with the instructor's existing evaluator; this was **not** a trained model exercise.

**Interpretations strongly consistent with the evidence, but unconfirmed**

- Sample truth likely reflects the information available in a dataset release around **Oct 9**, potentially the scheduled 21:00 release containing records through 20:00.
- Example weekly boundary likely tracks the instructor's release cutoff time **21:00 Taiwan**, not an arbitrary UTC-midnight week.
- Delayed record discovery plausibly explains the two false positives; no need to posit malformed author mapping based on these examples.

**What this study does not prove**

- The actual code/process used to generate the instructor's labels; the cutoff was inferred by calibrating **against those labels**.
- Whether 20:00 or 21:00 was the actual cutoff, since both are indistinguishable here.
- Whether every later historical week has identical event/observation semantics or similar discovery delays.
- Whether future November truth is measured with the same number of days between target-window end and label freeze.
- Full re-creation of past **direct-event ingestion snapshots**, because Notebook 03 has no independent observed-time field for the three direct-event sources.
- That all six labels can be reconstructed from *every historical snapshot* with 100% accuracy.
- That a predictive model can achieve 1.0 F1, or that EDA-03's heuristic 0.3504 validation mean is improved by this equality result.
- That the September historical weeks are fully untouched after using sample-truth answers for label-method and time-cutoff selection.

### 11.10 A calibrated label specification for the *example*, not a universal frozen rule

For an example user `u`, target `k`, start `t`, end `t+7d`, and independently meaningful row observation `s`, an **example-reconstructing** rule is:

```text
positive(user=u, target=k, start=t, end=t+7d, as_of=s)
    = exists row of source table for k where:
        actor maps to GitHub UserID u;
        original event timestamp in [t, t+7d);
        for open_issue: is_pull_request == False;
        row-visibility time <= s (when this timestamp exists separately);
        historical actor and primary/FK relations are interpreted within this one snapshot.
```

For the validated sample:

```python
sample_week_start_utc = '2026-09-27T13:00:00Z'
sample_week_end_utc   = '2026-10-04T13:00:00Z'
example_label_as_of_utc = '2026-10-09T12:00:00Z' # 20:00 Taiwan; 13:00Z also matches
```

Mathematically, for an applicable independent visibility time:

\[
y_{u,k}(t,s) = \mathbb{1}\{\exists e:\; a(e)=u,\; k(e)=k,\; t\le\tau_{event}(e)<t+7d,\; \tau_{visible}(e)\le s\}.
\]

**Critical design boundary:** These are **retrospective labels**. Model features at prediction cutoff `t` must continue to use only information visible **strictly before `t`** (`seen_utc < t`) and must never access `s` or any later rows. The official November prediction window remains **Nov 12 00:00 → Nov 19 00:00 Taiwan**; use its exact specified boundary, not the sample week boundary.

### 11.11 Holdout contamination and experiment independence

Notebook 03 originally reserved four September Monday weeks (Sep 7, 14, 21, 28 starts) as a historical holdout. Setting `EVALUATE_HOLDOUT=False` prevented printing or building those future labels in the notebook. **However**, EXP-004 read the instructor's answer file for a seven-day window covering Sep 27–Oct 4, which overlaps that nominal holdout period.

Thus the overlapping September period **cannot be described as a pristine independent test of whichever label definition/cutoff was selected using sample truth**. Future work should distinguish the following:

- **Calibration:** official example truth for Sep 27–Oct 4, used to define/reconcile label semantics. Clearly tag this as in-sample calibration for the label generator.
- **Model validation:** August calendar weeks or other windows chosen for model threshold/hyperparameter selection, distinct from example truth when possible.
- **Untouched model holdout:** a new later set of mature weeks (potentially after more daily releases) that has **never** influenced labeling-rule or model-selection decisions; evaluate only once after freezing the pipeline.

The historical EDA-03 results should be preserved as-is. Record any future replacement splits and revised baseline scores under a new experiment identifier rather than overwriting **0.3504**.

### 11.12 Final EXP-004 conclusion and decision

**Decision:** Treat the six-table **event-time label definition** as the best-supported provisional labeling approach and implement **separate configurable `prediction_cutoff`, `label_window_end` and `label_as_of` parameters**, with explicitly documented source-specific visibility capabilities and left-closed/right-open intervals. Do not globally set `label_as_of = window_end+5days` simply because an Oct 9 cutoff matches this one calibrated example.

**Status:** `EXP-004 — COMPLETE (exact sample-truth reconstruction; generalization and actual TA implementation not established)`.

---

## 12. Updated decisions, open questions and implementation checklist

### 12.1 Decision register after EXP-004 (historical; superseded in part by EXP-005/006 §14–15)

| ID | Current state | Decision and supporting evidence | Verification still needed |
|---|---|---|---|
| D-01 | Accepted | Personal source code lives **outside** replaceable official `assignment2/` clone; teacher updates via force-replacement. | Confirm `.gitignore` and remotes locally. |
| D-02 | Accepted | Version every experiment by full **instructor SHA**, personal code SHA and exact cutoff/config. | Store personal code commit in each future run. |
| D-03 | Accepted | `github_user_id` stable for export and cross-release linkage; row-index FK joins only inside one release. | Test after a distinct new SHA arrives. |
| D-04 | Accepted | Features only from official row times `< prediction_cutoff`; mask six late-known fields. | Unit tests of all future engineered feature columns. |
| D-05 | Accepted for baseline | Start with safe fields; exclude late-known fields until availability logic is audited. | Revisit one field at a time if measurable benefit. |
| D-06 | Benchmarked, not fitted | Previous-week heuristic is a **measured** baseline: validation eligible-proxy mean **0.3504** under EDA-03 config; no learned model yet. | Rerun after label/split policy frozen. |
| D-07 | **Provisional label policy selected** | For retrospective activity labels, use original event timestamps and actor attribution; use non-PR filter for issue creation. EXP-004 exactly matches sample truth with calibrated clock/as-of policy. | Validate on an independent later week/snapshot if available. |
| D-08 | Partially completed | Historical calendar exists: 22/5/4 weeks, 7-day horizon, 7-day buffer, Monday 00:00 Taiwan. | Replace overlapping September holdout; justify label as-of policy and historical completeness. |
| D-09 | Pending | Choose time horizons, recency weights and class thresholds. | Time-based validation/ablations. |
| D-10 | Pending | Evaluate simple relational aggregates before any GNN. | Controlled incremental performance test. |
| D-11 | Pending | Use exact Oct 28 test user IDs once published. | Official roster not available in current snapshot. |
| D-12 | Pending | Identify whether summer collection/behavior differences cause distribution change. | Equal-length window/record coverage study. |
| D-13 | **Completed for supplied example** | `sample_truth.csv`: 28,710/28,710 reproduced for calibrated week and Oct 9 as-of cutoff, official scores all 1.0. | Do not mislabel as prospective model score. |
| D-14 | Required for next pipeline | Distinguish target-event occurrence time from feature row visibility and label-as-of observation time. | Implement tested `src/temporal_dataset.py`. |
| D-15 | Not yet verified | Whether TA used exactly Oct 9 release at 20:00/21:00; both agree with sample. | TA clarification or original generator, if permitted. |

### 12.2 Proposed follow-up experiments with precise acceptance criteria

**EXP-005 — Reusable temporal dataset generator.** Implement `src/temporal_dataset.py` (plus tests) that (a) reads the eight Parquet tables from a replaceable local official clone; (b) maps stable GitHub IDs; (c) represents `prediction_cutoff`, horizon and `label_as_of` explicitly and independently; (d) generates train/validation labels from original event times, with source-specific observation filtering; (e) builds **features strictly from visible records** and excludes late-known fields; (f) creates non-overlapping temporal splits and safely encodes unknown/missing labels. Add a regression test on the calibrated example with **all 28,710 labels identical** when running against instructor SHA `506d860...`. Test on future refreshed SHAs separately rather than hardcoding row numbers.

**EXP-006 — Stable baselines under updated split policy.** Recompute all-zero and prior-week occurrence heuristic under the selected label maturity and fresh independent holdout. Use identical cutoffs/population/label rules when comparing model families; record per-target TP/FP/FN, precision, recall, F1, all-cohort and eligible-proxy slices, weekly and pooled official metrics, and variance. Preserve historical EDA-03 result **0.3504** explicitly as a different protocol.

**EXP-007 — First learned model.** Fit transparent per-target logistic regression and/or gradient-boosted trees to 7/30/90-day observable counts/recency, week-frequency, trend, activity diversity, etc. Tune per-label thresholds on validation only using the **combined macro/micro** objective. Record seed, features, training dates and exact official evaluator metrics. Do not assume a higher macro-F1 if rare classes are sparsely labeled.

**EXP-008 — Relational ablations.** Test incremental value of distinct repos, recent repo popularity and collaborators, controlling for raw activity frequency. Start with regular aggregation, then consider RDL/GNN only if justified by baseline results and assignment time budget.

**EXP-009 — Release-to-release drift / retrospective label revision.** Capture each distinct instructor SHA after 21:00 Taiwan; track schema, record counts, stable GitHub keys and any new entities created historically but first seen later. Re-evaluate whether prior labels changed and whether features remain point-in-time valid. Never mix internal FK indices across releases.

**EXP-010 — Independent label-policy check.** If a truly untouched, later instructor truth or permitted official evidence becomes available, check source-specific label reconstruction without tuning boundary/as-of values to that new sample. Distinguish exact example-matching from generalizable label semantics.

### 12.3 Required automated tests for the data generator

1. Correct six-target source column/actor/filter mapping (including `Issues.is_pull_request=False`).
2. Stable `github_user_id` joining and uniqueness; no internal-row-index cross-release comparison.
3. `[cutoff, cutoff+7d)` membership; timezone-naive Parquet UTC converted safely to aware UTC; no off-by-one at 00:00 or 21:00.
4. Independent `prediction_cutoff`, `label_window_end` and `label_as_of` handling.
5. No data observed at or after the feature cutoff is used in features; no late-known state fields leaking.
6. For entity-derived labels, use original event time for week membership and independently meaningful first-observation time for as-of reconstruction.
7. For direct-event tables, do **not** pretend `created_at` is a separate ingestion timestamp; document limitations.
8. Maturation/completeness check does not overstate a heuristic latest-observed-timestamp watermark.
9. No overlapping train-label and validation cutoffs; no user-week duplicates; complete expected sample row set.
10. No leakage from official example-truth answers into actual model features or untouched test evaluation.
11. Official evaluator imported and zero-positive-class behavior tested.
12. Regression fixture: 4,785 users × six targets, **zero differences** on the calibrated sample against the stated pinned instructor SHA.
13. Failing tests if schema/target fields changed in a new daily dataset release, rather than silently relabeling.
14. Optional exports require explicit enablement, verify correct CSV header, binary labels and no extra/missing `UserID`s.
15. Track if changing an observation cutoff causes historical labels to be added; never silently reuse cached labels from another snapshot.

### 12.4 Statements suitable and unsuitable for the assignment presentation

**Evidence-supported wording:**

- “EDA revealed strong class imbalance and a pronounced mid-year change in recorded GitHub activity; the cause remains unknown.”
- “A simple previous-week heuristic obtained mean weekly combined F1 **0.3504** on our August validation cutoffs over proxy-eligible users, using retrospectively reconstructed labels and the official evaluator.”
- “We diagnosed timing discrepancies and exactly matched the 4,785-user provided example labels after calibrating the example week window and label observation cutoff.”
- “Timestamp availability was explicitly separated from target event time in our engineering plan.”

**Unsupported wording to avoid:**

- “Our predictive model achieved 100% F1.” (No model was trained; the 1.0 score is label reconstruction.)
- “The teacher definitely generated example truth on October 9 at 21:00.” (Oct 9 as-of hypothesis fits, but precise pipeline not verified and 20:00 fits too.)
- “We proved that 7 days is the correct maturation interval for all weeks.” (Two later-discovered sample issues contradict naive completeness assumptions.)
- “A GNN improved results” / “relational data causes higher push probability.” (Neither tested.)
- “September historical holdout remains perfectly independent.” (Sample truth used for label calibration overlaps that period.)
- “The official November prediction period starts at 21:00.” (Incorrect; it starts at **00:00 Taiwan** Nov 12.)

---

## 13. Reproducibility addendum and evidence ledger

### 13.1 Where the underlying measurements came from

| Artifact | Evidence source | Used for |
|---|---|---|
| `notebooks/01_data_audit.ipynb` | Executed `01_data_audit(1).ipynb` and original log §§4/9 | EDA-01 row counts, integrity, missingness, observation lag, historical volume. |
| `notebooks/02_deeper_data_analysis.ipynb` | Executed `02_deeper_data_analysis(1).ipynb` and original log §§5/9 | EDA-02 period regimes, turnover, persistence, eligibility and relational breadth. |
| `notebooks/03_label_and_split_validation.ipynb` | Executed `03_label_and_split_validation(5).ipynb` | EDA-03 exact configurations, split counts, candidate-disagreement table, prevalence, baseline F1, integrity results and initial optional sample comparison. |
| Push error inspection | User-supplied diagnostic output in this conversation | EXP-004 push FP/FN, early/late event concentration. |
| Shift sweep | User-supplied `shift_hours` diagnostic table | EXP-004 +13 hour perfect push-label boundary. |
| Six-target comparison | User-supplied combined and per-label official evaluator results | EXP-004 five targets exact, two issue false positives. |
| Two issue records | User-supplied issue-level diagnostic table | EXP-004 stable issue IDs, first-seen delays >10 days. |
| Cutoff sensitivity | User-supplied `Week end / +7 / +14 / none` table | EXP-004 need for a separate snapshot-as-of time. |
| Newly visible issue users | User-supplied ten-user first-observed table and separating interval | EXP-004 Oct 9 cutoff hypothesis. |
| Final four-cutoff comparison | User-supplied `Oct 8/9/10` table | EXP-004 zero mismatches / F1 1.0 on Oct 9. |
| `assignment2/evaluation/evaluate.py` | Instructor supplied code, imported during notebook run | Authoritative F1 and strict CSV format rules. |
| `assignment2/data/README.md`, `db/SCHEMA.md` | Instructor supplied documentation | Row-time definitions, source actor and stable-ID semantics. |

**Evidence limitation:** The later EXP-004 diagnostics were supplied as console/notebook outputs in this chat. Do not claim they are all saved within `03_label_and_split_validation(5).ipynb` unless the notebook is rerun with those extra cells and saved. For a durable repository, insert those diagnostic cells in a numbered appendix notebook or export their summary outputs to an audited experiment artifact, while retaining this log.

### 13.2 Official sample-label reproduction recipe (diagnostic only)

Use the already loaded `events` and `official` objects from Notebook 03, not an external data source:

```python
import pandas as pd

truth = pd.read_csv(OFFICIAL_REPO / 'evaluation/examples/sample_truth.csv')
start = pd.Timestamp('2026-09-27T13:00:00Z')
end = pd.Timestamp('2026-10-04T13:00:00Z')
label_as_of = pd.Timestamp('2026-10-09T12:00:00Z')

pred = truth[['UserID']].copy()
for target in official.EVENT_COLS:
    e = events[target]
    mask = (
        e['event_utc'].ge(start)
        & e['event_utc'].lt(end)
        & e['seen_utc'].le(label_as_of)
    )
    pred[target] = pred['UserID'].isin(set(e.loc[mask, 'UserID'])).astype(int)

assert all((pred[k] == truth[k]).all() for k in official.EVENT_COLS)
assert official.scores(pred, truth)['score'] == 1.0
```

This reproduces the **reported calibrated example agreement** if run on the **same instructor SHA and notebook data mappings**. It is not a substitute for prospective model training, nor proof of identical historical direct-event ingestion under alternate snapshots.

### 13.3 Reproducible future submission-time checklist

- Keep student code in the personal Git repository; re-clone the official `assignment2/` folder after needed daily 21:00 releases rather than `git pull` on the force-replaced history.
- Record instructor SHA and project code SHA, data source time coverage and timestamps in UTC and Taiwan.
- Confirm six actor mappings, non-PR issue filter, row visibility and event-time definitions after every schema change.
- Preserve a truly independent historical holdout distinct from Sep 27–Oct 4 example calibration; resist repeatedly tuning against it.
- Freeze feature and label policies before holdout scoring; record per-label and macro/micro/combined scores, clearly indicating mean-weekly versus pooled metrics.
- On final release Nov 11 21:00 Taiwan, rerun the already-validated ingestion/feature/training/prediction pipeline. Only **2 h 55 min** remain until the **Nov 11 23:55** deadline.
- Final future label horizon is **Nov 12 00:00–Nov 19 00:00 Taiwan**, not the release-aligned historical example week.
- Validate final submission CSV against the official `evaluation/evaluate.py`, with exact `UserID` set, header and binary values; do not claim to know withheld future ground truth.

### 13.4 EXP-004 concise archive record for future agents

```text
EXP-004 / 2026-10-14 / Official sample-truth reconciliation
Course SHA: 506d860693df60a698b68aa63a8edb4a984c8e1b
Example: 4,785 GitHub users; 6 binary targets = 28,710 cells
Window: [2026-09-27T13:00:00Z, 2026-10-04T13:00:00Z)
Label actor+timestamp: cohort UserID + original created_at/submitted_at;
                       open_issue excludes PR-conversation issues
Historical label observation cutoff: 2026-10-09T12:00:00Z OR
                                     2026-10-09T13:00:00Z (both match)
Official evaluator: assignment2/evaluation/evaluate.py
Exact sample matches: 28,710/28,710
Macro F1 = Micro F1 = Combined F1 = 1.000000
One-sample calibration, not prospective prediction performance
Direct-event observed-time is not independently measured in the snapshot
Original previous-week heuristic validation combined mean F1 = 0.3504
  (EDA-03 separate task, 5 August weekly cutoffs, eligible-proxy users)
Holdout: September nominal holdout overlaps sample label calibration;
         reserve a NEW genuinely independent later holdout
Next: src/temporal_dataset.py + regression tests + recalc baseline
```

**End of the EDA-03 and EXP-004 archive as it existed before pipeline and baseline implementation.** The appended §§14–15 below document later user-executed EXP-005 and EXP-006 evidence. Do not confuse EXP-004's calibrated reconstruction score with EXP-006's historical predictive F1.

---

## 14. EXP-005: Reusable temporal dataset generator, real execution

**Experiment ID:** `EXP-005`  
**Status:** **Completed** for the defined dataset-generation milestone; later release drift and feature-level QA are separate follow-ups.  
**Date (Asia/Taipei):** 2026-10-14.  
**Provenance:** User-supplied local Windows PowerShell/Conda output in the current conversation and the subsequently pasted `manifest.json` output; not an independent execution within this document.  
**Official instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`.

### 14.1 Implemented source and invocation

The EXP-005 code supplied to the project defines `src/data_loader.py`, `src/labels.py`, `src/features.py`, `src/temporal_splits.py`, `src/build_dataset.py`, `configs/exp005.json`, and corresponding tests. Implementation uses only the eight permitted source tables, keeps `github_user_id` for stable identity, and distinguishes historical feature visibility from event occurrence and retrospective label-as-of filtering. Those properties were exercised by synthetic tests; this log does **not** claim an independent line-by-line security or feature-value audit of the Windows source files.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q --basetemp=.\.pytest_tmp
python -m src.build_dataset --mode sample-check --require-perfect
python -m src.build_dataset --mode build --config configs/exp005.json
```

### 14.2 Observed Windows environment and tests

- The first local test attempts failed during third-party pytest autoload (`platformdirs.pytest_plugin`) and then from an inaccessible default Windows pytest temporary directory (`WinError 5`); these were **environment setup errors before relevant assertions ran**, not confirmed source-code assertion failures.
- With plugin autoload disabled and `--basetemp=.\.pytest_tmp`, the user reported **14 passed, 1 skipped, 6.19 seconds**.
- The later combined EXP-005/EXP-006 test suite reported **25 passed, 1 skipped, 9.51 seconds**; see §15.2.

### 14.3 Official sample regression (actual local Parquet execution)

The user ran `python -m src.build_dataset --mode sample-check --require-perfect` and reported:

| Item | Actual result |
|---|---|
| Official sample users | **4,785** |
| Binary decisions | **28,710** |
| Week start UTC | `2026-09-27 13:00:00+00:00` |
| Week end UTC (exclusive) | `2026-10-04 13:00:00+00:00` |
| Label-as-of UTC | `2026-10-09 12:00:00+00:00` |
| Data SHA | `506d860693df60a698b68aa63a8edb4a984c8e1b` |
| Macro-F1 / micro-F1 / combined score | **1.000000 / 1.000000 / 1.000000** |
| Per-target mismatches | **0** for all six |

Per-target true positives (and zero false positives/negatives): `open_pr` **592**, `open_issue` **333**, `issue_comment` **450**, `pr_review` **423**, `pr_review_comment` **304**, `push` **2,362**. These are **label-reconstruction verification results on a calibrated example**, not future prediction results.

### 14.4 Dataset build outputs and split configuration

The successful real-data build returned this directory:

```text
experiments/datasets/exp005/build-506d860693df-6040f635d4/
```

| Generated file | Reported rows | Status |
|---|---:|---|
| `train_features.parquet` | 189,000 | Generated |
| `train_labels.parquet` | 189,000 | Generated |
| `validation_features.parquet` | 81,000 | Generated |
| `validation_labels.parquet` | 81,000 | Generated |
| `holdout_features.parquet` | 9,000 | Generated, **labels sealed** |
| `label_audit.csv` | 31 | One row per scheduled cutoff |

Parameters recorded in the user-supplied build manifest: first cutoff **Mar 2 2026 Taiwan**; validation starts **Aug 3**; holdout starts **Oct 12**; horizon **7 days**; label maturation **7 days**; `purge_boundary_maturity=true`; lookbacks **7, 30, 90 days**; eligible proxy **≥5 observation records in 90 days**. There were **21 train weeks, 9 validation weeks, 1 holdout week** and **9,000 cohort users per week**. Latest available observed row timestamp was **Oct 13, 12:59:29 UTC**. The manifest explicitly set `holdout_unsealed=false`.

**Limitations:** The label-as-of buffer is a reconstruction policy rather than proof of data completeness; direct-event rows lack independent ingest timestamps; cohort membership and eligible proxy are retrospective; this is one instructor SHA, not a release-drift study. Generated label-audit contents were viewed in PowerShell, but a full independent feature-value/missingness audit is not supplied. The October holdout has **no labels** in these outputs.

### 14.5 EXP-005 decision and carry-forward

EXP-005 met the specific exit criteria of **real Parquet loading, deterministic configured dataset exports, synthetic tests and exact calibrated example regression**. Status recorded as **Completed**; subsequent cross-release checks and deeper feature QA remain future tasks. EXP-006 uses its generated output as a stable input and must identify the source run folder and SHA separately.

---

## 15. EXP-006: Historical baseline modeling, first real results

**Experiment ID:** `EXP-006`  
**Status as of 2026-10-14:** **Ongoing — real training and comparative historical evaluation executed; per-target/threshold and evaluator-audit evidence not yet supplied.**  
**Experiment question:** Do basic learned feature-based classifiers outperform transparent zero-activity and previous-week persistence baselines under an explicitly chronological, calibration-separated history split?  
**Evidence basis:** User-pasted Windows PowerShell output of actual test invocation, dry run, two training executions, the printed JSON leaderboards, and a CSV leaderboard preview. More detailed `per_target_metrics.csv`, `summary.csv`, `manifest.json`, and calibrated thresholds were **not** attached. This document does not assert their unseen contents.  
**Instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`. **Student source commit SHA:** not reported.

### 15.1 Reproducible commands and source-run binding

Executed by the user in the active `nycu-bda-asg2` Conda environment on Windows, from the Assignment 2 project root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q --basetemp=./.pytest_tmp
python -m src.train_baselines --dry-run
python -m src.train_baselines --config configs/exp006.json --models zero,persistence,logistic
python -m src.train_baselines --config configs/exp006.json --models zero,persistence,logistic,hgb
```

The dry run selected the same real EXP-005 source directory documented in §14.4:

```text
experiments/datasets/exp005/build-506d860693df-6040f635d4/
```

No instructor-prohibited supplemental GitHub activity data were reported.

### 15.2 Tests, dry-run result, features and populations

**Test result:** **25 passed, 1 skipped, 9.51 seconds** for the combined project suite. One skipped test was not reported as a failure. The dry run succeeded and identified **59 input features** and three baseline models for the first training command (`zero`, `persistence`, `logistic`).

| Data segment | Weekly cutoffs | User-week rows | Purpose |
|---|---:|---:|---|
| Historical fit (`train`) | 21 | **189,000** | Fit learned models |
| Threshold calibration | 3 | **27,000** | Choose thresholds under calibration-only policy |
| Primary report | 3 | **27,000** | Report performance on later, non-overlapping weeks |
| Primary report / `eligible_proxy` subset | 3 | **16,982** | **Headline scoreboard population** |

All cohort users appear in each complete weekly panel; `eligible_proxy` is based on the historical ≥5-record/90-day approximation and is not the official Oct 28 test roster. The headline denominator is **16,982 user-week rows, not 16,982 distinct people**, and equals **6,030 + 5,656 + 5,296** eligible proxy user-weeks at the three primary cutoffs.

**Calibrated reporting schedule (Taiwan local Monday 00:00, UTC shown for exactness):**

| Role | Taiwan weekly cutoff(s) | UTC cutoff(s) | Comment |
|---|---|---|---|
| Fit | Mar 2 to Jul 20 (21 weeks) | First `2026-03-01T16:00:00Z` | Source EXP-005 training set |
| Threshold calibration | Aug 3, 10, 17 | Aug 2, 9, 16 at 16:00 UTC | 27,000 cohort user-weeks |
| Gap/excluded | Aug 24 | `2026-08-23T16:00:00Z` | Maturation/embargo separation |
| **Primary validation** | **Aug 31; Sep 7; Sep 14** | Aug 30, Sep 6, Sep 13 at 16:00 UTC | 27,000 cohort; 16,982 eligible proxy |
| Excluded September weeks | Sep 21; Sep 28 | Sep 20 and 27 at 16:00 UTC | Avoid example-calibration overlap |
| **Sealed holdout** | **Oct 12** | `2026-10-11T16:00:00Z` | **9,000 feature rows, no labels** |

The dry run explicitly printed `calibration_latest_label_as_of=2026-08-30 16:00 UTC`, matching the first primary prediction cutoff `2026-08-30 16:00 UTC`. The holdout remained sealed. The September 27 official example interval was listed for exclusion checks: `[2026-09-27 13:00, 2026-10-04 13:00) UTC`.

### 15.3 Models actually run and their status

| Model | Role / training approach | Actual execution |
|---|---|---|
| `zero` | All six labels always predicted 0 | **Executed and scored**; fit time printed **0.0s** |
| `persistence` | Last-week activity rule / recurrence heuristic | **Executed and scored**; fit time printed **0.0s** |
| `logistic` | Six independently trained regularized logistic classifiers over EXP-005 features | **Executed and scored** in both full local commands; fit time **6.3s** on first, **6.9s** on repeat |
| `hgb` | Six histogram-gradient-boosting classifiers over the same generated features | **Executed and scored** in second local command; fit time **7.5s** |

Execution durations are **reported fit times**, not total program wall-clock times. Model-specific hyperparameters, saved artifacts and six per-target calibrated thresholds are defined in the source/config and generated files, but were not printed in the pasted evidence; **do not fabricate or infer them here**.

### 15.4 Primary validation leaderboard: actual measured predictive results

**Metric population:** `eligible_proxy` on the primary three weeks, **16,982 user-week predictions**; retrospective labels from the EXP-005 build and its configured maturity policy. Scores are from the **printed full-model output** (the four-model run); they were also consistent with the initial three-model run for overlapping models. The evaluation workflow was configured to use the instructor's F1 conventions and pooled TP/FP/FN across repeated user-week samples, distinct from taking an arithmetic average of weekly scores. Independent per-week scorer-check artifacts were **not provided**.

| Rank | Model | Macro-F1 | Micro-F1 | **Combined F1** | Validation rows | Weeks |
|---:|---|---:|---:|---:|---:|---:|
| **1** | **Persistence** | **0.45178059810352605** | **0.4575740986939861** | **0.45467734839875606** | 16,982 | 3 |
| 2 | Histogram Gradient Boosting (`hgb`) | 0.38548703829707015 | 0.4120467346273798 | **0.39876688646222497** | 16,982 | 3 |
| 3 | Logistic Regression (`logistic`) | 0.3482955461594621 | 0.4147671404308572 | **0.3815313432951597** | 16,982 | 3 |
| 4 | All-zero (`zero`) | 0.00000000000000000 | 0.0000000000000000 | **0.0000000000000000** | 16,982 | 3 |

The source script printed `selected_model_by_primary_eligible_score: "persistence"`. This is **model selection using the primary validation score**, not an unbiased score on an independent untouched test sample. The corresponding run directory was:

```text
experiments/models/exp006/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/
```

The earlier three-model comparison reported the same overlapping model scores and saved to:

```text
experiments/models/exp006/run-build-506d860693df-6040f635d4-e00a053e110a/
```

### 15.5 Pairwise score comparisons and interpretation

Calculated differences from the four-model **primary validation combined F1** scores (percentage points of a 0–1 F1 scale expressed as absolute differences):

| Comparison | Absolute combined-F1 difference | Interpretation |
|---|---:|---|
| Persistence minus HGB | **+0.05591046193653109** | Heuristic outperformed the nonlinear model on these weeks |
| Persistence minus logistic | **+0.07314600510359636** | Heuristic outperformed the regularized linear model |
| HGB minus logistic | **+0.01723554316706527** | Nonlinear model outperformed logistic, but not persistence |
| Persistence minus zero | **+0.45467734839875606** | Nontrivial recurrence-based predictions beat all-zero |

The **measured result** is that persistence achieved the highest reported pooled combined F1, with slightly higher reported **macro** and **micro** scores than the learned alternatives. This refutes the **specific** expectation that the two trained classifiers would automatically outperform the simple recurrence rule under this split, feature set and threshold policy.

**Possible explanations requiring tests, not established mechanisms:** time-varying base rates and the June–August regime shift; mismatch between historical training population and August/September selected users; suboptimal calibration thresholds; a feature representation that misses time-specific dependencies; and model capacity/regularization. The present console output cannot identify which mechanism dominates. Do not characterize logistic/HGB underperformance as definitively caused by overfitting, class imbalance, or miscalibration without supporting diagnostics.

**Comparison to EDA-03:** the original previous-week heuristic's eligible-proxy **0.3504** was a **mean of five individual August weekly scores**, on a different split/label policy. The EXP-006 value **0.454677** is a **pooled three-week score** on Aug 31/Sep 7/Sep 14. It is **invalid** to report their difference as an improvement or deterioration caused by the implementation without recomputing both under the same protocol. EXP-004's **1.0** is calibrated label-reconstruction agreement, **not** an EXP-006 predictive score.

### 15.6 What is verified and what remains unverified

**Verified from user-pasted commands/output:** combined test suite completed (**25 passed, 1 skipped**); dry-run selected 59 features and the same EXP-005 snapshot/build directory; the 189k/27k/27k fit–calibration–primary cohorts; all four models fit and printed primary eligible-proxy scores; a complete four-model run directory was created; persistence ranked first; Oct 12 holdout was flagged sealed by the dry-run output. The CSV preview also showed the same ranking and columns but truncated the long score strings; the full console JSON is authoritative for values in §15.4.

**Not independently supplied or inspected:**

1. `per_target_metrics.csv`: all six targets' support, TP/FP/FN, precision, recall, individual F1, and differences by model. Hence **no six-target confusion table is asserted in this log**.
2. The calibrated six-label thresholds and per-model threshold-tuning report; cannot yet determine whether rare-label false positives or recalls drove the differences.
3. `summary.csv` and `manifest.json` from the final four-model run, including source code commit, full model/config hyperparameters and validation-check warnings.
4. Per-week official evaluator consistency checks / metric aggregation audit and all-cohort leaderboard values; the output claims the configured pooled result, but their independent verification artifacts are not included.
5. Real inference run and final submission-file validation; no held-out October labels or official November truth exist in this evidence.

**Status decision:** Keep the canonical experiment **Ongoing** under the plan's predeclared completion gate: real comparative training and primary evaluation are finished, but the detailed required per-target/threshold and metrics audit has not yet been appended. The **four-model training-and-reporting subtask is completed**, which must not be confused with claiming all required verification is done.

### 15.7 Next validation action and completion gate

From the existing four-model output folder, preserve the original run artifacts and provide the full contents of:

```text
experiments/models/exp006/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/
    manifest.json
    summary.csv
    per_target_metrics.csv
    primary_leaderboard.csv
```

After inspecting these, append a **per-target** results table with TP/FP/FN/support/F1 per model, all six calibration thresholds, confirmation of the pooled/weekly evaluator consistency, and any explicit warnings. Record a personal-project Git commit SHA if available. Only then move EXP-006 to **Completed** in `EXPERIMENT_PLAN.md`, while keeping the Oct 12 independent holdout deferred until it matures and the model/threshold selection policy is frozen.

**Final scope:** Historical proxy-eligible development-validation evidence only. No claims about unseen November test labels or official grading performance.



## 16. Repository structure maintenance — October 14, 2026

This is a structural maintenance entry, not a new modeling experiment or a change
in EXP-005/EXP-006 completion status. The user requested descriptive filenames,
experiment workflows under `experiments/`, shared implementations under `src/`,
and clearer config names. The complete mapping is in `REPOSITORY_STRUCTURE.md`.

- EXP-005 workflow and outputs: `experiments/exp005_temporal_dataset/`.
- EXP-006 workflow and outputs: `experiments/exp006_baseline_comparison/`.
- Config names: `configs/temporal_dataset.json`, `configs/baseline_comparison.json`;
  values match the original run manifests exactly.
- Reusable dataset construction/loading, model implementations, metrics/thresholds,
  and compatibility helpers reside in descriptively named `src/` modules.
- Original scientific manifests, paths recorded within them, and historical log
  commands remain unchanged. Compatibility resolution maps original artifact paths
  and loads models serialized under their former Python module name.

### Executed regression evidence

**Instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`.
No new real-data model fitting, threshold search, or EXP-007 work was performed.

| Check | Observed result |
|---|---|
| Automated suite with plugin autoload disabled and Conda library paths configured | **28 passed, 1 skipped in 6.95s** |
| Byte-level saved-artifact comparison before/after relocation | **43 files unchanged** |
| Both descriptive configs versus archived configuration dictionaries | **Equal values** |
| New EXP-006 CLI, four-model `--dry-run` | Same **189,000 / 27,000 / 27,000** train/calibration/primary rows, **59 features**, and protected holdout |
| New EXP-005 CLI, `--mode sample-check --require-perfect` | **28,710 decisions**, **0 mismatches** for **4,785 users**; reconstruction verification only |
| Original four saved model bundles loaded through compatibility helper | **All 27,000 primary user-week predictions reproduced exactly for each model** |
| Historical holdout label file | **Absent; holdout remains sealed** |

Representative current commands (from the personal project root):

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp/reorganization-checks-20261014
python -m experiments.exp006_baseline_comparison.train --models zero,persistence,logistic,hgb --dry-run
python -m experiments.exp005_temporal_dataset.build_dataset --mode sample-check --require-perfect
python -m experiments.exp006_baseline_comparison.predict --help
```

Model compatibility verification loaded only trusted existing model bundles, used
existing primary feature rows, and compared binary outputs keyed by `(UserID, cutoff_utc)`
to the archived CSVs. It did not inspect or generate holdout labels. This establishes
behavioral preservation of the reorganization; it adds no evidence of better prediction
or future generalization. EXP-007 remains awaiting its separate approval.
