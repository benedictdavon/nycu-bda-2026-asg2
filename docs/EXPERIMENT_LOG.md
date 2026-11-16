# Assignment 2 — Experiment & EDA Log

> **Current results, November 8, 2026.** [§50](#50-revised-dataset-and-policy-comparison) records the revised dataset build, the four H2 policy definitions, the full-history benchmark and the October–November comparison. H2/P016 led the recent comparison on the three shared activities. H2/P021 led the earlier six-activity benchmark.

> **Latest navigation update, October 22:** §29 contains the completed seven-arm feature campaign and original EXP-013's incomplete closure. §30 records the stronger-model registration and course-source refresh. **For the active Luna Max run, EXPERIMENT_PLAN.md §31 controls science and §32 controls source; log §§31–37 record the source amendment, Step 0 controls, completed MLP/TabM results, and the score-free EXP-041 tabular-child registration.** EXP-041 selection scores are not yet read. Dated older summaries below retain their historical meaning.

> **Path migration, October 14:** original commands and paths below are historical execution records. Current entry points and relocated artifact paths are listed in [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md). Archived scientific files are unchanged.

> **Current-state addendum, October 18:** EXP-001–012 are completed with executed
> evidence. Read **§23.3–23.4** for EXP-012's bounded rhythm/state results and
> independent acceptance: retain **HGB with F0** for EXP-013; neither temporal
> addition passed the registered promotion rules. Full regression:
> **114 passed, 1 skipped**. The October 12 holdout remains sealed.
> Dated status statements below preserve their historical meaning; the current
> experiment register is [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md).


**Course:** NYCU · Big Data Analytics Techniques and Applications · Fall 2026  
**Assignment:** Assignment 2 — Predicting GitHub User Activity  
**Historical log status (2026-10-14):** EXP-001 through EXP-005 completed with recorded evidence; EXP-006 ongoing at that date. The newer refreshed-source Luna Max execution is documented in §§31–36: Step 0, EXP-036-R and EXP-036-T are fully executed and verified; both neural tabular families failed their development gate. EXP-041-tabular is next. No October outcome or official November test score is included.
**Data access:** Official instructor-supplied GitHub relational dataset only  
**Authoring date:** 2026-10-14 (Asia/Taipei)  
**Initial observed instructor snapshot:** `506d860693df60a698b68aa63a8edb4a984c8e1b`  
**Primary evidence:** Executed `01_data_audit(1).ipynb`, `02_deeper_data_analysis(1).ipynb`, and `03_label_and_split_validation(5).ipynb`; subsequent **user-supplied diagnostic outputs in this conversation** for EXP-004. All used the same instructor SHA unless explicitly marked otherwise.

> **Purpose.** This is the single persistent, auditable record for the assignment and for coding-agent handoff. The original EDA-01/02 measurements below are preserved from the supplied log. EDA-03 results are verified against saved executed-notebook outputs; EXP-004 follow-on diagnostics are supported by the exact tables printed and supplied in this conversation (not necessarily saved into the notebook file). A **non-trained previous-week heuristic** was scored on reconstructed historical labels, and **sample-ground-truth reconstruction** was verified separately. Later EXP-006 fitted logistic regression and histogram gradient boosting and compared them with two heuristics on a different historical validation protocol; see §15 for these measured results. No result here is an official November prediction score, and no untouched October historical holdout has been evaluated. Do not silently rewrite historical entries; append corrections and add dated current-state summaries.

## Reader's map

**Current state: read §§31–37 for the active refreshed-source campaign and its exact status.** Sections 1–30 preserve earlier dated evidence and may contain prior-status statements superseded by later append-only entries. Step 0 is independently verified in §32; the TabM pin is in §33; MLP is recorded in §§34–35; TabM is recorded in §36; and EXP-041's named two-member child registration is in §37. Both tabular neural families failed the frozen standalone gate. The EXP-041 child is registered; its selection scores are next and transfer scores remain locked. No October outcome or official November test score has been evaluated.

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


## 17. EXP-007: Frozen-model baseline diagnostics and EXP-006 audit closure — October 14, 2026

**Status:** **Completed** for EXP-007; this evidence also closes EXP-006's outstanding audit. This dated entry supersedes §15's earlier Ongoing status while preserving its original measurements and §16's structural history.

### 17.1 Authorization, provenance, and commands

The user authorized execution of the EXP-007 frozen-model plan on October 14 (Asia/Taipei). No fitting or feature changes were performed. Instructor SHA: `506d860693df60a698b68aa63a8edb4a984c8e1b`. Student base commit: `7cff786911570598007681901b4bd4c9e5272d70`; EXP-007 implementation was an uncommitted working-tree addition at execution. The run manifest records the source-tree status and SHA256 of each implementation/config/test file, so the base commit alone must not be treated as the implementation revision.

- Dataset: `experiments/exp005_temporal_dataset/runs/build-506d860693df-6040f635d4/`.
- Frozen four-model archive: `experiments/exp006_baseline_comparison/runs/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/`.
- Final diagnostic run: `experiments/exp007_baseline_diagnostics/runs/run-frozen-fc2acd6fe5b2-grid001-final/`.
- Earlier implementation-validation run: `runs/run-frozen-fc2acd6fe5b2-grid001/` under the same experiment; preserved, never overwritten. The final run uses the canonical shared scorer within search and contains the expanded findings.
- Config: `configs/baseline_diagnostics.json`; **27 consumed input files pinned by SHA256**, including dataset/audit, models, predictions, thresholds, archived scores/checks and instructor evaluator.

Executed from the project root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "2"
$env:LOKY_MAX_CPU_COUNT = "2"
python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp/exp007-final
python -m experiments.exp007_baseline_diagnostics.run
python -m pytest tests/test_baseline_diagnostics.py -q -p no:cacheprovider --basetemp=.pytest_tmp/exp007-findings-check
```

The full regression suite reported **45 passed, 1 skipped in 12.82s**. Following the last findings-generator edit, the diagnostic acceptance suite was rerun and reported **17 passed**. The skip is the existing optional live official-label regression; no holdout was opened to run it. Tests cover key alignment/shuffling, duplicate/null/missing/mismatched keys, invalid probabilities/labels/thresholds, tied score boundaries, empty-positive scoring, exact maxima against exhaustive shared scoring, optimized search versus canonical coordinate scoring, pooled count aggregation and weekly official evaluation, calibration/holdout access guards, immutable inputs, refusal to overwrite runs/model paths, and stopping on archive mismatches before diagnostics.

Final real diagnostic runtime: **19.139s** on CPU. Runtime: Python **3.12.9**, NumPy **2.2.5**, pandas **2.3.3**, scikit-learn **1.8.0**. Archived training used Python 3.11.17 / NumPy 2.4.6 / scikit-learn 1.9.1. Six distinct cross-version estimator-loading warnings are retained in the manifest. Compatibility is accepted only for this tested run: all archived binaries, thresholds, calibration traces and metrics reproduce; original probability tables did not exist, so exact original-runtime probability equality is not asserted.

### 17.2 Hard reproduction gate and frozen protocol

- All **27,000 primary user-week rows × six targets** reproduce exactly for **each of four models** (648,000 binary decisions total), aligned by unique `(UserID, cutoff_utc)` keys.
- Bundle/JSON thresholds agree exactly; original learned-model calibration searches reproduce thresholds and pass traces.
- All **32 archived pooled summary rows** and **192 archived per-target metric rows** reproduce, including calibration/primary, fixed/tuned thresholds and both populations. The archived leaderboard and manifest metrics also match.
- **96 archived weekly score records** (four models × two slices × two threshold modes × two populations × three weeks) reproduce against saved checks and the instructor evaluator. Every additional diagnostic threshold vector is also checked week-by-week against that evaluator.
- Numeric tolerance: **absolute 1e-10, relative 0**; binary predictions and thresholds require exact equality.
- All 27 input hashes match before and after execution. Models and production thresholds are never written by the runner; eight generated probability Parquets stay local/ignored.
- **59 fixed features**; **189,000 fit rows**, **27,000 calibration rows**, **27,000 primary rows**. Eligible primary population: **16,982 user-weeks**.
- Fit: March 2–July 20; calibration: August 3/10/17; maturity gap: August 24; primary: August 31, September 7/14. Dates are Taiwan time. October 12 holdout labels remain absent and access is rejected.

### 17.3 Complete archived operating-point audit

Pooled primary metrics for both populations under every diagnostic mode:

| model | population | threshold_mode | rows | macro_f1 | micro_f1 | score |
| --- | --- | --- | --- | --- | --- | --- |
| zero | all | archived | 27000 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | archived | 16982 | 0.000000 | 0.000000 | 0.000000 |
| zero | all | default_0.5 | 27000 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | default_0.5 | 16982 | 0.000000 | 0.000000 | 0.000000 |
| persistence | all | archived | 27000 | 0.439896 | 0.444203 | 0.442050 |
| persistence | eligible_proxy | archived | 16982 | 0.451781 | 0.457574 | 0.454677 |
| persistence | all | default_0.5 | 27000 | 0.439896 | 0.444203 | 0.442050 |
| persistence | eligible_proxy | default_0.5 | 16982 | 0.451781 | 0.457574 | 0.454677 |
| logistic | all | archived | 27000 | 0.337233 | 0.402149 | 0.369691 |
| logistic | eligible_proxy | archived | 16982 | 0.348296 | 0.414767 | 0.381531 |
| logistic | all | default_0.5 | 27000 | 0.156665 | 0.270633 | 0.213649 |
| logistic | eligible_proxy | default_0.5 | 16982 | 0.162835 | 0.280683 | 0.221759 |
| hgb | all | archived | 27000 | 0.370280 | 0.392780 | 0.381530 |
| hgb | eligible_proxy | archived | 16982 | 0.385487 | 0.412047 | 0.398767 |
| hgb | all | default_0.5 | 27000 | 0.167020 | 0.256232 | 0.211626 |
| hgb | eligible_proxy | default_0.5 | 16982 | 0.173808 | 0.266131 | 0.219970 |
| zero | all | refined_calibration | 27000 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | refined_calibration | 16982 | 0.000000 | 0.000000 | 0.000000 |
| zero | all | matched_hindsight | 27000 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | matched_hindsight | 16982 | 0.000000 | 0.000000 | 0.000000 |
| persistence | all | refined_calibration | 27000 | 0.439896 | 0.444203 | 0.442050 |
| persistence | eligible_proxy | refined_calibration | 16982 | 0.451781 | 0.457574 | 0.454677 |
| persistence | all | matched_hindsight | 27000 | 0.439896 | 0.444203 | 0.442050 |
| persistence | eligible_proxy | matched_hindsight | 16982 | 0.451781 | 0.457574 | 0.454677 |
| logistic | all | refined_calibration | 27000 | 0.339942 | 0.402076 | 0.371009 |
| logistic | eligible_proxy | refined_calibration | 16982 | 0.351107 | 0.414789 | 0.382948 |
| logistic | all | matched_hindsight | 27000 | 0.459567 | 0.459945 | 0.459756 |
| logistic | eligible_proxy | matched_hindsight | 16982 | 0.492566 | 0.508121 | 0.500343 |
| hgb | all | refined_calibration | 27000 | 0.364737 | 0.393093 | 0.378915 |
| hgb | eligible_proxy | refined_calibration | 16982 | 0.377159 | 0.407722 | 0.392440 |
| hgb | all | matched_hindsight | 27000 | 0.425192 | 0.401367 | 0.413280 |
| hgb | eligible_proxy | matched_hindsight | 16982 | 0.476188 | 0.479261 | 0.477724 |

Archived six-target thresholds (order follows the named targets):

| model | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
| --- | --- | --- | --- | --- | --- | --- |
| hgb | 0.150000 | 0.100000 | 0.125000 | 0.075000 | 0.075000 | 0.450000 |
| logistic | 0.150000 | 0.125000 | 0.100000 | 0.125000 | 0.100000 | 0.400000 |
| persistence | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |
| zero | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 | 0.500000 |

Primary eligible-proxy confusion audit; all-cohort and all weekly/slice/mode tables are in `per_target_metrics.csv`:

| model | target | tp | fp | fn | tn | support | predicted_positive_rate | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | open_pr | 0 | 0 | 2969 | 14013 | 2969 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| zero | open_issue | 0 | 0 | 1286 | 15696 | 1286 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| zero | issue_comment | 0 | 0 | 2358 | 14624 | 2358 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| zero | pr_review | 0 | 0 | 1924 | 15058 | 1924 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| zero | pr_review_comment | 0 | 0 | 1376 | 15606 | 1376 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| zero | push | 0 | 0 | 4358 | 12624 | 4358 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| persistence | open_pr | 1202 | 1117 | 1767 | 12896 | 2969 | 0.136556 | 0.518327 | 0.404850 | 0.454614 |
| persistence | open_issue | 524 | 624 | 762 | 15072 | 1286 | 0.067601 | 0.456446 | 0.407465 | 0.430567 |
| persistence | issue_comment | 933 | 873 | 1425 | 13751 | 2358 | 0.106348 | 0.516611 | 0.395674 | 0.448127 |
| persistence | pr_review | 783 | 637 | 1141 | 14421 | 1924 | 0.083618 | 0.551408 | 0.406965 | 0.468301 |
| persistence | pr_review_comment | 513 | 476 | 863 | 15130 | 1376 | 0.058238 | 0.518706 | 0.372820 | 0.433827 |
| persistence | push | 1896 | 1725 | 2462 | 10899 | 4358 | 0.213226 | 0.523612 | 0.435062 | 0.475248 |
| logistic | open_pr | 597 | 221 | 2372 | 13792 | 2969 | 0.048169 | 0.729829 | 0.201078 | 0.315289 |
| logistic | open_issue | 327 | 203 | 959 | 15493 | 1286 | 0.031210 | 0.616981 | 0.254277 | 0.360132 |
| logistic | issue_comment | 572 | 260 | 1786 | 14364 | 2358 | 0.048993 | 0.687500 | 0.242578 | 0.358621 |
| logistic | pr_review | 315 | 104 | 1609 | 14954 | 1924 | 0.024673 | 0.751790 | 0.163721 | 0.268886 |
| logistic | pr_review_comment | 195 | 67 | 1181 | 15539 | 1376 | 0.015428 | 0.744275 | 0.141715 | 0.238095 |
| logistic | push | 2634 | 2608 | 1724 | 10016 | 4358 | 0.308680 | 0.502480 | 0.604406 | 0.548750 |
| hgb | open_pr | 520 | 187 | 2449 | 13826 | 2969 | 0.041632 | 0.735502 | 0.175143 | 0.282916 |
| hgb | open_issue | 398 | 280 | 888 | 15416 | 1286 | 0.039925 | 0.587021 | 0.309487 | 0.405295 |
| hgb | issue_comment | 507 | 212 | 1851 | 14412 | 2358 | 0.042339 | 0.705146 | 0.215013 | 0.329542 |
| hgb | pr_review | 533 | 289 | 1391 | 14769 | 1924 | 0.048404 | 0.648418 | 0.277027 | 0.388201 |
| hgb | pr_review_comment | 400 | 292 | 976 | 15314 | 1376 | 0.040749 | 0.578035 | 0.290698 | 0.386847 |
| hgb | push | 2068 | 1526 | 2290 | 11098 | 4358 | 0.211636 | 0.575403 | 0.474530 | 0.520121 |

Archived weekly primary scores, each independently checked with the instructor evaluator:

| model | population | cutoff_utc | rows | macro_f1 | micro_f1 | score |
| --- | --- | --- | --- | --- | --- | --- |
| zero | all | 2026-08-30 16:00:00+00:00 | 9000 | 0.000000 | 0.000000 | 0.000000 |
| zero | all | 2026-09-06 16:00:00+00:00 | 9000 | 0.000000 | 0.000000 | 0.000000 |
| zero | all | 2026-09-13 16:00:00+00:00 | 9000 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | 2026-08-30 16:00:00+00:00 | 6030 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | 2026-09-06 16:00:00+00:00 | 5656 | 0.000000 | 0.000000 | 0.000000 |
| zero | eligible_proxy | 2026-09-13 16:00:00+00:00 | 5296 | 0.000000 | 0.000000 | 0.000000 |
| persistence | all | 2026-08-30 16:00:00+00:00 | 9000 | 0.213783 | 0.242465 | 0.228124 |
| persistence | all | 2026-09-06 16:00:00+00:00 | 9000 | 0.478840 | 0.474352 | 0.476596 |
| persistence | all | 2026-09-13 16:00:00+00:00 | 9000 | 0.511161 | 0.525545 | 0.518353 |
| persistence | eligible_proxy | 2026-08-30 16:00:00+00:00 | 6030 | 0.219128 | 0.249344 | 0.234236 |
| persistence | eligible_proxy | 2026-09-06 16:00:00+00:00 | 5656 | 0.491784 | 0.488506 | 0.490145 |
| persistence | eligible_proxy | 2026-09-13 16:00:00+00:00 | 5296 | 0.524850 | 0.542525 | 0.533688 |
| logistic | all | 2026-08-30 16:00:00+00:00 | 9000 | 0.137456 | 0.202277 | 0.169866 |
| logistic | all | 2026-09-06 16:00:00+00:00 | 9000 | 0.343533 | 0.405152 | 0.374343 |
| logistic | all | 2026-09-13 16:00:00+00:00 | 9000 | 0.470674 | 0.531118 | 0.500896 |
| logistic | eligible_proxy | 2026-08-30 16:00:00+00:00 | 6030 | 0.141933 | 0.208853 | 0.175393 |
| logistic | eligible_proxy | 2026-09-06 16:00:00+00:00 | 5656 | 0.354796 | 0.418111 | 0.386454 |
| logistic | eligible_proxy | 2026-09-13 16:00:00+00:00 | 5296 | 0.484729 | 0.548846 | 0.516787 |
| hgb | all | 2026-08-30 16:00:00+00:00 | 9000 | 0.162084 | 0.201617 | 0.181850 |
| hgb | all | 2026-09-06 16:00:00+00:00 | 9000 | 0.397467 | 0.409915 | 0.403691 |
| hgb | all | 2026-09-13 16:00:00+00:00 | 9000 | 0.477214 | 0.495715 | 0.486464 |
| hgb | eligible_proxy | 2026-08-30 16:00:00+00:00 | 6030 | 0.166522 | 0.211044 | 0.188783 |
| hgb | eligible_proxy | 2026-09-06 16:00:00+00:00 | 5656 | 0.412914 | 0.429034 | 0.420974 |
| hgb | eligible_proxy | 2026-09-13 16:00:00+00:00 | 5296 | 0.498863 | 0.522088 | 0.510476 |


### 17.4 Threshold diagnostics, ranking, and transfer observations

Calibration and hindsight use the **identical** 0.001–1.000 grid (0.001 step), archived values included, two starting vectors (archived and 0.5), fixed TARGETS order, ten-pass limit, and deterministic ties (nearest 0.5, then smaller). All starts converged. Hindsight uses primary labels only for explicitly labeled diagnostics and never overwrites production settings. Exact individual-target maxima separately exhaust attainable score-boundary predictions in the shared [0,1] threshold domain, including >= ties; they are **not** a combined-F1 upper bound.

- Logistic eligible-proxy combined F1: archived **0.381531**, refined calibration **0.382948** (**+0.001417**), matched hindsight **0.500343** (**+0.118812**). Calibration improves two of three primary weeks, worsening September 14; hindsight improves all three by **+0.194706 / +0.155554 / +0.037116**.
- HGB: archived **0.398767**, refined calibration **0.392440** (**−0.006327**), matched hindsight **0.477724** (**+0.078957**). Refined calibration worsens all three weeks; hindsight improves all three by **+0.149752 / +0.100813 / +0.019405**.
- Persistence remains **0.454677** under every joint-search diagnostic. All-cohort hindsight logistic **0.459756** exceeds all-cohort persistence **0.442050**, while hindsight HGB **0.413280** still trails it. Population differences matter; eligible-proxy hindsight results cannot be generalized to all users or the future test roster.
- On five non-push targets, eligible-proxy calibration prevalence is **1.029–2.018%**, versus primary **7.573–17.483%**. Push instead moves **27.032% → 25.662%**. At archived thresholds both learned models have higher precision and lower recall than persistence on every non-push target. Their low predicted-positive rates directly explain the recall/F1 tradeoff at those operating points; why the underlying activity/observation distribution changed remains a hypothesis.
- Both learned models exceed persistence's **average precision in all 18 target-week comparisons** on the eligible primary population. Pooled logistic AP ranges **0.431078–0.564408**, HGB **0.394823–0.575500**, versus persistence **0.230857–0.372781**. AP and AUC are supplemental ranking metrics, with prevalence reported alongside AP; no-positive AP and one-class AUC are explicitly undefined. AP superiority does not imply better F1 at every attainable threshold.
- Logistic exact pooled individual-target F1 maxima exceed persistence on **all six targets**. HGB's `open_pr` maximum is **0.400742**, below persistence **0.454614**; the other five HGB maxima exceed persistence. Even independently hindsight-tuned weekly HGB `open_pr` maxima trail persistence on September 7 (**0.456534 vs 0.565581**) and September 14 (**0.470588 vs 0.509218**). This is evidence of a target-specific thresholding/ranking limitation, not just insufficient grid resolution.
- Push complementarity: logistic adds **72 / 483 / 204** true positives beyond persistence across the three weeks, but also **423 / 395 / 107** false positives; HGB adds **39 / 336 / 91** true positives and **175 / 220 / 38** false positives. Each learned model beats persistence's push F1 in **two of three weeks**. Useful disagreement is observed, but a simple union or production combination is not validated.

Detailed primary score comparisons, exact thresholds/counts, weekly ranking/prevalence, and overlap counts are retained in `threshold_comparison.csv`, `exact_target_maxima.csv`, `ranking_metrics.csv`, `distribution_transfer.csv` and `complementarity.csv`. `FINDINGS.md` contains the concise measured synthesis and full opportunity tables.

### 17.5 Decision, limitations, and completion

**Decision:** Prioritize **EXP-008 chronological replay** of the frozen baseline and threshold-calibration policy. Both hindsight gains exceed the **0.02 investigation trigger**, while refined calibration does not improve reporting consistently. Investigate rolling threshold transfer and activity regimes before changing training/features. Keep HGB `open_pr` training-recency/representation analysis as a target-specific follow-up; carry push disagreement into a later controlled target-specific combination experiment. EXP-008 is **Pending**, not executed by this request.

**Limitations:** Primary weeks were already inspected development data. Hindsight is an opportunity diagnostic, never a promotion criterion or independent evidence of November performance. Joint coordinate search is local, even though these runs converged. Individual maxima optimize separate target F1 under [0,1] thresholds, not joint combined F1. Proxy eligibility is retrospective and differs from the official roster; seven-day label maturation is an assumption, not certified data completeness. Version warnings and the absence of original probability archives limit cross-runtime probability claims. No causal collection-regime explanation, new model fit, independent holdout evaluation, or official final test score is established.

**Completion:** EXP-007's reproduction checks, full diagnostics, tests, limitations, artifacts and next decision are recorded. This evidence satisfies EXP-006's remaining six-target, calibrated-threshold, manifest/config, both-population and pooled/weekly evaluator-audit checklist. Mark **EXP-006 and EXP-007 Completed** in the plan; retain the independent October holdout as a separately deferred task.


## 18. EXP-008: Chronological threshold-transfer replay — October 15, 2026

**Status:** **Completed** for the configured replay. This entry supersedes §17.5's
statement that EXP-008 was pending; that statement correctly described the state
at the close of EXP-007.

### 18.1 Frozen protocol and reproducible execution

The replay keeps all four EXP-006 model bundles, their 59 features, model settings,
and archived thresholds fixed. It compares the archived thresholds with two
predeclared rolling policies for the learned models: thresholds tuned on the latest
**one** or **three** validation weeks whose labels are mature by each reporting
cutoff. The tuner uses the original EXP-006 eligible-proxy calibration population,
26-point threshold grid, three-pass coordinate search, and pooled official combined
F1 objective. The zero and persistence rules remain fixed at 0.5.

Reporting cutoffs match EXP-006 exactly: Taiwan dates **August 31, September 7,
and September 14** (UTC cutoffs August 30, September 6, and September 13 at 16:00).
All policies score the same user-week keys in both the all-user and
eligible-proxy populations. The inherited calibrated-example check excludes any
window whose label interval intersects September 27 13:00 UTC through October 4
13:00 UTC. The October 12 holdout remains sealed.

At each origin the selected one-week calibration cutoffs were August 17, August
24, and August 31. The three-week windows were August 3/10/17, August 10/17/24,
and August 17/24/31. Their latest `label_as_of_utc` equals the corresponding
report cutoff; earlier labels in each window matured seven or fourteen days
before it. Each exact cutoff, maturity timestamp, row count, and origin is recorded
in `calibration_window_audit.csv`. `tune_mature_thresholds` accepts only the
selected calibration frame and maturity audit, so reporting labels are not an
input to threshold selection.

The configured invocation from the project root is:

```powershell
python -m pytest tests/test_chronological_threshold_transfer.py -q -p no:cacheprovider --basetemp=.pytest_tmp/exp008-unit
python -m experiments.exp008_chronological_threshold_transfer.run --output experiments/exp008_chronological_threshold_transfer/runs/run-frozen-fc2acd6fe5b2-rolling-1-3-final-v2
```

The focused suite reported **5 passed**. The final real replay used Python
3.11.17, NumPy 2.4.6, pandas 2.3.3, and scikit-learn 1.9.1. It completed in
12.104 seconds with no model-load warnings. The configuration pins **29 consumed files** across EXP-005 data and
audit artifacts, EXP-006 models/results, the evaluator, and temporal/baseline
configs. All locks are rechecked before the final manifest is written. The final
source/config/test hashes and environment are captured in the run manifest.

Final run:
`experiments/exp008_chronological_threshold_transfer/runs/run-frozen-fc2acd6fe5b2-rolling-1-3-final-v2/`.
It contains weekly and pooled scores, per-target confusion metrics, per-target
ranking metrics, rolling threshold vectors and search traces, maturity-window
audit, score variance, report probabilities, per-origin binary predictions,
archive reproduction checks, findings, and a manifest. There are no model files
in the output.

### 18.2 Reproduction gate and measured results

All four frozen bundles reproduced all **27,000** archived report binaries exactly
(648,000 target decisions). The archived thresholds reproduced the saved EXP-006
weekly official scores for both populations: **24/24** model/population/week
checks passed. Archived pooled summary and per-target rows also match the EXP-006
CSV archive within absolute `1e-10` tolerance. The 29 pinned input hashes were
unchanged at completion.

Pooled official combined F1:

| Model | Population | Archived | Rolling 1 mature week | Rolling 3 mature weeks |
|---|---|---:|---:|---:|
| Persistence | all | 0.442050 | — | — |
| Persistence | eligible proxy | 0.454677 | — | — |
| Logistic | all | 0.369691 | 0.370296 | 0.392674 |
| Logistic | eligible proxy | 0.381531 | 0.408198 | 0.405006 |
| HGB | all | 0.381530 | 0.345641 | 0.396974 |
| HGB | eligible proxy | 0.398767 | 0.399386 | 0.416591 |

Eligible-proxy weekly scores, in reporting order August 31 / September 7 / September 14:

| Model and policy | Weekly scores |
|---|---|
| Persistence, archived | 0.234236 / 0.490145 / 0.533688 |
| Logistic, archived | 0.175393 / 0.386454 / 0.516787 |
| Logistic, rolling 1 | 0.164899 / 0.426010 / 0.483260 |
| Logistic, rolling 3 | 0.175393 / 0.388518 / 0.547107 |
| HGB, archived | 0.188783 / 0.420974 / 0.510476 |
| HGB, rolling 1 | 0.184171 / 0.440924 / 0.450597 |
| HGB, rolling 3 | 0.188783 / 0.416914 / 0.536984 |

Eligible-proxy sample standard deviation across those three weekly scores was
**0.161790** for persistence; logistic archived/rolling-1/rolling-3 was
**0.172281 / 0.169711 / 0.186523**; HGB was **0.166037 / 0.151106 / 0.176873**.
The complete all-user and eligible-proxy variance table is in `variance.csv`.
The 192-row `ranking_metrics.csv` contains pooled and weekly AP/AUC, prevalence,
and explicit undefined statuses for every model, population, and target. The
per-target output contains TP/FP/FN/TN, precision, recall, F1, support, and
predicted-positive rate for each weekly and pooled policy result.

### 18.3 Decision and limits

Neither learned model surpassed persistence's pooled score on either population
under the rolling policies. On eligible users, logistic rose by **0.026666** with
one mature week and **0.023475** with three; HGB rose by **0.000620** and
**0.017824**. Those gains still leave logistic **0.046480 / 0.049671** and HGB
**0.055291 / 0.038086** below persistence. In the all-user population, the best
rolling scores also remain below persistence. Results vary by origin: for example,
one-week rolling HGB rises on September 7 and falls on September 14. **Decision:**
keep archived thresholds as the reference and do not promote the rolling policies.

This replay covers only three previously inspected development weeks. The
eligible-proxy cohort is retrospective and may differ from the official test
roster. The seven-day label-maturation allowance is an explicit reconstruction
policy, not a certified collection-completeness watermark. The final run manifest
records the active runtime and exact model-load warnings; its archived bundles
loaded under scikit-learn **1.9.1** with **no warnings**. Archived binary
predictions reproduce exactly, but original archived probability tables do not
exist, so equality with original-run probabilities is unverified. AP/AUC and
rolling calibration use probabilities from the recorded runtime. No independent
future performance or official final test score is established.


## 19. EXP-009: Controlled HGB training-recency replay — October 15, 2026

**Status:** **Completed** for the predeclared development replay. This experiment
compares training-window recency while holding the 59-feature HGB setup and
chronological evaluation schedule fixed. It answers the specific HGB `open_pr`
recency question raised by EXP-007; it is not a new production training run.

### 19.1 Protocol and execution

The three controls are the frozen EXP-006 HGB archive trained on 21 weekly
periods, a current-runtime refit using the same full 21-week window, and a
predeclared recent-eight-week refit. The recent period is June 1 through July 20
Taiwan dates (May 31 through July 19 at 16:00 UTC in the saved training audit).
All 21 archived training weeks have `label_as_of_utc` no later than the first
calibration origin, **August 2, 2026 16:00 UTC**. Features, HGB parameters, and
the EXP-006 model role are held constant. Refit artifacts are local to this
experiment and do not replace the archived bundle.

Each control is evaluated at the same three EXP-006 reporting origins (Taiwan
August 31, September 7, and September 14) under archived thresholds and the
rolling one- and three-mature-week threshold policies. At each origin the rolling
thresholds use only the designated prior validation cutoffs whose labels mature
by that origin; the eligible-proxy group and original threshold grid/search are
retained. The report labels never enter model fitting or calibration. The
EXP-006 calibrated-example interval exclusion is retained. October 12 labels
were neither read nor created.

From the project root, the focused check and replay are:

```powershell
python -m pytest tests/test_hgb_recency.py -q -p no:cacheprovider --basetemp=.pytest_tmp/exp009-unit
python -m experiments.exp009_hgb_recency.run
```

The focused suite reported **3 passed**. The final run is
`experiments/exp009_hgb_recency/runs/run-hgb-recency-final-v2/`; it records
Python 3.11.17, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, no load/fit
warnings, and about 25.3 seconds elapsed. Its manifest contains hashes for **29
inputs** and **14 source/config/test files**. It saves the refit bundles,
probabilities, per-origin predictions, threshold traces, week/window audits,
confusion and ranking metrics, variance, findings, and provenance.

The current-runtime full-21 refit's report probabilities match the frozen
archive exactly (maximum absolute difference **0**). Independent rerun produced
the same parameters, report probabilities, and compact evidence; serialized
joblib bytes may differ because model archive serialization is not byte-stable.

### 19.2 Measured results

Pooled eligible-proxy official combined F1 (persistence is **0.454677**):

| HGB control | Archived thresholds | Rolling 1 mature week | Rolling 3 mature weeks |
|---|---:|---:|---:|
| Frozen archive, 21 weeks | 0.398767 | 0.399386 | 0.416591 |
| Current-runtime refit, 21 weeks | 0.398767 | 0.399386 | 0.416591 |
| Refit, recent 8 weeks | 0.352259 | 0.405976 | 0.426468 |

For pooled eligible-proxy `open_pr`, persistence has F1 **0.454614** and average
precision **0.313896**. The frozen and full-21 refit controls have F1/AP of
**0.282916 / 0.409868** with archived thresholds, **0.319636 / 0.409868** with
rolling-1, and **0.300393 / 0.409868** with rolling-3. The recent-eight-week
control has F1/AP **0.236139 / 0.538654** under archived thresholds,
**0.394233 / 0.538654** with rolling-1, and **0.380237 / 0.538654** with
rolling-3. Thus recency improves ranking and rolling F1 relative to the frozen
model, but `open_pr` F1 remains below persistence. The complete per-week and
per-target confusion/ranking outputs and score variance are in the final run.

### 19.3 Decision and limits

The recency effect is measurable but does not close the predictive-performance
gap: the best recent-eight-week six-target combined F1 is **0.426468**, still
**0.028209** below persistence, and its best `open_pr` F1 is **0.394233**, still
**0.060381** below persistence. Keep the frozen 21-week model as the reference;
do not promote recent training from these scores. The full-history refit control
isolates runtime effects for this environment, but portability to other runtimes
is not established. These are three already-inspected development weeks, and the
eligible-proxy population is not the official final-test roster. The seven-day
maturation rule is a reconstruction allowance, not a completeness watermark.


## 20. EXP-010: Controlled push-target combinations — October 15, 2026

**Status:** **Completed** for the predeclared development replay. The user
clarified that EXP-010 is a push-target combination study. The older proposal in
§12.2 using this ID for an independent label-policy check remains historical and
was not executed under this ID.

### 20.1 Protocol and execution

EXP-010 reuses the frozen EXP-006 persistence, HGB, and logistic bundles/scores;
it does not fit models or write model artifacts. Persistence remains fixed on the
five targets other than `push`. The fixed 13-candidate set contains persistence;
tuned HGB and logistic push thresholds; HGB and logistic AND/OR rules; and
probability blends of each learned score with persistence at weights **0.25, 0.50,
and 0.75**. Tuned thresholds use the predeclared 26-point grid. A deterministic
tie preference resolves equal calibration scores.

At each of the three EXP-006 origins, candidate and threshold selection uses only
the latest **three mature prior validation weeks**, eligible-proxy users, and the
official combined-F1 objective. The report cutoffs are Taiwan August 31,
September 7, and September 14; selection cutoffs and label-maturity evidence are
recorded per origin. Reporting labels are evaluation-only. The seven-day
label-maturation allowance does not certify collection completeness. The replay
retains the calibrated-example interval exclusion and leaves October 12 sealed.

From the project root, the focused check and replay are:

```powershell
python -m pytest tests/test_push_combinations.py -q -p no:cacheprovider --basetemp=.pytest_tmp/exp010-unit
python -m experiments.exp010_push_combinations.run --output experiments/exp010_push_combinations/runs/run-reproduction
```

The final recorded run used `run-push-combinations-final-v2`. The runner requires
a new output directory and will not overwrite a saved run; use another unused
direct child of `runs/` for each reproduction.

The focused suite reported **4 passed**. The final run is
`experiments/exp010_push_combinations/runs/run-push-combinations-final-v2/`; it
records Python 3.11.17, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, no model
load warnings, and about 11.9 seconds elapsed. Its manifest pins **29 inputs**
and **14 source/config/test files**. It retains candidate score tables, all
threshold-search traces, per-origin policy selection, calibration-window audit,
report probability and prediction Parquets, weekly and pooled official metrics,
per-target rankings/confusions, true-positive overlaps and counts, variance,
findings, and provenance. The final source correction clarifies that the seven-day
value is the label-maturation allowance, distinct from the three-week calibration
window.

### 20.2 Calibration choices and report results

The mature calibration replay selected **HGB tuned push at threshold 0.45 for all
three origins**. Calibration combined-F1 was **0.404764**, **0.396777**, and
**0.328073**, respectively. Pooled eligible-proxy official combined F1 is:

| Policy evaluated on the same report weeks | Combined F1 | Delta vs persistence |
|---|---:|---:|
| Persistence | 0.454677 | +0.000000 |
| Calibration-selected HGB push | 0.465391 | +0.010714 |
| HGB tuned push | 0.465391 | +0.010714 |
| Logistic tuned push | 0.471867 | +0.017190 |
| HGB AND / OR | 0.449597 / 0.469249 | -0.005080 / +0.014571 |
| Logistic AND / OR | 0.453904 / 0.472437 | -0.000774 / +0.017760 |
| Logistic blend 0.25 / 0.50 / 0.75 | 0.474577 / 0.474577 / 0.474537 | +0.019900 / +0.019900 / +0.019860 |

The HGB blend scores are **0.472446 / 0.472446 / 0.469916** at weights 0.25 / 0.50
/ 0.75. Logistic blend 0.25 and 0.50 had the highest pooled report score, but
report scores did not affect selection; that post-selection comparison does not
replace the HGB calibration choice. For the selected policy, pooled eligible push
F1/AP is **0.520121 / 0.575500**; persistence is **0.475248 / 0.372781**. The
per-origin true-positive overlap table distinguishes new and missed positives
relative to persistence, HGB, and logistic predictions. Every non-push binary
prediction remains identical to persistence.

The independent audit of the final EXP-008–010 artifacts checked **256 summary
rows, 768 metric rows/values, 6,144 target-count cells, 192 direct official
weekly scores, and 1,056 ranking rows**. It verified the locked inputs and source
hashes for all runs; a separate calibration recomputation matched all **39
candidate choices** and **624 threshold-grid objectives** across the three
origins. Full regression completed with **57 passed, 1 skipped**.

### 20.3 Decision and limits

The calibration-selected candidate exceeds persistence by **0.010714** on the
same pooled eligible-proxy reports while preserving persistence on the other five
targets. Some unselected fixed candidates score higher on these reports; no
report-optimal rule was promoted. The chosen candidate and threshold are therefore
the calibration result, not a claim that the report leaderboard identifies a
stable production policy. With only three previously inspected development weeks,
the result does not establish independent future performance or warrant
production promotion. Eligible-proxy users differ from the official roster, and
the seven-day maturity allowance is not a completeness guarantee. October 12
labels remain sealed.

## 21. EXP-008–010 — Primary-agent independent verification — October 15, 2026

The requested GPT-6 Luna subagent at Max reasoning effort implemented and executed
the three experiments. The primary agent then reviewed the code, requested fixes
for probability/frame alignment, duplicate calibration rows, integer-mask overlap
counting, fractional-label validation, and provenance/documentation defects, and
independently reran the finalized workflows.

The independent audit recomputed **256 score rows**, **768 macro/micro/combined-F1
values**, **6,144 confusion-count values**, **192 direct instructor weekly scores**,
and **1,056 AP/AUC rows**. It checked threshold predictions, exact mature windows,
the 21-/8-week training windows, the fixed five non-push predictions, all push-rule
formulas, and new/missed/net true-positive overlaps. A separate calculation from
raw calibration data and frozen models reproduced all **39 candidate scores**
and **624 threshold-grid scores**, confirming HGB push at **0.45** was selected
without reporting labels at every origin.

All **27 original frozen inputs** match the primary agent's pre-task hash capture;
the two original configs also remain unchanged. Final input/source hashes pass
for all three runs, and October 12 holdout labels remain absent and sealed.
EXP-008's **14** and EXP-010's **27** non-manifest artifacts reproduce byte for
byte. EXP-009's compact results and every report probability reproduce exactly;
refitted joblib bytes differ while parameters, thresholds and report predictions
agree. The full regression suite reports **57 passed, 1 skipped** (the existing
optional live instructor-sample test); `git diff --check` passes.

Detailed evidence and reproducible audit commands are in
[the verification report](../reports/EXP008_010_VERIFICATION.md) and
`outputs/verification/exp008_010_final_audit.json`. These checks accept the
configured development experiments, not independent future-performance claims or
production promotion. The controlled EXP-010 selected policy's eligible score is
**0.465391**, **+0.010714** versus persistence; higher unselected blend scores
remain exploratory comparisons.

## 22. EXP-011: stronger tabular learners — October 16, 2026

### 22.1 Authorization and protocol registration

**Status: Ongoing; no candidate result accepted at registration.** The user
authorized implementation using GPT-6 Luna subagents at Max reasoning effort,
with the primary agent orchestrating and independently evaluating the result.
The controlling protocol is [EXP011_EXECUTION_PLAN.md](EXP011_EXECUTION_PLAN.md).

Before candidate execution, the primary agent captured SHA256 values for 30
archived dataset/model/config/evaluator files and the eight raw instructor
Parquet files in `outputs/verification/exp011_parent_before_hashes.json` and
`exp011_parent_source_before_hashes.json`. The existing suite passed **57 tests,
1 skipped**, before EXP-011 changes. On Windows the Conda environment's
`Library/bin` must be on PATH; invoking its Python without activation produced a
SciPy DLL exception. The activated retry passed in 16.70 seconds.

**Prespecified tracks:** archival Track A retains the original 21 training weeks,
August 3/10/17 calibration and August 31/September 7/14 reporting. Track B is a
separately named 18-week rolling-refit and threshold-transfer study with
`C=S-4h`, `A=S-3h`, seven-day targets and the existing seven-day label allowance.
Reporting targets: August 24/31 and September 7/14; calibration uses each origin's
`R-35d`, `R-28d`, `R-21d`. Selection uses earlier June 8 calibration predictions
(12 mature fitting weeks) and June 29 selection predictions (15 mature fitting
weeks), with the latter labels mature July 13, before the earliest outer
calibration decision on July 19 at 21:00 Taiwan. The shorter inner fitting
history is explicit. All compared learners share samples and timing within a
track; cross-track changes are not pure learner effects.

The six proposed treatments are two each for LightGBM, CatBoost and XGBoost,
as specified in the execution plan. Exact versions, resolved parameters, fold
and config hashes must be stored before their candidate report results are
exposed. Optional extended tuning and recency studies are not required to force
a winner. October 12 holdout labels remain prohibited throughout this study.

Actual execution, failures, metrics and independent verification will be appended
below; this registration is not evidence of completed candidate fits.

### 22.2 Pre-fit clock amendment and implementation checks

Before any full candidate fit or reporting result, the independent reviewer
requested a strict selection-before-fit ordering. Selection is now registered at
**July 19, 20:00 Taiwan**, one hour before the earliest outer calibration fit at
July 19, 21:00. The earlier June 29 selection labels mature July 13, so this
changes no allowed examples, candidate parameters or score population. The
original configuration SHA and amended SHA are retained in
`outputs/verification/exp011_parent_registered_config.json` and
`exp011_parent_config_amendment_01.json`. Equality between selection and the first
fit clock is no longer admitted.

The primary agent's first full suite after implementation returned **82 passed,
1 skipped in 16.48 seconds**. Subsequent clock/audit repairs still require their
own checks before accepting a final run. The new optional libraries were verified
in `.deps-exp011`; wheel versions/hashes and metadata are preserved in
`configs/stronger_tabular_dependencies.json`. The base Conda environment was not
modified.

### 22.3 Executed bounded benchmark and observed decision

The fresh run `experiments/exp011_stronger_tabular/runs/run-exp011-core-v1`
completed with exit code **0** at October 16, 23:57:54 Taiwan. It saved **60
six-target classifier bundles**: eight archival, 16 earlier inner fits and 36
outer fits across nine unique calibration/report starts. Persistence is a
separate deterministic control. The manifest records **1,116,000 cache rows**,
**7,079,724 long reporting rows**, **162 official weekly checks**, **917.53
seconds fitting** and **1,046.54 seconds total**. All learners use the exact
original 59 finite numeric F0 columns; the anchored eligibility field is metadata.

These are **pooled user-week combined F1** on the legacy eligible proxy. The
mean-week column is a different aggregation and is reported separately.

| Track A: fixed 21-week archival fit | Pooled score | Mean weekly score |
|---|---:|---:|
| HGB | 0.398767 | 0.373411 |
| Logistic | 0.381531 | 0.359545 |
| LightGBM, 15 leaves | 0.381426 | 0.357895 |
| LightGBM, 31 leaves | 0.393111 | 0.369001 |
| CatBoost, depth 4 | 0.387416 | 0.362142 |
| CatBoost, depth 6 | 0.389197 | 0.362897 |
| XGBoost, depth 3 | 0.386561 | 0.363178 |
| XGBoost, depth 5 | 0.380921 | 0.356877 |
| Persistence | 0.454677 | 0.419356 |
| Calibrator-selected EXP-010 push hybrid (rolling calibration) | 0.465391 | 0.430597 |

The hybrid is operational context under its separately recorded rolling policy;
it is not another fixed-calibration learner. Every alternative trails HGB on
all three matched archival report weeks. Exact refitted HGB/logistic controls
match archived probabilities with maximum absolute difference **0.0**.

Earlier June selection chose **CatBoost depth 4** and **XGBoost depth 3** for
Track B; LightGBM 31 leaves was its family's representative but was not selected
for outer fitting. Inner selection scores were respectively **0.506908**,
**0.506600**, and **0.506302**. No reporting leaderboard selected these families.

| Track B: four-hour gap, common rolling 18-week fit | Pooled score | Mean weekly score |
|---|---:|---:|
| HGB | 0.405331 | 0.381438 |
| Logistic | 0.366941 | 0.342622 |
| CatBoost, depth 4 | 0.398286 | 0.371524 |
| XGBoost, depth 3 | 0.402824 | 0.377995 |
| Persistence | 0.445092 | 0.399637 |
| Calibrator-selected EXP-010 push hybrid | 0.439463 | 0.398835 |

| Track B report start (Taiwan) | CatBoost delta versus HGB | XGBoost delta versus HGB |
|---|---:|---:|
| August 24 | -0.019214 | -0.002605 |
| August 31 | -0.012953 | -0.009566 |
| September 7 | -0.016328 | -0.016358 |
| September 14 | +0.008842 | +0.014758 |

The registered median improvement gate is +0.005 with at least three of four
nonnegative weeks. Both have only **one of four**, with median deltas
**-0.014641 CatBoost / -0.006086 XGBoost**. Both also trigger the target-review
guard: CatBoost loses >0.02 target F1 on at least two origins for open_pr,
open_issue and pr_review_comment; XGBoost does so for issue_comment. The complete
per-origin TP/FP/FN/TN/support tables are preserved rather than concealing sparse
targets. Pooled target-positive supports are **3,122 / 1,420 / 2,484 / 2,031 /
1,439 / 5,792** in registered target order. All six archival candidates fail
the same registered consistency screen and target guard.

Sensitivity populations agree on the learned ranking. Track B pooled
all-cohort scores are **0.393569 HGB / 0.390898 XGBoost / 0.386496 CatBoost**;
anchored-proxy scores are **0.400018 / 0.397598 / 0.392792**. The higher historical
hybrid score does not transfer here: the new rolling hybrid trails persistence
by **0.005629** pooled combined F1. Cross-track deltas also change history,
cutoff, calibration and report dates and cannot be treated as algorithm effects.

**Decision: retain the fixed HGB control for EXP-012.** This bounded F0 study
does not support replacing it with any tested alternative or expanding the
hyperparameter search. The optional second-stage configurations, weighting and
recency sensitivity remain `not_run` by the registered core scope. An isolated
target gain may inform a later preregistered EXP-014 ablation; it does not select
a new hybrid here. For cost context, the 11 comparable Track B fits took
**101.82 seconds HGB / 115.36 seconds XGBoost / 362.12 seconds CatBoost**. Fit-log
RSS values are before/after process measurements, not measured peak memory.

An initial no-output preflight rejected a wrong source-SHA lookup. The first
partial candidate attempt later stopped when the hybrid gate rejected an
immature calibration week. That directory is preserved as
`run-exp011-core-v1-attempt1`; it is not accepted evidence. The corrected exact
latest-three-mature schedule has a regression test, and the entire fresh run
completed with source/config hashes frozen. The full corrected suite passed
**85 tests, 1 skipped in 18.31 seconds** before this run. The skip remains the
existing optional live instructor-sample test.

Evidence: final run `manifest.json`, `fit_log.csv`, `inner_selection.csv`,
`screening_rules.csv`, `metrics_by_origin.csv`, `metrics_pooled.csv`,
`per_target_metrics.csv`, and `per_target_metrics_pooled.csv`. Independent final
acceptance is recorded separately below before changing the plan status.

### 22.4 Primary-agent acceptance — October 17, 2026

**Status: Completed development benchmark.** The primary agent reviewed the Luna
Max implementation, required corrections, and independently executed
`reports/verify_exp011.py` against the completed fresh run (exit **0**).
`outputs/verification/exp011_parent_final_audit.json` records:

- **38 original files unchanged** and **60 run input/source/code hashes** valid.
- **7,079,724 reporting rows** checked for unique stable keys, binary labels and
  decisions, finite probabilities and consistent population views.
- **162 direct instructor weekly scores**, **630 metric scalars**, and **5,040
  target confusion-count values** independently reproduced. Weekly and pooled
  scores were recomputed separately from saved decisions; AP/AUC/Brier were
  checked against probabilities where defined.
- **21 report truth/population slices** and **18 unique cache origins** checked
  directly against authorized source events, including labels driving inner
  model selection and calibration. Eligibility groups were reconstructed from
  observed records and the declared anchor clock.
- All **29 gap feature panels** rebuilt exactly across every cohort user and
  all 59 columns; the complete **12 gap fold schedules** and 21/12/15/18-week
  training widths checked. Selection precedes the first outer calibration fit.
- **60 saved native model bundles**, their hashes and **5,832,000 probability
  values** verified against saved caches, with exact equality. Persistence
  caches matched the original prior-seven-day feature flags.
- **32 joint threshold searches**, June family/variant/top-two selection,
  **91 hybrid candidate scores**, **1,456 threshold-grid objectives**, selected
  hybrid report decisions, and all **eight screening rows** independently
  reconstructed. Exact calibration windows satisfy label maturity.

The separate `reports/verify_exp011_controls.py` command also exited **0** on
the final v1 path, retaining exact archived-control parity in
`outputs/verification/exp011_parent_control_final_audit.json`. Independent
threshold-search code previously matched the stable implementation on 21
randomized/empty-target cases. Full regression is **85 passed, 1 skipped**;
`git diff --check` passes. The final report is
[EXP011_VERIFICATION.md](../reports/EXP011_VERIFICATION.md); run-local
`findings.md` and `tests-and-validation.txt` retain execution details.

**Accepted next action: EXP-012 with HGB fixed as the learned control**, alongside
persistence and a separately evaluated calibrator-selected hybrid reference.
There is no evidence here to promote any tested alternative as an all-six
replacement. This is a negative bounded-screen result, not evidence that these
families cannot improve under other representations or future registered
settings. The optional second-stage search and recency arm are explicitly
outside the completed core scope.

No October 12 holdout labels were constructed or read. The target week ends
October 19 and the configured allowance ends October 26; its one-shot protocol
remains a separate gate. Repeatedly inspected development dates, a single source
release, retrospective source visibility, proxy rosters, shorter inner fitting
history and only three/four report origins limit generalization. No independent
November performance or deployment claim follows from this completion.

## 23. EXP-012: temporal rhythm and weekly-state ablations — October 17, 2026

### 23.1 Authorization, implementation and prospective registration

**Status: Ongoing; results not yet accepted.** The user requested planning first,
then authorized implementation with **Luna at Max reasoning effort as the main
implementers**, with the primary agent orchestrating and evaluating. Work was
split into temporal features, replay/preflight, execution and an independent
saved-run auditor. The controlling protocol is
[EXP012_EXECUTION_PLAN.md](EXP012_EXECUTION_PLAN.md).

The bounded core preserves HGB and all 59 F0 predictors, adding separately the
24 rhythm and 24 state columns. An earlier inner screen controls the optional
union; fresh removal fits are conditional on selection of that union. The
shuffle diagnostic changes only added training columns within individual weeks.
Broader temporal/cross-target/regime groups are outside this core run.

Before candidate fitting, the primary agent checked the 38 historical locks and
captured 122 historical/source/accepted-run file hashes in
`outputs/verification/exp012_parent_before_hashes.json`. All eight raw source
tables have unique, non-null primary keys. Current archived HGB/logistic
probabilities reproduce exactly. Separate direct calculations matched **6,144
new feature cells**; a full-user reporting-origin smoke check matched **531,000
F0 cells**, all six labels and the anchored mask against EXP-011.

The score-free preflight completed in
`experiments/exp012_temporal_features/runs/exp012-protocol-freeze-v1/`:

- **29 gap origins / 261,000 rows** and **27 archival origins / 243,000 rows**;
  each origin contains the fixed 9,000-user cohort.
- **12 unique fit schedules**: two earlier 12/15-week fits, nine 18-week primary
  fits and one fixed 21-week archival fit. The primary agent independently
  reconstructed every maturity window and checked all 14 code and 15 payload
  hashes before registration.
- Registered at **2026-10-17 22:31:30 Asia/Taipei**, before observing EXP-012
  candidate scores, in `outputs/verification/exp012_parent_protocol_registration.json`.
- Frozen `run_config.json` payload SHA-256:
  `fa6ea277d550c7c6ab3dc4671af979aa0eb69fdc7a92d8b9df3d38106d3f4553`.
- Protocol-freeze SHA-256:
  `ab2258ff7ded8a460d4bdd100846ee0095d1c3d79deeb72e7e857bd7f37d6ec7`.
- Execution-code SHA-256:
  `27d52dde8c71ec239f425c3ee4019e088aecf5ac3d78ea4f46b7b8994e737228`.

The source remains instructor SHA
`506d860693df60a698b68aa63a8edb4a984c8e1b`. The runtime records Python **3.11.17**,
NumPy **2.4.6**, pandas **2.3.3**, scikit-learn **1.9.1**, PyArrow **23.0.1**,
four CPU threads, a two-hour model-execution cap and a five-GiB process working
allowance. No new learner dependencies were installed.

### 23.2 Pre-execution checks and repairs

The primary agent's final pre-execution regression is **112 passed, 1 skipped**,
19.80 seconds, saved in `outputs/verification/exp012_parent_regression_03.txt`.
The skip remains the opt-in instructor-sample test. A failed earlier attempt
retains its traceback in `exp012_parent_regression_01.txt`: old pytest temporary
directories denied writes, and one expected cache-key schema was stale after
the permutation digest was added. A fresh workspace basetemp and disabled
cache provider resolved the local permission problem without changing the data.

Review corrections before fitting included timestamp-unit normalization;
explicit float32-recency/float64-gap arithmetic; classifying inactive streak as
a recency transformation; holdout/date guards; actual origin-valid OOF
calibration rather than a fit-once test seam; reference population coverage;
stable policy IDs for multiweek pooling; typed Parquet output schemas; failure
checkpoint argument handling; and logging the shuffle/attempted-union evidence.

Independent threshold-search code matched 24 randomized/empty-target cases.
Parent engine checks independently verified pooled counts versus weekly means,
the output schema, seed-drift rejection, time-cap rejection and checkpoint
writing. These are implementation checks, not predictive results.

The standalone auditor was corrected against the frozen fold schema. Its raw
gap oracle also originally included every record tied at the tenth timestamp;
the registered feature uses exactly the last ten records, retaining ties within
those ten. Stable sorting followed by `tail(10)` corrected the oracle without
relaxing the 1e-6 comparison or changing the feature builder. Historical panel
matching was corrected to join both track and origin. The independent score-free
audit then passed: 56 panels, 690,816 sampled temporal cells, 15,399,000 F0
control cells, 56 source-rebuilt label origins and every frozen payload hash.
The parent separately checked the median-gap values for 257 users at all 56
origins; no discrepancy remained.

The first real execution, `run-exp012-core-v1`, stopped after its twelve F0
bundles at the historical threshold parity gate. EXP-011 stores one archival
threshold policy, reused across its three reporting weeks; the new gate wrongly
looked for a distinct policy row at each week. No temporal candidate was fitted
or scored. The failed directory and traceback remain preserved. Repairing this
lookup requires a new code freeze and a new run directory; it does not justify
changing the registered candidate matrix or expected control values.

For clarity, the source configuration file has SHA-256
`41cee5c2a0fdc480566f05f8fb7b72051b01fb6511df81107f33a65a77b1c494`.
The `fa6ea277...4553` digest above identifies the serialized frozen
`run_config.json`, not the source file; both are bound separately by the freeze.
The original registration JSON's `config_sha256` field used the payload digest;
this terminology correction preserves its original bytes and timestamp.

The repaired execution gate passed a read-only integration recheck against the
first run's saved F0 cache: **17 exact probability origins and seven exact
threshold report comparisons**. The repair also preserves persistence/hybrid
model IDs in reference threshold rows. Two focused regressions were added;
the updated full suite is **114 passed, 1 skipped** in 16.54 seconds
(`outputs/verification/exp012_parent_regression_04.txt`).

The second score-free freeze, `exp012-protocol-freeze-v2`, was registered at
**2026-10-17 22:48:06 Asia/Taipei**, before temporal candidate fitting. Every
feature, fold, inventory, reference and frozen-config payload is byte-identical
to v1; only the environment record differs. The only changed runtime code is
`execution.py`. The registration is
`outputs/verification/exp012_parent_protocol_registration_v2.json`:

- Freeze SHA-256: `4114f35036586a5013d8e7dc75246cd21a33c322cf0b9a1864f695c541868018`.
- Execution SHA-256: `cdc001ce25311f547a3afcad6bce0ac72491950dd7bb26595d55b72d0aec7a31`.
- Source-config and frozen-config digests remain the distinct values above.

The registered execution writes to a fresh `run-exp012-core-v2` directory;
the failed v1 remains unchanged. The prospective gates and trial budget are
unchanged by these implementation repairs.
No October 12 holdout labels have been constructed or read. Completion requires
the actual bounded run and saved-output acceptance below.

### 23.3 Executed candidate results

The v2 execution passed the exact F0 gate on **17 prediction origins and seven
report-threshold vectors** before temporal fitting. Its earlier June 29
legacy-proxy screen (8,602 reporting users, June 8 calibration) then recorded:

| Arm | Model columns | Inner combined F1 | Delta vs F0 | Screen |
|---|---:|---:|---:|---|
| F0 | 59 | 0.501830073 | — | Control |
| F0 + rhythm (R) | 83 | 0.506170636 | +0.004340563 | Fails minimum +0.005 gain; issue-comment F1 also drops 0.020860495 |
| F0 + weekly state (S) | 83 | 0.514320619 | +0.012490546 | Passes gain and target guard |

The recorded earlier choice is **S**, with no use of outer metrics. The union
is `not_run` because R failed its individual gate. Fresh union-removal fits
are therefore inapplicable. The registered shuffle diagnostic uses S, with
only its added training columns permuted within each earlier training week.
The outer/archival single-group comparisons remain prespecified diagnostic
work for both R and S. The reporting screen and independent acceptance are
still pending at this entry; an inner gain alone does not promote S.

The completed v2 execution subsequently saved **38 six-target bundles / 228
binary target models**, all with fixed HGB settings. No union or removal fits
were triggered. Execution completed in **829.718 seconds**, below the two-hour
cap, with peak sampled fit RSS **1,124,749,312 bytes** and completion RSS
**1,049,849,856 bytes**, below the five-GiB allowance. Four threads were used.
Fit/inference costs by arm and the one-panel build profile are recorded in
[EXP012_VERIFICATION.md](../reports/EXP012_VERIFICATION.md).

**Primary reporting result: the earlier-selected state arm fails the registered
screen.** Matched legacy-proxy combined F1:

| Taiwan report week | F0 | Rhythm | State | State delta |
|---|---:|---:|---:|---:|
| Aug 24 | 0.400570022 | 0.385051848 | 0.399092989 | −0.001477033 |
| Aug 31 | 0.206271793 | 0.202072458 | 0.204634747 | −0.001637046 |
| Sep 7 | 0.422264899 | 0.424945543 | 0.428444634 | +0.006179735 |
| Sep 14 | 0.496643813 | 0.485689369 | 0.496977978 | +0.000334166 |

State has median weekly delta **−0.000571434** versus the required **+0.005**,
and only **2/4 nonnegative origins** versus the required three. No target loses
more than 0.02 on two origins, but that guard alone is insufficient. The
prospective fallback is **F0**; no replacement arm is selected from outer scores.

| Legacy-proxy pooled combined F1 | Gap / 4 reports | Archive / 3 reports |
|---|---:|---:|
| HGB F0 | 0.405330950 | 0.398766886 |
| HGB F0 + rhythm | 0.396842760 | 0.389976008 |
| HGB F0 + state | 0.406467323 | 0.380150013 |
| Persistence reference | 0.445091535 | 0.454677348 |
| Accepted EXP-010 hybrid reference | 0.439462705 | 0.465391270 |

The **+0.001136373** pooled state gain on the gap track does not override its
weekly stability failure; archival state loses **0.018616874**. On all-cohort
and roster-anchored gap populations state scores **0.394750455 / 0.401207687**,
versus F0 **0.393568610 / 0.400017762**. With fixed F0 thresholds, legacy state
scores **0.406919980**. These sensitivity diagnostics do not change the gate.
Neither new group exceeds persistence on either track.

The earlier shuffled-state diagnostic scores **0.505120700** on the primary
legacy proxy, below unshuffled state **0.514320619**, with F0 **0.501830073**.
It was not used for selection. All attempted arms, thresholds, AP/AUC,
Brier/log loss, per-target support/confusion counts and population views remain
in the run's CSV/Parquet artifacts. The verification report gives the six-target
state counts on **23,839** primary legacy-proxy user-weeks and distinguishes
pooled confusion metrics from mean weekly scores.

The primary agent independently recomputed **192 weekly metric groups and 48
pooled groups**, confirmed the screen failure and rehashed all **122** protected
files (`outputs/verification/exp012_parent_result_review.json`). Final manifest
SHA-256 is `bc988725fe726fd5aaa747f53f9987b7319ff01c7aa508faf79ddb545392ebc5`.
The full independent raw-source/native-model/threshold/prediction audit remains
the last acceptance step; status stays Ongoing until its recorded pass.

### 23.4 Independent acceptance — October 18, 2026

**EXP-012's registered rhythm/state core is Completed.** The full independent
CLI audit exited successfully, with `status: passed`, in
`outputs/verification/exp012_independent_final_audit.json`; its terminal trace
is `exp012_independent_final_audit_07.txt`. It authenticates the same final
manifest SHA-256 reported above and independently checked:

- **215 manifest/provenance hashes**, 15 frozen preflight payload copies,
  eight raw source files, 130 parent-lock rows and all 122 protected files.
- Every one of **38 saved bundles and fit-cache bases**, including reconstructed
  training-row digests and **2,862,000 native model probability values**.
- **64 threshold vectors**, including **25 independently repeated joint searches**
  from 603,000 calibration-cache rows; earlier selection, shuffle maps and the
  complete registered output schedule.
- **8,628,480 prediction rows**, **192 official weekly scores**, 48 pooled groups,
  5,664 metric scalars and 7,200 target-confusion checks. Raw-source truth was
  rebuilt at eight scored track/origin slices and 17 cached track/origin slices.
- The recorded state-arm reporting failure, all per-target deltas and the
  prohibition on using outer scores for earlier feature choice.

Final independent-auditor source SHA-256:
`307990929e482b052a8103b1195a5b25d4d7d811a64a6e3bbc210d498ef9b6e5`.
The primary agent corrected final timestamp-key/JSON-array serialization issues
in the evaluator after Luna's implementation. Earlier auditor failures and the
interrupted sixth attempt are retained or recorded separately; no tolerance,
feature definition, source input, fitted object or experiment result was changed
to obtain this pass. The finalization checks are in
`outputs/verification/exp012_parent_auditor_finalization_checks.json`.

**Accepted next control: HGB with the original 59 F0 columns, with no added
temporal columns.** State's earlier eligibility did not survive the prospective
reporting stability gate; rhythm failed earlier eligibility. The machine-readable
decision is `outputs/verification/exp012_feature_decision.json`. The next main
milestone is **EXP-013 relational ablations** against this fixed learned control,
with persistence and the accepted origin-valid hybrid retained as references.
No EXP-013 fits were started by this acceptance.

Broader temporal/momentum/cross-target/regime groups remain explicitly deferred
follow-ups. This completion satisfies the registered core and the main plan's
minimum rhythm/state acceptance criteria; it does not claim the full feature
catalog was implemented. Four reused development weeks, a single source release,
shorter inner histories, proxy populations and retrospective source visibility
limit generalization. No October 12 holdout labels were constructed or read,
and no independent November performance is established. The detailed results,
costs, repairs and limitations are in
[EXP012_VERIFICATION.md](../reports/EXP012_VERIFICATION.md).

## 24. EXP-013: bounded repository relations — October 18, 2026

### 24.1 Authorization and implementation start

The user approved implementing `docs/EXP013_EXECUTION_PLAN.md`, with **Luna Max
subagents as the main implementers** and the primary agent responsible for
orchestration and evaluation. Three `gpt-6-luna` agents at `max` reasoning were
assigned the feature builder/tests, frozen replay/configuration, and independent
verifier/tests with separate file ownership. This is implementation initiation,
not experiment completion or a measured result.

The bounded study retains **HGB and the original 59 F0 columns**, tests nine
repository-self additions from timestamped pushes/comments, and conditionally
tests seven cohort-neighbor additions and their union under the registered
earlier selection rules. Broader workflows/graphs remain deferred.

Before implementation, the primary agent verified existing source/archive locks
and captured `outputs/verification/exp013_parent_before_hashes.json`: **288
protected files**, including existing shared code/config/tests, the accepted
EXP-012 run and freeze, historical controls, feature decision and audit. No
existing protected files may change to obtain a passing comparison. Actual
protocol/config/code hashes, tests, execution and results will be recorded in
subsequent entries. **Status: Ongoing; no EXP-013 fits yet.** The October 12
holdout remains sealed, with no labels created or read.

## 25. EXP-014a: recent-regime rolling evaluation and calibrated decision layer — October 18, 2026

### 25.1 Scope and naming

This is a **sub-experiment (EXP-014a)**, not the registered EXP-014 in §16.4 and §22. It tests a narrow hypothesis from the EXP-006–010 record: the old threshold policy failed because its thresholds came from a period with 5–10x lower base rates. The test trains on recent data, recalibrates probabilities, and chooses 0/1 decisions by maximizing expected official score. It uses only the **original 59 F0 columns with HGB** (plus, in one arm, 12 origin-constant cohort regime columns). EXP-013 relational features are not used.

### 25.2 Frozen protocol and execution

- **Report origins:** Thursdays 00:00 Taiwan (Wednesdays 16:00 UTC) on 2026-09-02, 09-09, 09-16, 09-23 and 09-30. The 09-30 origin is **provisional**, because issue and PR rows can still arrive late. Report labels use `label_as_of = latest_observed()`, which is 2026-10-13 12:59:29 UTC.
- **Clocks:** feature cutoff = origin − 4 h; decision time A = origin − 3 h. Training and calibration labels use `label_as_of = A`. Training origins are daily at 16:00 UTC with s ≥ S − 63 d and s + 7 d ≤ A (56-day window). Calibration origins are S − 21 d and S − 14 d. M_cal trains only on s + 7 d ≤ (S − 21 d) − 3 h.
- **Population:** roster-anchored proxy. The anchor is origin − 14 d − 4 h. A cohort user qualifies with at least 5 visible rows in the 90 days before the anchor, summed over the six target tables. Population sizes: 7,417; 6,382; 5,862; 5,497; 5,178.
- **Models and arms:** HGB with the configured hyperparameters (80 iterations, learning rate 0.07, 15 leaves, minimum leaf 75, L2 3.0, seed 42), fit separately per target. Arms are F0 (59 columns) and F0+regime (+12 columns).
- **Policies:** P_persist7; P_persistL (lookback from {7, 14, 21, 28} chosen on the calibration origins); P_thr_raw (the existing `tune_thresholds` run on M_cal's raw calibration probabilities, applied to M_final); P_expscore (per-target Platt scaling, then label-free expected-score coordinate ascent for k); P_ratematch (k = round of the sum of calibrated q).
- **Artifacts:** config `configs/recent_regime.json` (SHA-256 `e752561f60b7ead880ce931a60508522179b375ccde56685fff34ba96c66927e`); code `src/decision_policies.py`, `experiments/exp014_recent_regime/run.py`, `tests/test_decision_policies.py`; run `experiments/exp014_recent_regime/runs/exp014-recent-regime-v1/`.
- **Data and commit:** assignment2 snapshot `506d860693df60a698b68aa63a8edb4a984c8e1b`; repository commit `7cff786911570598007681901b4bd4c9e5272d70` with 56 uncommitted entries at run time. Runtime was about 377 s, using base Python 3.12 (numpy 2.2.5, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 22.0.0).

### 25.3 Verification

- New tests: **35 passed** in `tests/test_decision_policies.py`. They cover the expected-score optimizer against brute force, rate matching, Platt scaling, the population boundary and the no-leakage rules.
- Full suite: **146 passed, 2 skipped, 5 collection errors.** The errors are EXP-011/012/013 test modules that import `psutil`, which is absent from the base Python. They are not caused by this work.
- Official parity: 40 origin-policy checks against the instructor's `evaluate.py`, maximum absolute difference **0.0**.
- Leakage: training and calibration bounds are asserted at runtime and in tests. P_persist7 is asserted equal to the `active_prior_7d` feature.

### 25.4 Measured results

Pooled over origins (confusion counts pooled, then scored):

| policy | recent3 score | all5 score |
|---|---|---|
| **P_persist7** | **0.5075** | **0.4763** |
| P_persistL | 0.4673 | 0.4629 |
| P_thr_raw_F0 | 0.3227 | 0.3442 |
| P_thr_raw_F0R | 0.3836 | 0.3956 |
| P_expscore_F0 | 0.3692 | 0.3808 |
| P_expscore_F0R | 0.4302 | 0.4202 |
| P_ratematch_F0 | 0.3883 | 0.3934 |
| P_ratematch_F0R | 0.4432 | 0.4187 |

Paired cluster bootstrap over UserID (1000 replicates, seed 42), delta versus P_persist7, 10th to 90th percentile: every learned policy is negative in both pools, with P(delta > 0) = 0. Mean deltas: in recent3 the closest learned policy is P_persistL (−0.040), then P_ratematch_F0R (−0.064); in all5 the closest is P_persistL (−0.013), and the furthest is P_thr_raw_F0 (−0.132). Per-origin scores are in `metrics_by_origin.csv`.

Positive rates (report origin / final training window, mean over six targets): 09-02 0.107 / 0.062; 09-09 0.153 / 0.060; 09-16 0.127 / 0.065; 09-23 0.121 / 0.073; 09-30 0.134 / 0.079. The early-origin training windows did **not** have lower base rates than later windows, so the "dip-period" explanation is not supported by these measurements.

### 25.5 Decision and limits

- **Decision:** no recalibrated, recent-trained or expected-score policy beats the 7-day persistence rule in either pool. Persistence is retained as the reference. No policy is promoted to the registered EXP-014 or to the November submission candidate.
- **Interpretation:** the measured losses are consistent with a base-rate shift between the training windows and the report origins, but the run does not isolate it as the cause.
- **Limits:** five origins, one of them provisional. These are development results. The October 12 holdout was not used, and no labels after 2026-10-13 were read.
- **Deviations from the brief:**
  - A 14-day training window is infeasible: the calibration slice is empty (checked: 0 origins at 14 days; 7 at 28 days). The smoke run used 28 days; the full run uses the configured 56 days. The smoke artifact remains at `runs/exp014-smoke-20260916-w28/`.
  - The project Conda environment `nycu-bda-asg2` crashes natively (`0xc06d007f`) on numpy linear-algebra calls on this machine. All runs used base Python. The Platt fit is therefore solved by a numpy Newton-Raphson method with a 1e-8 ridge, not scikit-learn.
  - The population anchor (origin − 14 d − 4 h) is applied to training and calibration origins as well as report origins.
- **Registered EXP-014 status:** unchanged, still Pending. It still requires the 011–013 candidates and the frozen OOF policy described in §22.

## 26. Unattended §29 session: EXP-014b Stage 1, EXP-016–021 screens, EXP-027 harness — October 18, 2026

Session `s29-unattended-20261018`. Full record: `reports/unattended_sessions/s29-unattended-20261018/`
(`PREFLIGHT.md`, `SESSION_REPORT.md`, `PROTOCOL_DISCREPANCIES.md`, `TRIAL_LEDGER.csv`,
`CANDIDATE_MANIFEST_confirm_v0.json`). Data release `506d860…` only (latest row 2026-10-13 12:59:29 UTC). Repository
HEAD `7cff786…` with uncommitted work. The Oct 12 holdout stayed sealed.

### 26.1 Registrations (made before the corresponding numbers existed)

- EXP-014b frozen by appended §10 (pre-appendix prereg SHA-256 `644aa6d4…`; config `configs/exp014b_rawp_topk.json`
  `79053a4a…`) before any C3 number. C1/C2 remain contaminated (prereg §2).
- EXP-016–019: `docs/exp/EXP016_019_PREREGISTRATION.md` (`00433d80…`), config `configs/phase_a_016_019.json`.
  Exploratory development screen. User-approved history rule: parameters are selected on {S − 14d, S − 21d}, with
  labels rebuilt as of A_S = S − 3h. History predictions for 08-19/08-26 came from a frozen-code EXP-014a replay
  (`experiments/exp014_recent_regime/runs/exp014-history-0819-0902-s29/`), whose 09-02 `p_raw` equals the saved run
  exactly.
- EXP-020/021: `docs/exp/EXP020_021_PREREGISTRATION.md` (`41fc84f3…`), registered after the EXP-016 screen was seen
  (disclosed).

### 26.2 Measured (all4 = 09-09/16/23/30, 09-30 provisional; Δ vs P_persist7 0.5166; gate §29.2)

| candidate | pooled Δ | weekly wins | LOO min | status |
|---|---|---|---|---|
| P021_blend50 (0.5·rank HGB p + 0.5·rank EW τ=7, k = persistence count) | +0.0288 | 4/4 | +0.0263 | screen_pass |
| P016_a100 (k = persistence count, top-k by HGB p) | +0.0205 | 4/4 | +0.0180 | screen_pass |
| P020_multi (persist7 OR active in ≥ 2 of 4 weeks) | +0.0092 | 4/4 | +0.0070 | screen_pass |
| EXP-014b C1 raw-p expected-score | +0.0064 | 3/4 | −0.0008 | Stage 1 pass, **nominated for Stage 2** (contaminated) |
| EXP-014b C2 / C3 | +0.0008 / −0.0840 | 2 / 0 | — | dropped |
| P018_sel, P017_drop, P020_aligned | +0.0018 / −0.0085 / −0.0006 | — | — | rejected |
| P019_sel (per-target selection) | +0.0021 | 3/4 | −0.0029 | screen_pass, contaminated and dominated; not carried forward |

All 19 rows, with per-origin deltas, are in `TRIAL_LEDGER.csv`. Checks: official-evaluator parity 0.0 on every
origin × policy; EXP-014b P1 `p_raw` parity 0.0 at 5/5 origins; end-to-end negative controls (36,550 permuted report
labels) left every decision and selection unchanged for Phase A, Phase B and EXP-019. Unit tests: 11 + 4 + 22 + 23
pass. Independent audit of the Phase A screen (own numpy code, labels rebuilt from the tables): every protocol check PASS, including parity, gate reproduction, chronology and label isolation. It also found that the §29.2 gate does not control for set size. Under label permutation, size-increasing candidates pass about 100% of the time and size-neutral ones about 27%. Follow-up: the size-neutral P021_blend50, P016_a100 and P021_ew7 have permutation p ≈ 0.001. Size-increasing candidates are not distinguishable from a 'predict more' effect by permutation, but their extra users beat size-matched random extras (p ≈ 0.002). Details: `SESSION_REPORT.md` §6 and `audit/`.

### 26.3 Decision and limits

- **No promotion.** These are development-screen results on September weeks that have been inspected many times.
  `P_persist7` remains the default. Confirm-tier plan v0 (registered before any confirm data existed): primary
  `P021_blend50`; secondaries `P016_a100` and EXP-014b C1; rule fallback `P020_multi`.
- **EXP-027:** the harness is built. Its 09-23 older-fold self-test reproduces all 11 policies on 32,982 of 32,982
  rows, and on 10-07 it correctly returns `NOT_MATURE` (maturity 2026-10-20T13:00Z > latest row). This is not
  confirmation evidence.
- **Open protocol issue D3:** the 10-07 and 10-14 confirm label windows together cover the entire Oct 12 holdout
  outcome week. Using them for selection makes that holdout non-independent. This needs a user decision before
  Oct 21.
- **EXP-013** is unchanged (Ongoing, 0 fits; auditor blockers not repaired). EXP-022 has a draft preregistration only.

### 26.4 D3 decision (user, 2026-10-18)

The user accepted the holdout overlap. Confirm runs at 10-07/10-14 will use `--holdout-overlap-ack`. The Oct 12
holdout (§24) is **downgraded to a non-independent sanity check**: it cannot be reported as independent evidence or
override the confirm tier. Maturity gates are unchanged, and no confirm run has happened. Record:
`reports/unattended_sessions/s29-unattended-20261018/PROTOCOL_DISCREPANCIES.md` (D3 decision).

## 27. EXP-022: training window and recency weighting (ranking, AP-selected) — October 18, 2026

Side session of `s29-unattended-20261018`. Development screen on September weeks only: **no promotion**. Data release
`506d860…` (latest row 2026-10-13 12:59:29 UTC); repository HEAD `7cff786…` with uncommitted work. No origin after
09-30 was touched, and the Oct 12 holdout was not read.

### 27.1 Registration

- Preregistration `docs/exp/EXP022_PREREGISTRATION_DRAFT.md`. Freeze appendix added 2026-10-18 13:40 UTC, before any
  EXP-022 fit; pre-appendix SHA-256 `5752bdd6…`, post-appendix `1516b749…`. Config `configs/exp022_ranking_window.json`
  (`f6a8b510…`) inherits `configs/recent_regime.json` (`e752561f…`) and pins the decision-layer code hashes.
- Arms (one change each vs REF = EXP-014a M_final, 56-day daily window, unweighted): W28/W84/W112 windows; H2/H4
  sample weights 0.5^(age_weeks/h), normalized to mean 1; THU = Thursday-grid origins only (8 of 56). `fit_bundle`
  has no weight argument, so `fit_weighted_bundle` in the runner builds the identical HGB and passes `sample_weight`.
  No existing file was edited.
- Code: `experiments/exp022_ranking_window/run.py` (`22f453c1…` at run start, reusing the EXP-014a `Workspace`/`_stack`
  and the EXP-016 `screen_gate`); tests `tests/test_exp022_ranking_window.py`, 17 pass (and Phase A/B tests 15 pass).
- Smoke run `runs/exp022-smoke-0902-0909/` (09-02 + 09-09, 5 null reps) was made **after** the freeze, to test the
  code path. It showed 09-09 numbers, which could not change the frozen rule.

### 27.2 Checks

REF reproduces the saved EXP-014a `p_raw_F0` at 5/5 origins (max |Δ| = 0.0). Under REF, `P016_a100` and
`P021_blend50` equal the saved EXP-020/021 decisions exactly. P_persist7, the population and the report labels equal
EXP-014a. Official-evaluator parity is 0.0 on all 75 origin × policy pairs (90 parity rows in total). Wall time 754 s.

### 27.3 Measured — `experiments/exp022_ranking_window/runs/exp022-ranking-window-v1/`

Primary metric: mean per-target AP; all4 = 09-09/16/23/30 (09-30 provisional). The bootstrap is a paired
UserID-cluster bootstrap (500 reps), reported only.

| arm | all4 mean AP | Δ vs REF | wins vs REF | eligible | bootstrap Δ [95%] | 09-02 (continuity) |
|---|---|---|---|---|---|---|
| REF | 0.5312 | — | — | — | — | 0.4684 |
| **H2** | **0.5375** | **+0.0063** | **4/4** | **yes — selected** | +0.0064 [+0.0033, +0.0097] | 0.4619 |
| W28 | 0.5346 | +0.0034 | 4/4 | yes | +0.0035 [−0.0001, +0.0068] | 0.4638 |
| W112 | 0.5333 | +0.0021 | 3/4 | yes | +0.0021 [−0.0016, +0.0059] | 0.4359 |
| H4 | 0.5342 | +0.0030 | 2/4 | no | +0.0031 [+0.0006, +0.0057] | 0.4641 |
| W84 | 0.5311 | −0.0001 | 3/4 | no | −0.0000 [−0.0032, +0.0031] | 0.4736 |
| THU | 0.5192 | −0.0120 | 2/4 | no | −0.0116 [−0.0164, −0.0066] | 0.4594 |

H2's all4 AP gain is spread over five of six targets: pr_review_comment +0.013, open_pr +0.011, pr_review +0.011,
issue_comment +0.008 and push +0.002. open_issue is −0.007.

Official score (reported only, not used for selection; pooled all4; §29.2 gate vs P_persist7 0.5166):

| model × layer | score | Δ vs persist7 | wins | LOO min | gate | Δ vs same layer on REF | wins vs REF |
|---|---|---|---|---|---|---|---|
| REF × P021_blend50 | 0.5454 | +0.0288 | 4/4 | +0.0263 | screen_pass | — | — |
| **H2 × P021_blend50** | **0.5495** | **+0.0329** | 4/4 | +0.0298 | screen_pass | **+0.0041** | **4/4** |
| H4 × P021_blend50 | 0.5473 | +0.0307 | 4/4 | +0.0279 | screen_pass | +0.0019 | 4/4 |
| REF × P016_a100 | 0.5371 | +0.0205 | 4/4 | +0.0180 | screen_pass | — | — |
| H2 × P016_a100 | 0.5390 | +0.0224 | 4/4 | +0.0197 | screen_pass | +0.0019 | 3/4 |

All 14 arm × layer rows pass the §29.2 gate vs persistence (full table: `screen_gate.csv`, `findings.md`). The nulls
are computed against P_persist7, as in `audit/size_null_phase_b.py`. Every row has label-permutation p = 0.001 and
size-matched p = 0.002. The null gate pass rate is 0.27–0.30, the same as the audit's size-neutral rate. These nulls
show that each layer beats persistence. They **do not** test an arm against REF. The arm-vs-REF evidence is the AP
bootstrap above and the per-origin vs-REF deltas.

### 27.4 Decision and limits

- **Registered AP selection: H2** (2-week half-life recency weighting). It is eligible for freeze #1 (§29.5, Oct 25)
  consideration only. P_persist7 stays the default, and the confirm plan v0 is unchanged.
- The gains are small: +0.006 AP and +0.004 official score under P021_blend50, on weeks that have been inspected
  many times. On the 09-02 continuity week, H2 is **worse** than REF: AP −0.0065, and −0.014 / −0.002 score under
  P016 / P021. The 09-16 P016 delta is −0.001. H2 is a down-weighting of the same 56-day slice, so it can only be
  confirmed on fresh origins frozen after this run (10-14 / 10-21 if it is frozen by Oct 25).
- THU is clearly worse (−0.012 AP): the 7× larger daily-origin training set helps ranking. Long windows (W84/W112)
  do not help.
- Follow-ups that depend on this result: EXP-023 (capacity) and EXP-024 (short-horizon features) can use H2 as their
  base if the orchestrator accepts it.

## 28. EXP-015 bounded RDL pilot admitted; environment built — October 18, 2026

### 28.1 Decision D9 (user, 2026-10-18)

The user admitted a **bounded relational deep learning pilot** as EXP-015. This amends the plan's §29.3 exclusion of
EXP-015 for this pilot only and waives §23.1 gate 1 (stable EXP-013 relational gain). The label D9 continues the D1–D8
numbering of `reports/unattended_sessions/s29-unattended-20261018/PROTOCOL_DISCREPANCIES.md`. Rationale and scope:
plan §29.7. Reasons recorded:

- The assignment brief invites models that directly analyze relational databases, naming GNNs and RDL (Lecture 5).
- Presentation Q2/Q3 grade cross-table and temporal modeling.
- Under §29.2 a GNN can be screened as a ranker (all4 mean AP) through the frozen `P016_a100` / `P021_blend50` layers,
  as EXP-022 was.

**No EXP-015 fit or graph has been built. No label data was read for this entry.**

### 28.2 Draft preregistration

`docs/exp/EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md` (draft, not frozen). Arms:

- MLP0: node-only, F0 inputs.
- GNN1: 2-layer heterogeneous GraphSAGE on the per-origin as-of relational entity graph.
- GNN1T: GNN1 plus edge recency.

The EXP-014a/EXP-022 protocol is unchanged. An arm is eligible if it beats REF and MLP0 on ≥ 3/4 all4 origins and on
the average. The draft carries 15 open points with orchestrator draft resolutions. The freeze (config
`configs/exp015_rdl_pilot.json` with SHA-256 and a dated appendix) must precede any fit; it is planned for Oct 19.

### 28.3 Environment `asg2-rdl`

New conda env (python 3.11.17, everything else via pip), about 5.1 GB. Versions:

- torch 2.5.1+cu124 (CUDA runtime 12.4, cuDNN 9.1)
- torch_geometric 2.6.1, with pyg_lib 0.4.0, torch_scatter 2.1.2 and torch_sparse 0.6.18 (pt25cu124 wheels)
- numpy 2.2.5, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 22.0.0, scipy 1.17.1

These match base Python's numpy/pandas/scikit-learn/pyarrow versions. Spec: `configs/exp015_rdl_environment.yml`;
exact pins: `requirements-exp015.txt`.

Smoke test `experiments/exp015_rdl_pilot/env_smoke.py`, rerun independently by the orchestrator, passed all six checks:

- (a) numpy inv/eigh/lstsq: no 0xc06d007f crash, unlike `nycu-bda-asg2`
- (b) CUDA matmul on the RTX 3070 Ti Laptop GPU
- (c) heterogeneous `to_hetero` GraphSAGE forward/backward/Adam step on GPU
- (d) NeighborLoader through pyg_lib
- (e) temporal NeighborLoader (`time_attr`, `input_time`)
- (f) Users.parquet read, 45,590 rows

Peak GPU memory was 56 MiB. Carry-forward rules:

- Run the env's python with `-s`, because the per-user site-packages holds unrelated packages.
- Sample from a CPU-resident graph and move batches to the GPU; sampling a CUDA-resident graph segfaulted.
- Installing torch pulled numpy 2.4.6, which was pinned back to 2.2.5.

Implementation agents for this step ran on Haiku 5.5 (high effort) at the user's request.

### 28.4 Freeze and as-of graph builder (2026-10-18)

**Freeze.** On the user's instruction, the preregistration was frozen with all orchestrator resolutions accepted:

- Freeze appendix at 15:34 UTC. Pre-appendix SHA-256 `50ae42b4…`.
- Config `configs/exp015_rdl_pilot.json`, SHA-256 `89f2b3c0…`.
- Late-known fields are not read at all.
- The exact 8-type edge list is fixed.
- No fit may start until the graph tests pass.

**Builder.** `src/rdl_graph.py` and `tests/test_exp015_rdl_graph.py` were implemented by a Haiku 5.5 (high effort)
agent. Amendment A1 in the preregistration records the fill-ins it needed:

- A null PR/issue `created_at` keeps the node visible, with age 0 and `created_at_known = 0`.
- A known `created_at` ≥ C hides the node.

**Verification.**

- Tests in `asg2-rdl`: 36 EXP-015 tests pass, and with the shared relational/decision tests 76 pass. This needs
  `--basetemp` in the scratchpad, because pytest's default temp dir fails with WinError 5 on this machine. That
  failure predates this work.
- Real-data graph at C = 2026-09-09 12:00 UTC (built in 0.3 s; no labels read):
  - users 8,556 (6,382 labelled = the EXP-014a population); repos 33,208; PRs 14,914; issues 13,627.
  - forward edges: pushes 31,998; opens_pr 9,882; reviews 4,395; review_comments 2,372; pr_in_repo 9,882;
    opens_issue 9,226; comments 6,804; issue_in_repo 12,795.
- F0 parity with the EXP-014a cache is exact. The 7-day push users match the official pipeline exactly (1,165).
- The orchestrator independently checked truncation invariance at 09-09 and 09-30: graphs from the full tables equal
  graphs from tables with every row timed ≥ C deleted.

**Known limitation.** 5,032 of 14,914 PR nodes (34%) have a null `created_at`. Their `pr_in_repo` edge is therefore
dropped, and they reach a repository only through users.

**Status:** EXP-015 Ongoing. There are no fits and no scores. The runner `experiments/exp015_rdl_pilot/run.py` is not
written yet.

### 28.5 Amendment A2, runner and no-fit preflight (2026-10-19 Asia/Taipei)

**Amendment A2.** Recorded in the preregistration on the user's instruction, before any fit.

- **A2.1:** `pr_in_repo` / `issue_in_repo` are timed by `created_at`, or by `first_seen_at` when `created_at` is null.
- **A2.2:** runner fill-ins, recorded before the runner code existed:
  - early-stopping split: holdout = 7 latest training origins; fit = s + 7 d ≤ first holdout origin;
  - epoch and loss definition;
  - fit-row-only F0 standardization;
  - exact architectures, including the edge-aware SAGE layer for GNN1T;
  - seeds and AP;
  - preflight-default modes;
  - OOM fallback.

**Builder after A2.1** (Haiku 5.5 high agent):

- At C = 09-09 12:00 UTC, 22,743 of 22,810 PR nodes now have a repository edge. Before A2 it was 9,882 of 14,914.
- The PR node count rises because repository-linked PRs without in-window user edges are now included.
- 67 PR nodes still have no repo edge; their creation or first-seen time is outside the 90-day lookback.
- The orchestrator reran the truncation-invariance check at 09-09 and 09-30: still identical.

**Runner** `experiments/exp015_rdl_pilot/run.py` and `models.py` (Haiku 5.5 high agent):

- It reuses the EXP-022 / EXP-014a workspace, decision-layer, gate, AP and bootstrap code by import.
- The default mode is a no-fit preflight. Fit mode refuses without `--user-approved-fit`, refuses an existing run
  directory, and runs `tests/test_exp015_rdl_graph.py` as a precondition.
- Spec gaps the agent filled:
  - BLEND is computed only if a GNN arm is selected.
  - Holdout AP is pooled over the 7 holdout origins per target.
  - The NeighborLoader fallback uses batch size 1024.
  - The 45-minute guard covers all arms and seeds of a report origin.
  - Permutation and size nulls are not computed (not in the registered outputs).
- The orchestrator reviewed the leakage-critical code:
  - training labels use `_stack(..., A)`;
  - report labels are attached for scoring only;
  - the models read node features, never labels;
  - standardization uses fit-origin labelled rows only.
- The agent flagged that the config's pre-appendix hash did not reproduce. That was a cut-point artifact: it
  reproduces (`50ae42b4…`) when the file is cut before the blank line preceding the appendix.

**Tests** (`asg2-rdl`, FutureWarnings as errors): 153 passed. This covers EXP-015 graph and runner, EXP-022, the shared
relational, decision, Phase A and Phase B tests.

Code hashes at this entry (re-hashed into the manifest at the first fit):

- `src/rdl_graph.py` `9597870c…`
- `tests/test_exp015_rdl_graph.py` `ead0527a…`
- `run.py` `9d985932…`
- `models.py` `f122f940…`
- `tests/test_exp015_runner.py` `4a536e53…`

**Preflight, real data, no model built** (run by the orchestrator):
`experiments/exp015_rdl_pilot/preflight/preflight-20261018T161914Z/preflight.json` (`4702350a…`).

- All 322 checks PASS:
  - 19 input-hash and environment checks;
  - 5 report-origin alignments with EXP-014a: population, (UserID, target, y) and P_persist7 identical;
  - 285 no-future-edge graph audits;
  - 5 early-stopping splits (fit 43 / holdout 7 / gap 6 per report origin);
  - F0 column and training-table evidence checks.
- Report-origin graphs: 160k–267k edges, built in 0.3–0.4 s each. The largest training-origin graph (07-01) has
  about 216k nodes and 629k edges.
- Labels were built for the parity checks only. They are the same September development labels inspected in
  EXP-014a/022.

**Status:** EXP-015 Ongoing, ready to fit. **No fit has run.** The fit waits for the user's explicit approval
(`--mode fit --user-approved-fit`).

### 28.6 EXP-015 fit and registered selection — 2026-10-19

**Authorization.** The user approved exactly one fit on 2026-10-19: "I approve the EXP-015 fit: run --mode fit
--user-approved-fit in the background, then report the registered selection result and log it." It covers only run
`exp015-rdl-pilot-v1` under the frozen protocol (freeze appendix, A1, A2). No code, config or protocol changed, and
there was no rerun.

**Pre-fit checks** (orchestrator, before launch):

- Config `configs/exp015_rdl_pilot.json` SHA-256 `89f2b3c07ad56da13c0b2d68a394610f569f1e355ab76300d31e52c756f82661`.
- Code SHA-256 prefixes match log §28.5:
  - `src/rdl_graph.py` `9597870c`
  - `tests/test_exp015_rdl_graph.py` `ead0527a`
  - `run.py` `9d985932`
  - `models.py` `f122f940`
  - `tests/test_exp015_runner.py` `4a536e53`
- The run directory did not exist.

**Execution.**

- Command (repository root, `asg2-rdl`):
  `python -s -m experiments.exp015_rdl_pilot.run --config configs/exp015_rdl_pilot.json --mode fit --user-approved-fit`.
- Started 2026-10-19 04:09:42 UTC and ended 05:24:56 UTC; exit code 0.
- The runner's precondition (`tests/test_exp015_rdl_graph.py`): 51 passed.
- Run directory: `experiments/exp015_rdl_pilot/runs/exp015-rdl-pilot-v1/`.

**Manifest.**

- Code hashes at start equal the pre-fit hashes above. Config `89f2b3c0…`. Data release
  `506d860693df60a698b68aa63a8edb4a984c8e1b`, latest row 2026-10-13 12:59:29 UTC.
- Repository commit `7cff786911570598007681901b4bd4c9e5272d70`, with 87 uncommitted entries at run time.
- Preregistration SHA-256 at run end: `09d1f1f6…`.
- Environment: python 3.11.17, torch 2.5.1+cu124, torch_geometric 2.6.1, numpy 2.2.5, pandas 2.3.3, scikit-learn 1.8.0,
  RTX 3070 Ti Laptop GPU; `-s` flag confirmed.
- Artifact hashes (first 12 hex): `predictions.parquet` `1579e9aefc3d`, `ap_by_origin_target.csv` `e411de348dc3`,
  `selection.json` `a38f9d8d6b5c`, `manifest.json` `14e26633d5cf`.

**Measured: mean per-target AP** (six targets; all4 = 09-09/16/23/30; 09-30 provisional; 09-02 continuity only).

| arm | 09-02 | 09-09 | 09-16 | 09-23 | 09-30 | all4 mean | Δ vs REF | wins vs REF | Δ vs MLP0 | wins vs MLP0 | Δ vs H2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| REF (HGB F0) | 0.4684 | 0.5674 | 0.5280 | 0.5107 | 0.5186 | 0.5312 | — | — | +0.0646 | 4/4 | −0.0063 |
| H2 (EXP-022) | 0.4619 | 0.5710 | 0.5375 | 0.5131 | 0.5286 | 0.5375 | +0.0063 | 4/4 | +0.0709 | 4/4 | — |
| MLP0 | 0.4520 | 0.5429 | 0.4529 | 0.3652 | 0.5054 | 0.4666 | −0.0646 | 0/4 | — | — | −0.0709 |
| GNN1 | 0.4417 | 0.5647 | 0.4946 | 0.4490 | 0.5168 | 0.5063 | −0.0249 | 0/4 | +0.0397 | 4/4 | −0.0312 |
| GNN1T | 0.4459 | 0.5712 | 0.5009 | 0.4555 | 0.5230 | 0.5126 | −0.0186 | 2/4 | +0.0460 | 4/4 | −0.0249 |

Per-origin Δ vs REF, in all4 order (09-09/16/23/30):

- GNN1: −0.0027, −0.0334, −0.0617, −0.0018.
- GNN1T: +0.0038, −0.0272, −0.0552, +0.0044.

Δ vs MLP0, same order:

- GNN1: +0.0218, +0.0417, +0.0838, +0.0113.
- GNN1T: +0.0283, +0.0480, +0.0903, +0.0175.

On 09-02 every neural arm is below REF (−0.016 to −0.027), and both GNNs are also below MLP0.

all4 per-target AP (open_pr / open_issue / issue_comment / pr_review / pr_review_comment / push):

- REF: 0.488 / 0.470 / 0.490 / 0.509 / 0.468 / 0.764.
- GNN1: 0.455 / 0.418 / 0.463 / 0.501 / 0.460 / 0.742.
- GNN1T: 0.450 / 0.434 / 0.471 / 0.506 / 0.473 / 0.742.
- MLP0: 0.425 / 0.365 / 0.411 / 0.452 / 0.413 / 0.734.

GNN1T is above REF only on pr_review_comment (+0.005). The largest gap is on open_issue (−0.036).

**Registered selection: none eligible — recorded negative.**

- GNN1 fails the REF condition: 0/4 wins and −0.0249 on average.
- GNN1T fails it too: 2/4 wins (09-09 and the provisional 09-30) and −0.0186 on average.
- Both GNNs pass the MLP0 condition: 4/4 wins and +0.040 / +0.046 on average.
- BLEND was not computed, because it is defined only for a selected arm.

**Reported only (not used for selection).** Official score, pooled over all4. The §29.2 gate is against P_persist7
(pooled 0.5166). "Δ vs REF layer" compares with REF under the same layer, and "wins" counts origins.

| arm × layer | score | Δ persist7 | wins | LOO min | gate | Δ vs REF layer | wins vs REF layer |
|---|---|---|---|---|---|---|---|
| REF × P021_blend50 | 0.5454 | +0.0288 | 4/4 | +0.0263 | screen_pass | — | — |
| H2 × P021_blend50 | 0.5495 | +0.0329 | 4/4 | +0.0298 | screen_pass | +0.0041 | 4/4 |
| GNN1T × P021_blend50 | 0.5433 | +0.0266 | 4/4 | +0.0224 | screen_pass | −0.0022 | 2/4 |
| GNN1 × P021_blend50 | 0.5420 | +0.0253 | 4/4 | +0.0211 | screen_pass | −0.0035 | 2/4 |
| MLP0 × P021_blend50 | 0.5302 | +0.0135 | 4/4 | +0.0075 | screen_pass | −0.0153 | 1/4 |
| REF × P016_a100 | 0.5371 | +0.0205 | 4/4 | +0.0180 | screen_pass | — | — |
| H2 × P016_a100 | 0.5390 | +0.0224 | 4/4 | +0.0197 | screen_pass | +0.0019 | 3/4 |
| GNN1T × P016_a100 | 0.5293 | +0.0127 | 4/4 | +0.0077 | screen_pass | −0.0078 | 2/4 |
| GNN1 × P016_a100 | 0.5258 | +0.0092 | 3/4 | +0.0043 | screen_pass | −0.0113 | 2/4 |
| MLP0 × P016_a100 | 0.5075 | −0.0091 | 1/4 | −0.0180 | rejected | −0.0295 | 0/4 |

Every GNN row passes the §29.2 gate vs persistence, as REF and H2 do. Under both layers each GNN row is below REF
pooled, with 2/4 origin wins vs REF.

Paired UserID-cluster bootstrap of the all4 mean-AP Δ vs REF (500 reps, seed 20261022), with 95% intervals:

- H2: +0.0064 [+0.0033, +0.0097].
- GNN1T: −0.0186 [−0.0266, −0.0104].
- GNN1: −0.0247 [−0.0328, −0.0164].
- MLP0: −0.0642 [−0.0770, −0.0534].

P(Δ > 0) = 0.000 for every neural arm.

**Compute.**

- Total wall time 4,500 s (75 min).
- Per report origin (all arms × 2 seeds): 09-02 1,521 s, 09-09 918 s, 09-16 962 s, 09-23 426 s, 09-30 603 s. The
  maximum is 25.4 min, under the 45-min guard.
- Peak GPU memory 1,162.5 MiB (GNN1 at 09-02). MLP0 peaked at about 120 MiB.
- Every arm ran full-batch: no OOM fallback, no NeighborLoader, and no blocked origin.
- Early-stopping split: fit 43 / holdout 7 / gap 6 at every origin.
- Epochs run per seed (best epoch = epochs − 3), at 09-02/09/16/23/30:
  - MLP0: 19/17, 29/29, 24/27, 8/17, 12/13.
  - GNN1: 18/16, 18/9, 19/16, 7/7, 9/13.
  - GNN1T: 18/24, 18/9, 15/11, 7/9, 12/13.
- No arm hit the 30-epoch cap.
- Report-origin graphs had 160k–267k edges.

**Verification (orchestrator, own scripts; nothing imported from `src/` or `experiments/`):**

- **Manifest:** the five code hashes, config hash, data release and commit are as above. The code files still hash the
  same after the run.
- **Comparators:** `p_REF` and `p_H2` in the run equal `p_raw_F0` (EXP-014a `3ff17746…`) and `p_H2` (EXP-022
  `c4ad7120…`) on all 182,016 (origin, UserID, target) rows, with max |Δ| 0.0. Labels `y` are identical.
  Populations: 7,417 / 6,382 / 5,862 / 5,497 / 5,178.
- **AP:** per-target AP was recomputed with `sklearn.metrics.average_precision_score` for MLP0, GNN1 and GNN1T from the
  run, and for REF and H2 from their source parquet files.
  - It equals `ap_by_origin_target.csv` on 150/150 rows (max |Δ| 5.6e-17).
  - REF and H2 equal the EXP-022 `ap_by_origin_target.csv` on 60/60 rows. all4 mean AP is REF 0.5312 and H2 0.5375,
    as in log §27.3.
- **Selection:** re-derived from my AP numbers with the registered rule (≥ 3/4 wins and the all4 average, against both
  REF and MLP0; highest eligible; tie → GNN1). The result is "none", equal to `selection.json`.
- **Decision layers:**
  - P016_a100 and P021_blend50 were re-implemented: top-k at the P_persist7 count, ties by UserID ascending, and
    scipy average ranks for the blend. They reproduce every saved decision cell for all arms and origins (0
    mismatches).
  - Scores from the official `assignment2/evaluation/evaluate.py` equal my own F1 code (max gap 0.0).
  - The pooled gate, LOO and Δ-vs-REF-layer numbers equal `screen_gate.csv` (max |Δ| 0.0), and the gate statuses
    match.
- **Parity:** `parity.csv` has 55 `official:` rows, which is every scored decision (5 origins × (P_persist7 + 5 arms ×
  2 layers)). The max |Δ| is 0.0. Its 20 key-parity rows are written as 0.0 after the runner's alignment assertions
  pass; they are not measured differences. The comparator check above measures them independently.
- **Reproducibility:** the MLP0 same-seed rerun (seed 42, 09-09, full-batch) gave mean AP 0.530665 both times, so
  |Δ| = 0.0 ≤ 1e-4.
- **Bootstrap:** re-implemented with sklearn AP and `sample_weight`, using the same RNG draws. It reproduces
  `ap_bootstrap.csv` to max |Δ| 1.6e-16.
- **Leakage:** all 255 graphs recorded in `training_info.json` (fit, holdout and report, across 5 origins) have a
  maximum edge time before their own cutoff.

**Decision** (preregistration "Selection and gates" and resolution 2):

- No GNN arm is eligible, so EXP-015 is a **recorded negative**.
- HGB stays. The REF-vs-H2 choice stays with freeze #1 under the EXP-022 result (§29.6).
- No arm enters freeze #1 from this pilot, and P_persist7 stays the default.
- The deferred relation-family, drop-neighbor-relation and randomized-neighbor ablations are not run, because their
  condition (a selected arm) is not met.

**Interpretation and limits.**

- Relational message passing adds clearly over a node-only MLP trained the same way (+0.040 to +0.046 AP, 4/4).
- Neither GNN matches gradient-boosted trees on the same 59 F0 features.
- MLP0 is a weak neural baseline: −0.065 vs REF, and 0.365 at 09-23, where seed 42 stopped at epoch 8. The GNN-vs-MLP0
  gain therefore shows that the graph helps this neural recipe. It does not show that relations add information
  beyond HGB on F0.
- This is one bounded run with small fixed models (2 layers, hidden 64) and no tuning, by design. It is not evidence
  that larger RDL models (out of scope, §29.7) would fail.
- It is development evidence only, on September weeks inspected many times; 09-30 is provisional.
- EXP-013 has no result yet, so the handcrafted-relational comparison for Q2 is still missing.
- The run had 87 uncommitted repository entries.

## 29. Overnight feature campaign session — October 21, 2026 Asia/Taipei

### 29.1 Preparation and authenticated baseline reproduction (score-free for new features)

Session `reports/feature_campaign_20261020/sessions/session-20261020T165311Z/` began 2026-10-20 16:53:11 UTC / October 21 00:53:11 Taipei. Hard deadline: October 21 00:53:11 UTC / 08:53:11 Taipei; last 20 minutes reserved for reporting. §30 controls the new campaign; the pasted overnight prompt limits the executable queue to S/M/SM, original EXP-013, A/conditional B/AB, R/W/RW. No X/TR/TX/capacity/bagging or completed-study reruns are authorized in this session.

- `startup_audit.json`: pinned instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`; eight Parquets, schema, evaluator and README copied to a separate snapshot and hashes verified. Existing user changes are preserved. All 19 confirm-plan v0 source/config locks pass; its original manifest is copied byte-identically.
- Runtime used: `C:/Users/Davon/miniconda3/envs/asg2-rdl/python.exe -s`, Python 3.11.17, NumPy 2.2.5, pandas 2.3.3, scikit-learn 1.8.0. One fit process, four threads, six-GiB process-RSS guard. Sole delegated implementation agent: supported `gpt-6-luna`, reasoning `max`; no nested delegation. The main model setting cannot be changed or authenticated by the exposed tools, so Sol High is not claimed as verified.
- `tests-readiness-1.log`: **112 passed in 15.36s** across original EXP-013 auditor/runner/feature tests and preserved ranking/decision tests. These were rerun by the orchestrator.
- `controls/result.json`, `controls/parity.csv`: fresh REF/H2 reproduction at all five registered origins; **450 checks, maximum absolute difference 0.0**. Keys, labels and decisions exact; probabilities, AP and instructor scores exact. All **86 freshly built F0 panels** match accepted EXP-022 panels exactly, including dtypes/order. `folds.csv`, `training_membership.parquet`, model bundles, predictions, hashes and costs preserved. These are mandatory control reproductions, not a new research rerun of EXP-022.
- Original EXP-013 preservation lock initially failed because two execution-plan documents had moved into `docs/exp/`. Added byte-identical copies at the former paths; moved files and frozen packages untouched. Added the analogous EXP-013 plan compatibility copy for the new preflight. `exp013_v1_integrity.json` then passes: 22 frozen artifacts, 311 external input hashes and all 288 protected files. A fresh score-free preflight is underway; no original EXP-013 candidate fit has started.
- New instructor SHA `ebbef8c7c94d838d19e8371818f1558700adac95` downloaded only from the instructor remote into a separate snapshot. `new_source_coverage.json`: pipeline latest observed time **2026-10-20 11:59:25 UTC**, **3,635 seconds short** of the first origin's 13:00 UTC threshold. Raw table maxima are also recorded. October confirm labels are not built. D3 holdout overlap is acknowledged as already authorized; immature source blocks confirmation, independent of date.
- Frozen v0 also has a harness schema incompatibility: original manifest has `primary`/`secondary` fields but no `candidates` field required by the existing EXP-027 entry point. No manifest bytes are changed to bypass it. Record compatibility work separately if maturity later passes.
- `source_pk_coverage.csv`: unique non-null internal PKs checked separately within each of all eight source tables in both releases. `exp028_september_label_drift.csv` contains the two registered fixed_asof/release_latest panels, stable-user matches and explicit as-of clocks. `exp028_september_f0_drift.csv`, source coverage/delay summaries, and fixed baseline error strata are produced without October outcome inspection. This is partial EXP-028 preparation, not final shortlisted-feature drift acceptance.

New feature arms remain unexecuted pending implementation review, meaningful correctness checks and machine-readable freeze. A code review found `active_bins_3d` initially stored as int8; it is a count and is being corrected to registered int32 before any freeze or score. No feature formula or scientific gate changes.

### 29.2 Fresh original EXP-013 preflight and preserved auditor failure

Fresh score-free package: `experiments/exp013_relational_features/runs/exp013-protocol-freeze-session-20261020T165311Z/`. It completed in **462.985 seconds**, measured process RSS at freeze **3,274,653,696 bytes**, below its registered five-GiB bound. No candidate scores or holdout labels were built.

The orchestrator then ran the actual independent preflight CLI. It exited with an **AssertionError: Frozen config digest differs from its saved bytes**; failure is retained in session `exp013_fresh_preflight_audit.json` and `.log`. Inspection identifies an audit-only serialization mismatch: source config SHA256 `111ef40a...`; canonical saved `run_config.json` SHA256 `6b7b70fc...`; decoded JSON objects compare equal. Both are independently bound in the package (source config lock and canonical-copy artifact lock). The auditor currently compares the canonical copy against the source-file digest. No original candidate fit is authorized through this failure. The sole implementer was assigned a scoped authentication correction and adversarial tests: verify each file against its own recorded hash and require decoded equality. Frozen scientific files and rules remain unchanged; any accepted audit revision will be recorded before execution.

Main-independent campaign auditor selfcheck: **40 official score checks passed**, exact H2 self-comparison AP/score deltas zero, whole-origin omission detected. Independent cluster bootstrap selfcheck used all **1,000 replicates**, all valid, **144 weighted-AP checks against sklearn**, zero control-vs-itself intervals. Instructor all-negative convention was separately checked as score/macro/micro = **1.0**. The initial model-bundle parameter inspection used an incorrect attribute (`models` rather than `classifiers`) and failed without changing a model; the corrected inspection confirms actual max_iter80 / learning_rate0.07, and actual estimator parameter arrays are recorded separately from library defaults.

### 29.3 Independent raw-feature QA before campaign fitting

`campaign_source_oracle.py` independently maps raw source actor PKs to stable cohort IDs and computes all S18/M12 formulas without importing their formula helpers. **150 origin/column checks pass exactly** across all five report populations, including final stored dtypes (S18 active-bin counts int32). All R24/W24 columns exactly match the accepted builders, and A9/B7 exactly match the independent raw-source reconstruction from `reports/verify_exp013.py`, including storage casts; even pre-cast maximum discrepancy is **0.0**. Removing all future-visible event rows leaves S18/M12/R24/W24 features unchanged at each report origin. No labels were built by this check. Runtime **104.531 seconds**. Evidence: session `campaign_source_oracle_result.json`, `campaign_raw_formula_checks.csv`, and `.log`; authenticated builder hash `9d69c2d3657cc4aea0580517ec974a2cade2ba01d254ee3174087a20e04326b2`.

### 29.4 Auditor clock contradiction and integrated implementation checks

The config-copy repair allowed independent preflight audit v2 to reach the next check, where it failed: **Feature or decision timestamp is not before its target origin**. The failure is preserved, along with the exact auditor hash. The global check rejected equality for every track, contradicting the original secondary track's explicit required legacy cutoff/decision == target origin. An audit-only helper now rejects all future timestamps, rejects primary equality, and preserves archival equality; the exact primary S-4h/S-3h and archival equality checks remain. Three boundary tests cover those cases. No registered scientific clock or frozen panel was changed. Full independent audit v3 is running; acceptance remains pending its result.

Sole Luna implementation handoff reported **93 tests passed in 11.24s**, no real preparation or model fits. The orchestrator then independently ran the combined new/preexisting suite: **171 passed in 22.84s**, session `tests-integration-1.log` and a new preserved workspace basetemp. It covers feature math, keys/dtypes, full-output schedules, fixed decisions, conditional gates, parent hashes, bootstrap oracle, exact config-copy authentication, primary/archive clocks and preserved temporal/relational/ranking rules.

Read-only review identified and repaired several implementation errors before any new candidate score: late conditional checks, dtype-sensitive equality on identical integer values, premature target-axis collapse in bootstrap scoring, support-trace sentinel handling, and explicit resource-feasibility accounting. A final preparation-path defect was found after the passing suite: feature-panel metadata pointed into a temporary directory later renamed. The sole implementer is assigned a narrow final-path repair and a score-free preparation/rename integration test. No actual campaign prepare/freeze or fit has started.

### 29.5 Label-free campaign preparation begins; original auditor remains separate

After the staging-path repair, the orchestrator reran campaign and audit-boundary tests: **33 passed in 6.40s**, session `tests-stagingfix-review-1.log`. Both actual `prepare --help` and `fit --help` were checked. The real campaign preparation command now runs with four thread environment variables and the fixed nested directory `reports/feature_campaign_20261020/sessions/session-20261020T165311Z/campaign`; it builds registered additions and a machine freeze without labels. No candidate fit has started.

Original independent preflight audit v3 progressed through panel checks but failed on source-audit cutoff keys. The saved 56 keys use timestamp strings with a space while the auditor required ISO strings with T; parsed UTC sets are exactly equal. Inspection also finds a field-definition mismatch in the independent count helper: saved `source_event_rows` means all raw direct-stream rows; the helper currently returns only pre-cutoff rows, which already have their separate field. A scoped audit-only correction and synthetic contract tests were assigned. Failures and all code-version hashes remain preserved. No source, feature, frozen artifact, scientific clock, or gate is being changed. Original EXP-013 remains unaccepted until a complete audit passes; the separately validated temporal campaign may proceed under §30's explicit compatibility-block exception.

### 29.6 Source-audit contract tests and process accounting

The orchestrator reran the latest original auditor compatibility suite: **41 passed in14.08s** (session tests-source-audit-review-2.log). UTC-normalized cutoff keys, all-raw source row counts, production concentration records wrapper and sparse zero-count stream maps are authenticated without changing scientific inputs. Full independent preflight audit v4 remains pending. Sandbox process inventories cannot see the prepare process; elevated native host inspection verified the exact command/PID29840, and a start-time-checked memory monitor records lifetime peak2,192,973,824 bytes so far. No candidate fit has started. A complete-history instructor Git bundle has also been created and verified for pinned SHA506d860...; development source remains unchanged.


### 29.7 Machine-readable score-free freeze authenticated

Freeze logical SHA256 **36001ba0aefdc451da49bf10648fd7f41be535f6fcada27c9aa91e9afbc2dc0a**; file SHA256 **c30d81b892e2fc178860fc348b97e06d97e8fc185cc223f399e3ce3b42eeb231**. Root campaign_freeze_audit.json passes:18code files,8source files,86feature origins,94additions,1,638,144expected keys across9possible arms,2,287,272exact training membership rows,280fold clocks/weights and2,580trace examples. All actual column counts match the registered arms. No labels were built by preparation.

The final staging-directory rename failed with WinError5 after every artifact and freeze was written. The failed staging package and log are preserved. A byte copy to the intended new final directory passed full independent hash verification; no feature recomputation or source/scientific change. A separate retained directory-rename probe passes. The initial root freeze audit failed because Windows default decoding corrupted Unicode when reconstructing canonical JSON; explicit UTF-8 decoding corrected only independent auditors and the first failure log remains. Host-measured preparation lifetime peak2,192,973,824bytes is below6GiB.

One additional H2/F0 six-target control bundle at09-09 is now charged to runtime/fit accounting to validate the new runner fitting boundary against the already reproduced probabilities and exact estimator parameters. This adds no candidate or scientific hypothesis. Candidate S remains unexecuted until this parity check passes. Instructor remote recheck still returns ebbef8c...; maturity is unchanged.


### 29.8 Shared fitting-boundary control parity passes; EXP-024 starts

Session shared_runner_control_0909/result.json:38,292raw probabilities match H2 exactly (maximum difference0.0); every actual estimator parameter equals its authenticated H2 counterpart. One charged six-estimator bundle,29.266seconds, peak1,158,889,472bytes. The updated independent metric auditor selfcheck passes40official-score comparisons with zero H2 self AP delta. EXP-024 S begins under authenticated freeze36001ba0... with77columns, all five report origins in fixed order. This entry records execution start, not a result.


### EXP-024 S: independently accepted completed result

Columns 77; all5 origins/182,016 rows/30 estimators. AP_all4 0.537942819459; matched H2 delta +0.000408081049; mature3 AP delta +0.001448737122. P021 pooled all4 delta vs H2 -0.000639610370; vs persistence +0.032273479881; mature3 vs H2 -0.000569075720. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_S.json and audit_S_*.json. Wall 200.224s; max origin measured RSS 1576845312bytes. September development only; no promotion or October outcome exposure.

### 29.9 Session closure and limitations

2026-10-21T02:38:51.519833+00:00: bounded session closed. OnlyS is complete/rejected; original013 blocked preflight; all remaining new arms not_run. See session SESSION_REPORT.md, trial_ledger_final.csv, parent_selection.json and feature-v1 manifest. No feature nominee, no October outcomes, no promotion. All19v0pins/manifest bytes and18campaign code locks pass. No live fit/audit process or active delegate. The8-hour limit/reporting reserve were missed: elapsed9.761h; late S audits and auditor-only repair/tests after deadline disclosed. No fit started after S17:55:45Z (finished17:59:05Z). Latest full original audit v4 failed rolling-window predicate;44tests for new audit-only repair pass but full rerun missing. Luna final drift task hit usage limit, no files. Root model/metric/bootstrap acceptance for S is complete. Governing plan bytes retained for unfinished-arm resume; dated status addendum written separately. No public source commit.

### 29.10 Authorized two-hour continuation — October21 10:44 Taipei

2026-10-21T02:45:25.6550362+00:00 — User authorized a new two-hour session. Start02:44:19Z, fit-stop04:24:19Z, hard report deadline04:44:19Z; frozen campaign cap04:57:17Z unchanged. Native host query finds no active experiment. Prior S remains completed/rejected; no refit. Original ledger closure and feature-v1 manifest are historical snapshots retained; resumed results will receive separate final ledger/manifest revision.


Resume validation passes same campaign freeze36001ba0..., unchanged S,19v0pins and original12fold/210training-week/2448000membership/62role-key audit. v5 independent auditor revision recorded before any original fit;44prior tests pass. M71column execution and read-only full original preflight audit begin under external UTC guard04:24:19Z; there is exactly one fitting process.


### EXP-031 M: independently accepted completed result

Columns 71; all5 origins/182,016 rows/30 estimators. AP_all4 0.533385822262; matched H2 delta -0.004148916148; mature3 AP delta -0.003230104481. P021 pooled all4 delta vs H2 -0.002622482024; vs persistence +0.030290608227; mature3 vs H2 -0.001789663420. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_M.json and audit_M_*.json. Wall 206.911s; max origin measured RSS 1430646784bytes. September development only; no promotion or October outcome exposure.

### EXP-035 SM: independently accepted completed result

Columns 89; all5 origins/182,016 rows/30 estimators. AP_all4 0.536615426669; matched H2 delta -0.000919311741; mature3 AP delta +0.000629247322. P021 pooled all4 delta vs H2 -0.002899047358; vs persistence +0.030014042893; mature3 vs H2 -0.001997910442. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment, interaction:screen_eligible, interaction:P021_all4_exceeds_both_by_0_002, interaction:AP_not_below_either_parent, interaction:mature3_not_below_stronger_parent. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_SM.json and audit_SM_*.json. Wall 235.909s; max origin measured RSS 1625473024bytes. September development only; no promotion or October outcome exposure.

### 29.11 Original-only compatibility blocker and permitted separate A follow-up

2026-10-21T03:11:53.4193122+00:00: Original preflight remains blocked by archival operational-threshold auditor representation. A direct last-stage check provesv5 cannotpass: archivalHGB fixed first-report thresholds versus requiredperreport records; archival persistence is fixed0.5 and has no fitted threshold rows. Read-only v5 stopped withidentity-checked PID, failure andguard preserved; nofitinterrupted. Audit-only v6 accepts exactlythe registered static HGB vector with exact calibrationprovenance and fixedunfitted persistence, keepsprimary/hybrid perreport strict.51targetedroot tests pass13.40s; direct operational andfullpreflight still pending. SM independently acceptedasrejected, exactS/Mparentcomparisons fail complexityincrement. §30 allows separate rawvalidated work whileoriginal is blockedsolelybyhistoricalauditorcompatibility; A68 isrunning withpassingseparaterawQA. Original study is not Completed and itsprotocol/gates are unchanged. Logcommand parserfailure delayed thisentry, notactual independentSMchecks; recorded in decisionlog.


Original auditor v7 direct operational-reference reconstruction passes288000probabilitykeys,63populationvectors/officialscores and4pinnedpooledscores.52targetedroot tests pass15.85s. A second direct failure identifiedprimarypersistence fixed_rule metadata, nowstrictly authenticatedas0.5unfittedrule whilelearnedvectorsrequirelegacyproxycalibration. No scientific/frozen execution change. Versioned fullpreflightv7 starts; originalacceptance remains pending.


### EXP-033 A: independently accepted completed result

Columns 68; all5 origins/182,016 rows/30 estimators. AP_all4 0.538654243054; matched H2 delta +0.001119504644; mature3 AP delta +0.001951334562. P021 pooled all4 delta vs H2 -0.001326222342; vs persistence +0.031586867909; mature3 vs H2 -0.001531030724. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_A.json and audit_A_*.json. Wall 187.973s; max origin measured RSS 1423814656bytes. September development only; no promotion or October outcome exposure.

### EXP-032 R: independently accepted completed result

Columns 83; all5 origins/182,016 rows/30 estimators. AP_all4 0.537672524061; matched H2 delta +0.000137785651; mature3 AP delta -0.000031682217. P021 pooled all4 delta vs H2 -0.001086180010; vs persistence +0.031826910241; mature3 vs H2 -0.001369265336. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_R.json and audit_R_*.json. Wall 230.693s; max origin measured RSS 1590169600bytes. September development only; no promotion or October outcome exposure.

### EXP-032 W: independently accepted completed result

Columns 83; all5 origins/182,016 rows/30 estimators. AP_all4 0.537151825010; matched H2 delta -0.000382913400; mature3 AP delta +0.000629397645. P021 pooled all4 delta vs H2 -0.000549379191; vs persistence +0.032363711060; mature3 vs H2 -0.000941800290. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_W.json and audit_W_*.json. Wall 225.218s; max origin measured RSS 1549533184bytes. September development only; no promotion or October outcome exposure.

### EXP-035 RW: independently accepted completed result

Columns 107; all5 origins/182,016 rows/30 estimators. AP_all4 0.538281989784; matched H2 delta +0.000747251374; mature3 AP delta +0.001487092138. P021 pooled all4 delta vs H2 -0.001104568198; vs persistence +0.031808522053; mature3 vs H2 -0.000929380402. Eligible=False; failed clauses: ranking_magnitude, ranking_stability, operating_policy_increment, interaction:screen_eligible, interaction:P021_all4_exceeds_both_by_0_002. Independent audits replay all30 estimators, reconstruct280 training fold digests/weights, verify exact decisions/metrics/official scorer and independently reproduce1,000-user-cluster intervals. Evidence: session acceptance_RW.json and audit_RW_*.json. Wall 270.105s; max origin measured RSS 1949585408bytes. September development only; no promotion or October outcome exposure.

Feature queue locked 2026-10-21T03:40:54.649198+00:00: seven independently verified feature arms; B/AB not_run after A fails. T*=None, R*=None, nominee=None. Max-relative preregistered ties and both union complexity clauses independently reconstructed. Parent/nominee values and audit hashes in continuation parent_selection.json, union_clause_audit.json and separate CANDIDATE_MANIFEST_feature_v2.json. v1 and v0 bytes preserved; no October outcomes, no promotion. No final-nominee null diagnostic because nominee is none.


### Continuation reporting and audit acceptance

2026-10-21T03:50:24.365240+00:00: Root independently reviewed the authenticated-preflight reuse wrapper and reran 68 auditor tests (18.07s), all pass; these overlap previous counts. Full original source audit v7 still pending, no original fits yet. Matched reporting tables now cover840target/origin/reference/layer rows and168all-four-origin target summaries, including open_issue and worst-week F1; 28combined comparisons independently reproduce all7H2/P021 gate deltas within1e-12. Files in session-20261021T024419Z: matched_target_deltas.csv, matched_target_summary.csv, matched_combined_scores.csv. Final feature exposure ledger and manifest binding retain v0/v1/v2 bytes and record all7rejections/BABconditional skips, no October outcomes. EXP028 shortlisted-group audit actually completed score-free with an empty group set because no feature nominee exists; no rejected group rebuilt, existing F0 drift evidence retained. No predictive promotion.


Original EXP013 independent source preflight v7 passed2026-10-21T04:02:27Z,48m35.5s,peak2897969152bytes. Actual full-source counts above retained in session exp013_preflight_audit_v7.json; root receipt sealed04:03:13Z after reauthentication. Historical frozen execution begins under stricter04:10Z fit stop; no fitted result accepted yet.


Original execution operational failure: supervisor handoff at04:06:48Z terminated Windows child,11F0bundles complete/1F0interrupted; no A/B/union/shuffle candidate fit started. This was root orchestration error; child-preserved/null-exit claims in experimental handoff record were invalidated by native inventory. All prior output remains. Before any candidate fit, pre-candidate retry amendment records fresh retry1 output, aggregate74attempt cap (12old+at most62new), unchanged original science, fixed04:17UTC/5GiB guardian. Aggregate-cap adversarial test passes on own sleeping child; no further supervisor handoff. New scientific fit acceptance pending.


### Original EXP013 final continuation status — incomplete, not a negative candidate result

2026-10-21T04:20:16.043913+00:00: 1.8.0/numpy2.2.5 runtime completed12F0bundles then strict parity failed primaryJun7 maxabs0.08488271343558931, before any candidate. Every12fold F0feature order, training keys/row digests/counts equals accepted EXP012; all6first-primary estimator parameter dicts equal. Accepted original runtime is sklearn1.9.1/numpy2.4.6; direct startup without environment DLLPATH failed threadpoolctl0xc06d007f before first estimator fit. Proper Library/bin+DLLs+environment+Scripts PATH resolves4thread check. Preregistered oneF0control probe in accepted stack reproduces all54000reference probabilities EXACTLY; this proves runtime-stack restoration for that fold, not which package causes drift or full12fold retraining parity. Full original candidate schedule/fitted audit NOTRUN; no original candidate fit everstarted. Total original complete controls24bundles/144estimators;1interruptedF0bundle plus1pre-fit DLLfailed bundle entry,26conservative totalstarts. Sharedfeaturecampaign46bundles/276estimators; combined70complete bundles/420estimators plus unknown interrupted work. Allattempts preserved. Stop furtherfits; finalreport before04:44Z. No gate/parameter/source/config change or tolerance relaxation.


Final root probe audit: accepted-runtime independent read-only replay validates54000probabilities exactly and all6parameter dictionaries. Initial1.8cross-runtime bitwise replay rejected1e-16–2e-16differences; warning/failure and later exact1.9pass preserved. No tolerance or gate relaxed. Scientific work ended04:23:13Z before authorized04:24:19Zstop. v0byteidentity/all19pins,18campaign code/8source/10topartifact hashes and all21feature-audit hashes pass finalauthentication. Source archive v3 contains53verifiedfiles. Original fullstudy remains incomplete/no candidate; allresearch processes terminal at closure.

## 30. Stronger-model research registration and instructor-source refresh — October 22, 2026

### 30.1 Plan amendment executed; model work remains Pending

At the user's request, EXPERIMENT_PLAN.md §31 now controls the forward research program. New studies are EXP-036 (residual MLP/TabM), EXP-037 (GRU/TCN and conditional sequence transformer), EXP-038 (RelGNN), EXP-039 (RelGT), EXP-040 (graph representation ablation), and EXP-041 (neural–tree ensembles). The amendment specifies finite searches, earlier August inner-fold configuration selection, purged epoch selection and full 56-origin refits, controls/ablations, resources, source isolation, chronology, nomination and confirmation rules. October 25 is a progress checkpoint; the main research allocation extends through November 3, with later exploratory work permitted subject to actual evidence eligibility and the November 11 deadline.

No new model is implemented or fitted by this edit. EXP-036–041 remain Pending. Original EXP-013 remains Ongoing/incomplete, EXP-015 remains a completed negative, and the seven §30-plan feature arms retain their accepted negative findings. No confirmation label or score has been built/read in this session; persistence remains the default until applicable confirmation passes.

The pre-amendment plan is saved byte-identically at `reports/research_refresh_20261022/EXPERIMENT_PLAN_before_research_amendment.md`, SHA256 `1b69a69d549fd3e2d474b6c97b3d3e272e8674e57fa316072d469b3d2c8cc046`. Confirm-plan v0 remains SHA256 `5a96a4f3222a084d8d52c3d1d73133e176992037a48389a609da1078def7bdbc`; no old scientific manifest was edited. Primary matched research retains old instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`; refreshing the active course path is a distinct source operation, with its measured receipt appended below after validation.

### 30.2 Course-data refresh completed and verified

**Installed:** `2026-10-21T16:16:07.9928829Z` = October 22, 00:16:07 Taiwan. The instructor README explicitly says not to use `git pull` because every daily update force-replaces the prior commit. A fresh shallow clone was validated before replacing the active path; the original checkout was moved intact to a dated archive. No course history was merged, no old data was deleted, and no model/label build ran.

- **Remote:** `https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git`.
- **Previous source:** `506d860693df60a698b68aa63a8edb4a984c8e1b`; all eight table hashes still match the frozen feature-campaign startup audit.
- **Active source:** `2053302ba75751be975b9e779f567557b66f4071` at `assignment2/`, matching the checked remote HEAD and the successful clone.
- **Preserved full checkout:** `assignment2_backup/506d860693df60a698b68aa63a8edb4a984c8e1b_20261022/assignment2/`, including its Git metadata and both untracked evaluation bytecode files. Their hashes, all old/new table hashes and support-file hashes were rechecked after installation. Active checkout has no tracked/untracked changes.
- **Primary neural research data path:** explicitly use the preserved full checkout's `data/db/` and assert its loader SHA; active `assignment2/data/db/` is the updated-data lane. Table-only snapshots are useful hash mirrors but do not carry the instructor Git identity themselves. Never let an old-source experiment silently take the refreshed active directory by default.

| Table | Previous rows | Updated rows | Delta |
|---|---:|---:|---:|
| Users | 45,590 | 45,813 | +223 |
| Repositories | 109,081 | 109,917 | +836 |
| PullRequests | 301,518 | 304,018 | +2,500 |
| Issues | 172,007 | 173,805 | +1,798 |
| IssueComments | 106,204 | 106,747 | +543 |
| PRReviews | 90,710 | 91,306 | +596 |
| PRReviewComments | 66,422 | 66,863 | +441 |
| Pushes | 1,661,256 | 1,669,842 | +8,586 |
| **Total** | **2,552,788** | **2,568,311** | **+15,523** |

**Integrity checks:** Git object verification completed without errors. All eight PKs are unique/non-null. All 20 declared FK relations have zero non-null orphan references. User/repository stable GitHub identifiers are unique/non-null. The stable-ID cohort is identical (9,000 users); no cohort internal ID happened to change in this release, but cross-release joins must still use stable IDs. All eight Arrow schemas are unchanged. The only changed inspected support file is `data/db/SCHEMA.md`, and its diff contains only the eight row-count headings; evaluator, course README and data/evaluation README hashes are unchanged. `evaluation/test_input.csv` is not yet present, consistent with the advertised October 28 release.

**Coverage without outcome inspection:** `OfficialData.latest_observed()` is `2026-10-21T11:59:00Z` = October 21, 19:59 Taiwan. This is the actual pipeline's cohort observation maximum, not a completeness certificate or the maximum of every row in every table. Per-table maxima are separately recorded in the audit JSON. Relative to the registered `S+12d21h` proxy, October 7 at 16:00 UTC now passes its October 20 13:00 UTC threshold; October 14, 21 and 28 do not pass. No October label matrix, outcome summary, candidate score or policy decision was produced. Passing source maturity alone does not fix the previously documented v0 harness/schema compatibility issue or authorize relaxing any freeze/exposure rule.

**Operational attempts:** the first sandboxed clone failed before access with `getaddrinfo() thread failed to start`. The authorized network clone then stalled before receiving a pack; its exact process identity was checked and only that transport was stopped. A premature input audit against that incomplete staging tree failed with FileNotFoundError; it produced no passing receipt and no active-path change. The successful retry used HTTP/1.1 with a 30-second low-speed timeout and received a complete 178-MiB Git pack. Partial staging is retained under `assignment2_new/refresh_20261022/`; the successful staging checkout was moved to active `assignment2/`. No trial/model budget was consumed by these operations.

**Reproducibility evidence:** `reports/research_refresh_20261022/audit_data_refresh.py`, `source_audit_before_install.json`, `install_verified_source.ps1`, and `installation_receipt.json`. The input audit records full per-table hashes/schemas, all FK/null counts, stable cohort fingerprint, support/untracked-file hashes and four maturity outcomes. The installer checked resolved move paths stay within the workspace, authenticated the source before each move, retained the complete old checkout, and checked both directories afterward. Input-audit receipt SHA256: `910f8e231729275345c0e50687c985a1622bd35f1962e1e311f1c13f784fec0b`. Python/PowerShell script syntax checks passed. These checks validate the data operation; no new training regression suite or predictive result is claimed.

**Final verification:** at `2026-10-21T16:19:07.9584302Z`, a second remote HEAD check still matched the active release. At `16:19:42Z`, all eight recorded final checks passed: the 14 new research sections exist, their local artifact links resolve, both active/archived loaders report the expected instructor SHAs, the preserved plan and v0 retain their exact hashes, all 19 v0 source/config pins match, and active HEAD matches remote HEAD. Evidence: `remote_HEAD_verification.json` and `final_verification.json` in the same report directory. Documentation whitespace checks pass. New model fits = 0; confirmation label builds = 0.

## 31. Luna Max source-migration amendment — October 22, 2026

### 31.1 Authority, scope and preserved history

The user's active `/goal` selects refreshed instructor release `2053302ba75751be975b9e779f567557b66f4071` for all new research data, caches, controls and models. This score-free amendment, mirrored in EXPERIMENT_PLAN.md §32, supersedes only the old-source-first direction in plan §31.4 for this execution. The previous instructor release `506d860693df60a698b68aa63a8edb4a984c8e1b`, its source-specific findings, caches, predictions, manifests, exposure history and statuses remain historical and unchanged. No old-source feature or prediction is reused as a new-source control.

The pre-migration living documents were preserved byte-identically in `reports/neural_refresh_20261022T0033Z/`: plan SHA256 `6251c28b86a88b1bcb63331ad00c0d4239e2abbc5f02226218972fb6720b94a0`; log SHA256 `ac56a96aa4cdc4edc5744b801bde59d1d28bcf06ddf1b049d8c3e8adbeff55b0`. The active campaign directory is `reports/neural_refresh_20261022T0033Z/`; its state, source manifest, resolved protocol, progress ledger, exact commands and final report are maintained there. Derived caches and outputs use the separate empty root `outputs/neural_refresh_20261022T0033Z/`.

### 31.2 Refreshed-source identity

The active `assignment2/` checkout reports SHA `2053302ba75751be975b9e779f567557b66f4071`, matching the refresh receipt, remote-HEAD verification and successful installation receipt. The prior full checkout remains preserved at `assignment2_backup/506d860693df60a698b68aa63a8edb4a984c8e1b_20261022/assignment2/`. The eight source table SHA256 values independently read from active `assignment2/data/db/` are:

| Table | SHA256 |
|---|---|
| Users | `daaf17c9758563b509635d35fbda193cdfae14f2bac338a9144065715316e8c4` |
| Repositories | `b48ac41fcb14cbb424e2cd6e0522b7d75f5bf01886b876395780ae2e8d2c5452` |
| PullRequests | `36fc044e5e8f064b1e36330f0e9dcd5d2d5aba8dfb253b7817adeDae2da6dbc9` |
| Issues | `a3be147ac671ec041a19c19d59be3b770c317d57b0752792b7c4944a1f528638` |
| IssueComments | `6967c2011fb99a3f02a3edfbef165c4d8287ba1a7c1164d55ccf61021931d5c8` |
| PRReviews | `9dba90cc745533a62d6f16f10676432477b6f27d72d7dcd9dae418f7066d58c1` |
| PRReviewComments | `94f8bd83c4cfd348c3f890578e580e7abf0872dfca83a47517526f12cd3d40a0` |
| Pushes | `30bdb59d19016fe956794566e318048ae09eb55a363f1eda13b4e6bce1ea0aa9` |

All eight matched the source-refresh audit. The supporting receipt remains `reports/research_refresh_20261022/{source_audit_before_install.json,installation_receipt.json,final_verification.json}`. This confirms snapshot identity and input coverage only; it supplies no new model or label result.

### 31.3 Registered work order and restrictions

Before any new-source fit, rebuild point-in-time research features, labels, persistence inputs and source-specific caches from the full refreshed snapshot, using new cache/output directories. Refit matched P_persist7, REF/F0 HGB and H2/F0 HGB with identical within-release keys, labels and registered settings. Save complete predictions and policies, alignments, official scorer parity, maturity, source deltas on stable-ID intersections, and every audit required by plan §31 and the goal contract. Do not read October target-week outcomes or build October labels.

The active scientific order is Step 0, EXP-036-R, EXP-036-T, EXP-041-tabular immediately after both complete tabular finalists, EXP-037-G, then EXP-037-C and its temporal diagnostic controls. All original neural grids, training/refit schedule, preprocessing contrast, blend cells/policy, sequence definitions/diagnostics, verification requirements, resource limits and acceptance criteria remain as registered. The all4/mature3 pools remain fixed; source-specific maturity is recalculated and recorded. Historical studies and reserved October outcomes remain untouched. Work runs directly in this task with at most one trainer at a time and no subagents.

This entry is registration only. At entry time no refreshed-source research datasets, caches, matched baselines or neural fits have been accepted. It makes no performance claim and does not change EXP-036–041 to Completed. Progress, every attempted configuration, outputs, resource evidence, repairs and exact reproduction/resume commands are recorded in the campaign ledger.

## 32. Step 0 refreshed-source data and matched controls — Completed

**Campaign:** `neural_refresh_20261022T0033Z`; all timestamps are UTC unless marked Taiwan time. Step 0 used only the refreshed instructor release `2053302ba75751be975b9e779f567557b66f4071` from `assignment2/data/db`. The exact eight active table hashes, source identity, frozen protocol, and ordered F0 schema are recorded in the campaign source manifest. The old source `506d860693df60a698b68aa63a8edb4a984c8e1b` was read only to quantify stable-ID source deltas; its feature caches and predictions were not used as controls.

### 32.1 Data and source deltas

The source-aware build completed `2026-10-21T16:45Z`. It materialized 100 feature origins (797,182 user-origin feature rows), 403 separate source/as-of label cache keys, and the exact ordered 59-column F0. All 100 feature caches and all 403 label caches passed key, source/table hash, builder hash, schema, row count, population digest, file digest, and value-digest authentication. The build and cache manifests are under `outputs/neural_refresh_20261022T0033Z/data_build/`. The source observation maximum is `2026-10-21T11:59:00Z`. All four fixed September report origins pass the preregistered source-specific `S+12d21h` maturity proxy. No October-origin labels or target outcomes were built/read.

Cross-release matching used stable GitHub `UserID` values, never row PKs. At September 9/16/23/30, the refreshed populations are 6,382 / 5,862 / 5,497 / 5,178 users; each matched old-source population is identical (Jaccard 1.0, no one-sided IDs). At the common observation cutoff `2026-10-13T12:59:29Z`, 0 of 24 origin-target label cells changed. F0 changed for 0 / 1 / 1 / 1 users and 0 / 7 / 4 / 3 cells, respectively. The stable-intersection audit and change rows are in `outputs/neural_refresh_20261022T0033Z/data_build/source_delta_audit/`.

### 32.2 Fresh control results

Completed 48 sequential per-target HGB fits: four report origins × six targets × REF/H2. All REF estimators use no recency weights; all H2 rows use the exact registered half-life-two-week formula, normalized once to mean one. Each fit used all 56 permitted training origins, the registered HGB overrides, the accepted runtime's full resolved estimator defaults, and refreshed-source within-release keys/labels. The 22,919 report user-origin rows contain probabilities, labels, persistence flags, EW7 values and P016/P021 decisions. Checkpoints, configuration, process metadata and fit manifests are under `outputs/neural_refresh_20261022T0033Z/controls/`.

| System | All4 combined F1 | Mature3 combined F1 | All4 mean origin six-target AP | Mature3 mean origin six-target AP |
|---|---:|---:|---:|---:|
| P_persist7 | 0.516555 | 0.514126 | — | — |
| REF/P016 | 0.535302 | 0.530718 | 0.528543 | 0.532114 |
| REF/P021 | 0.544283 | 0.540260 | 0.528543 | 0.532114 |
| H2/P016 | 0.538366 | 0.534016 | 0.535030 | 0.537493 |
| H2/P021 | 0.547634 | 0.544169 | 0.535030 | 0.537493 |

The refreshed-source results are accepted as controls. H2 remains the registered matched neural baseline. The new-source control scores are not compared with old-source predictions as if source had been held fixed.

### 32.3 Verification and repair record

`experiments.neural_research_common.verify_controls` independently reauthenticated all source table hashes and caches; checked complete ordered prediction keys, labels and stable-ID populations; reconstructed the 56-origin training schedules, labels, H2 row weights and digests; independently recomputed P_persist7, EW7, average-rank P016/P021 with ascending-UserID ties, per-origin and pooled metrics/AP; matched the official evaluator's all-negative convention for each of five systems at each origin; and loaded all 48 target checkpoints for sample inference parity. It passed on all4 and mature3. Evidence is `outputs/neural_refresh_20261022T0033Z/controls/verification.json` (SHA256 `95298db7d23240384039d1fdcc5465f9cf4a4c4e2aac3036c1ee691b5186c8df`); the detailed console log is `verification.stdout.log`.

The data builder's scorer-acceptance field was corrected to false before any score claim, because schema validation had run before scorer parity. It was set true only after the independent verifier passed; both transitions and the verifier digest remain in the data-build manifest. The first trainer attempt hit a Windows `threadpoolctl.threadpool_info()` native-library error before fitting any model. The runtime logger now records the exception while retaining package versions and estimator defaults; the subsequent attached HGB trainer completed 48/48 fits. A pooled-AP aggregation condition left only that summary field blank; the condition was repaired and metrics were regenerated from the saved predictions. `train_controls_executed_v1.py` was reconstructed and its SHA256 matches the fit manifest (`aa5645d6bd951af4b67ed6cccb996019b2c24ea3f61acdf3dcb3f33f825f9738`). No fit/checkpoint was overwritten.

**Status at this amendment:** Step 0 is completed and independently verified. EXP-036-R is completed and verified negative under the unchanged gate (§35). EXP-036-T's fit and independent verification are recorded in §36; EXP-041-tabular is the active next package. EXP-037-G and EXP-037-C remain pending in the registered order. No October outcome or November test score has been inspected.

## 33. EXP-036-T official TabM pin and constructor preregistration — Score-free

Before any TabM fit, upstream source and dependency were fetched and pinned to official release commits: [Yandex Research TabM `a507095893d784c5702059d737ddfbd1299c41dd`](https://github.com/yandex-research/tabm/commit/a507095893d784c5702059d737ddfbd1299c41dd), package version 0.0.3, Apache-2.0; and [rtdl-num-embeddings `7381c5a4ed01253fbef9ea8cfed137f907265552`](https://github.com/yandex-research/rtdl-num-embeddings/commit/7381c5a4ed01253fbef9ea8cfed137f907265552), version 0.0.12. Byte-identical upstream modules and license files are installed under the campaign's isolated `outputs/neural_refresh_20261022T0033Z/third_party/python/` import root; global environments were not changed. The package paths, SHA256 hashes, licenses, runtime, versions, and API checks are sealed in `third_party/tabm_install_receipt.json` and `reports/neural_refresh_20261022T0033Z/tabm_pin.json`.

The verified official API is `tabm.TabM.make` with `n_num_features=59`, no categorical inputs, `d_out=6`, `num_embeddings=None`, `n_blocks=3`, `d_block in {128,256,512}`, `dropout=0.1`, `activation='ReLU'`, `k=16`, `arch_type='tabm'`, and `start_scaling_init='random-signs'`. All three constructor probes returned `(batch,16,6)` and an official `MLPBackboneBatchEnsemble`; parameter counts are 70,032 / 204,560 / 670,224. There is no adapter patch and no custom MLP under the TabM name. The scientific grid remains the six registered width × learning-rate configurations. Train the 16 member logits with equally weighted per-member six-target BCE using the registered row weights; average member probabilities for inference, then average the three seed predictions.

After EXP-036-R was fully verified, EXP-036-T's exact source-aware search/refit protocol was frozen score-free at `outputs/neural_refresh_20261022T0033Z/exp036/tabm/preregistration.json` (SHA256 `0065a106719740ce9dbe3f4e9ab6feb066c60ad6c17bf5ddc8a86571698a1ee3`). The registration includes the official TabM and rtdl-num-embeddings commit/source/license hashes from the preceding pin, the six width × learning-rate configs, both August selection folds, the four September report origins, seeds 42/43/44, H2 weights, per-member BCE, member-probability averaging, full-window refits, resource limits and runner/core hashes. Exact preregistration command: `C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe -m experiments.exp036_neural_tabular.run --family EXP-036-T --run-dir outputs/neural_refresh_20261022T0033Z --preregister-only`. At this freeze, no TabM fit or score had been read.

## 34. EXP-036-R residual MLP preregistration — Score-free

The first tabular neural fit is registered as **EXP-036-R** on the refreshed source in §32, using the shared protocol in §31. Its exact score-free preregistration is `outputs/neural_refresh_20261022T0033Z/exp036/residual_mlp/preregistration.json` (SHA256 `3072144db4834a6c3da4b49aad9631d721aa79d4ca586c1524e4b366f138f55f`); the byte-preserved superseded preregistration is `preregistration_superseded_b8a56cb6294b.json`. This freeze names the six width × learning-rate configs, exact residual block and six-logit head, primary signed-log preprocessing, the fixed raw-continuous preprocessing contrast, 42/7/7 purged inner folds, two August selection origins, four September reports, seeds 42/43/44, H2 row weights, 128-epoch/patience-12 checkpoint rule, fresh 56-origin refits, FP32 AdamW settings, batch/OOM retry, and 18 GPU-hour family ceiling. Full source table hashes, F0 order, runtime and implementation digests are recorded with it.

Before any fit, 9 focused contract tests passed (five existing source/timing/cache tests and four preprocessing/model/AP checks). The bounded synthetic training sanity artifact is `outputs/neural_refresh_20261022T0033Z/exp036/bounded_training_sanity.json`: BCE decreased from 0.691604 to 0.605188 over 12 epochs on CUDA; shuffled-label digest is recorded and the artifact explicitly marks this as non-predictive. A first score-free input-reader attempt failed because it looked for the report observation cutoff in the label-index manifest; no trainer or fit started. The reader was corrected to use `data_build_manifest.json` and assert refreshed source, table hashes and the no-October-label flag. The following preregistration pass authenticated all cached inputs. The exact command is `C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe -m experiments.exp036_neural_tabular.run --family EXP-036-R --run-dir outputs/neural_refresh_20261022T0033Z --preregister-only`.

At the time of preregistration, the six-config MLP search and all finalist/contrast refits had not run. Its subsequent execution and verification are recorded in §35.

## 35. EXP-036-R residual MLP — Complete search, finalist and preprocessing contrast

### 35.1 Search and frozen selection

The preregistered six width × learning-rate configurations were each fitted on both August tuning folds (12/12 search fits). Selection used the registered mean six-target AP rule; configurations within 0.0005 of the top mean were treated as tied, then fewer parameters and lexical config ID broke ties. `w512_lr0.001` had the highest observed two-fold mean (0.271076); `w256_lr0.0003` scored 0.270594, fell within the registered tie tolerance, and was selected on its lower parameter count (807,686 vs 3,188,230). The frozen `search_selection.json` was written before report refits.

| Config | Two-fold mean AP | Fold A | Fold B |
|---|---:|---:|---:|
| w128_lr0.0003 | 0.269572 | 0.281939 | 0.257205 |
| w128_lr0.001 | 0.270146 | 0.280885 | 0.259407 |
| w256_lr0.0003 | 0.270594 | 0.279550 | 0.261637 |
| w256_lr0.001 | 0.270002 | 0.281110 | 0.258893 |
| w512_lr0.0003 | 0.270569 | 0.280261 | 0.260876 |
| w512_lr0.001 | **0.271076** | 0.280280 | 0.261871 |

### 35.2 Primary finalist and fixed preprocessing contrast

The selected configuration completed all 12 primary report fits: four fixed September origins × seeds 42/43/44, each with a separate inner epoch-selection checkpoint and a fresh full-56-origin refit. The same schedule and seeds were executed for the single registered raw-continuous preprocessing contrast. Primary preprocessing used signed-log continuous F0 followed by fit-only standardization/clipping; contrast kept binary inputs unchanged and standardized raw continuous F0 without signed-log transformation. Both use the same six targets, H2 row weights, optimizer, early-stop rule, rank ensemble and P016/P021 decision policies. No OOM, source mismatch or report-key mismatch occurred.

| Arm and scope | Mean origin six-target AP | AP delta vs H2 | P021 combined F1 | P021 delta vs H2 |
|---|---:|---:|---:|---:|
| Primary MLP all4 | 0.518744 | -0.016286 | 0.545742 | -0.001892 |
| Primary MLP mature3 | 0.516276 | -0.021217 | 0.541222 | -0.002947 |
| Raw-F0 contrast all4 | 0.517974 | -0.017055 | 0.543782 | -0.003852 |
| Raw-F0 contrast mature3 | 0.519219 | -0.018274 | 0.539203 | -0.004966 |
| Matched H2 all4 | 0.535030 | — | 0.547634 | — |
| Matched H2 mature3 | 0.537493 | — | 0.544169 | — |

For context, all4 P021 scores were REF 0.544283 and P_persist7 0.516555; mature3 scores were REF 0.540260 and P_persist7 0.514126. The primary MLP's all4 AP deltas by report origin were +0.010007, -0.044574, -0.029084 and -0.001492; the fixed raw-F0 contrast deltas were -0.003885, -0.015567, -0.035369 and -0.013400. Neither preprocessing arm improved AP on at least three of four report origins.

The registered development gate requires all4 mean AP delta ≥0.002, improvement on at least three origins, all4 P021 F1 delta ≥0.001, mature3 P021 nonnegative, and no weekly P021 delta below -0.010. The primary MLP and raw-F0 contrast each passed only the final weekly guard; both are **ineligible**. The preprocessing contrast did not rescue the model.

Paired stable-user bootstrap uncertainty (1,000 draws; repeated report rows for each UserID stay together) supports the negative AP result. Primary P021 AP delta vs H2 was -0.016286 with 95% interval [-0.021568, -0.009271] on all4 and -0.021217 [-0.027613, -0.013300] on mature3. Primary P021 combined-F1 deltas were -0.001892 [-0.004825, 0.001071] and -0.002947 [-0.006550, 0.000133]. Raw-F0 AP deltas were -0.017055 [-0.022563, -0.010115] and -0.018274 [-0.025122, -0.010410]; P021 combined-F1 deltas were -0.003852 [-0.007322, -0.000546] and -0.004966 [-0.008229, -0.001604].

### 35.3 Independent verification, resources and process checks

The independent saved-output verifier passed for both arms. Each has 12 fitted origin/seed results and 24 checkpoint sample predictions. Primary maximum checkpoint-to-saved probability error was `2.384185791015625e-07`; contrast maximum was `1.9371509552001953e-07`, both below 1e-6. Source SHA and all eight source-table hashes match §32; prediction and matched-control key/label alignment passed; policies were independently reconstructed with stable-ID tie rules; and the official scorer matched every saved system on each report origin. Outputs include per-target metrics, pooled all4/mature3 scores, P016/P021 decisions, paired bootstrap, seed dispersion, resource summary, worst week, prediction digests and the development-gate artifact under `outputs/neural_refresh_20261022T0033Z/exp036/residual_mlp/`.

The bounded synthetic shuffled-label control passed after the real MLP search had begun: the ordinary synthetic-label run reduced training loss 0.690492→0.608916 with mean six-target held-out AP 0.694249; three independently shuffled-training-label runs averaged 0.527570 held-out AP (each lower than baseline). Its saved note explicitly excludes it as predictive evidence and records the timing. The earlier bounded CUDA training sanity also remains non-predictive. The verifier's first finalist invocation exposed a KeyError because the H2 AP summary was absent after the bootstrap had finished. No model fit failed or changed; the verifier was corrected to include REF/H2 AP summaries and a batched weighted-AP bootstrap. Four focused verifier tests, including tied-score weighted AP comparison against scikit-learn, passed before both complete verification reruns succeeded.

The primary family consumed 586.5 measured GPU seconds across search, inner epoch selection and full-window refits; the contrast consumed 278.1 GPU seconds. Across logged fits, peak allocated/reserved CUDA memory was 198,734,848/260,046,848 bytes. The run stayed far below the 18 GPU-hour family ceiling; trainers ran serially, with no subagents and no OOM retry.

### 35.4 Reproduction and artifacts

Reproduce the primary search/refit and fixed contrast with:

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp036_neural_tabular.run --family EXP-036-R --run-dir outputs/neural_refresh_20261022T0033Z
```

The completed authenticated checkpoint set resumes without refitting already matching fits. Recompute primary and contrast verification with:

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp036_neural_tabular.verify --family EXP-036-R --run-name finalist --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp036_neural_tabular.verify --family EXP-036-R --run-name preprocessing_contrast --run-dir outputs/neural_refresh_20261022T0033Z
```

Bounded process-control reproduction:

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.neural_research_common.run_shuffled_label_sanity
```

The score-free preregistration remains in `outputs/neural_refresh_20261022T0033Z/exp036/residual_mlp/preregistration.json`; frozen search selection and fit manifests, console log, raw/report predictions, checkpoints, sidecars and both verification directories remain in that output tree. Campaign state, source hashes, exact resume commands and this completion status are in `reports/neural_refresh_20261022T0033Z/`.

## 36. EXP-036-T official TabM — Complete search, finalist and verification

### 36.1 Score-free registration and six-config search

After both EXP-036-R arms were verified, the TabM search/refit protocol was frozen score-free at `outputs/neural_refresh_20261022T0033Z/exp036/tabm/preregistration.json` (SHA256 `0065a106719740ce9dbe3f4e9ab6feb066c60ad6c17bf5ddc8a86571698a1ee3`). The registration uses the official, isolated TabM v0.0.3 implementation and rtdl-num-embeddings v0.0.12 source pins recorded in §33, with no adapter patch. Its fixed 16-member, three-block, dropout-0.1, no-numerical-embeddings model uses the six registered width × learning-rate cells, memberwise weighted BCE, H2 weights, two August tuning origins, 42/7/7 purged inner folds and the common September three-seed finalist schedule.

All six configs × two folds completed (12/12 fits). The selected config was `w128_lr0.001`, with 70,032 parameters and two-fold mean AP 0.285605. Selection was frozen before opening September report scores.

| Config | Two-fold mean AP | August 19 | August 26 |
|---|---:|---:|---:|
| w128_lr0.0003 | 0.283992 | 0.295993 | 0.271992 |
| w128_lr0.001 | **0.285605** | 0.298726 | 0.272485 |
| w256_lr0.0003 | 0.282668 | 0.293909 | 0.271428 |
| w256_lr0.001 | 0.283447 | 0.294047 | 0.272847 |
| w512_lr0.0003 | 0.281676 | 0.292222 | 0.271129 |
| w512_lr0.001 | 0.283470 | 0.293514 | 0.273426 |

### 36.2 Outer refits and verified scores

The selected TabM completed all 12 September report fits: four origins × seeds 42/43/44, each with an inner epoch-selection fit and a fresh 56-origin refit. The 16 TabM members were averaged within each seed; the three seed predictions were then averaged for the ensemble. All fits used the refreshed source, same populations, labels, targets, weights and frozen policies as Step 0 and EXP-036-R.

| Arm and scope | Mean origin six-target AP | AP delta vs H2 | P021 combined F1 | P021 delta vs H2 |
|---|---:|---:|---:|---:|
| TabM all4 | 0.537569 | +0.002539 | 0.547965 | +0.000330 |
| TabM mature3 | 0.541200 | +0.003707 | 0.543702 | -0.000466 |
| H2 all4 | 0.535030 | — | 0.547634 | — |
| H2 mature3 | 0.537493 | — | 0.544169 | — |

For comparison, all4 combined F1 for TabM P016 was 0.540988 versus H2/P016 0.538366; P021 REF and persistence controls were 0.544283 and 0.516555. Mature3 TabM P016 was 0.535679 versus H2/P016 0.534016; REF/P021 and P_persist7 were 0.540260 and 0.514126. Per-target AP, precision, recall, F1, support and predicted positives for all4 and mature3 are in `finalist/verification/per_target_metrics_pooled.csv`; per-origin and all pooled systems are in the adjacent metric CSV files.

By report origin, TabM mean six-target AP was 0.582086 / 0.537546 / 0.503968 / 0.526675, compared with H2's 0.569152 / 0.534791 / 0.508537 / 0.527639. Corresponding AP deltas were +0.012934 / +0.002755 / -0.004569 / -0.000964. P021 combined F1 deltas versus H2 were -0.000638 / -0.001544 / +0.000033 / +0.003314. Its worst week was September 16 (P021 combined F1 0.517728; H2 0.519272).

The unchanged development gate is **not met**. TabM passed the all4 mean AP delta threshold (+0.002539 ≥ +0.002) and weekly guard, but AP improved on only two of four origins; all4 P021 combined-F1 gain (+0.000330) is below +0.001; and mature3 P021 is negative (-0.000466). The all4/mature3 stable-user bootstrap AP-delta intervals were [-0.001865, +0.006942] and [-0.000754, +0.008850]; paired P021 combined-F1 intervals were [-0.002155, +0.002825] and [-0.003418, +0.002024]. Both AP and F1 intervals include zero. TabM is a completed, verified development-gate negative and is not nominated as a standalone model.

Seed dispersion is saved in `finalist/verification/seed_dispersion_summary.csv`: all4 mean-origin AP across three seeds was 0.536161 (population SD 0.002437); P021 all4 pooled combined F1 was 0.548100 (SD 0.000295; range 0.547794–0.548498).

### 36.3 Verification, resources and reproducibility

The saved-output verifier passed source SHA and all eight source-table hashes, matched-control keys/labels, independent P016/P021 recomputation, and official scorer parity for every saved system and report origin. It reloaded 12 checkpointed fits and 24 sample predictions; maximum checkpoint-to-saved probability error was `1.1920928955078125e-07` under the 1e-6 tolerance. Search, report predictions, per-target metrics, bootstrap, worst-week/seed-dispersion/resource summaries and `verification.json` are under `outputs/neural_refresh_20261022T0033Z/exp036/tabm/`. Verification SHA256 is `7e54236372071d95e04d901b71816263e69b1d1bc6cc37a45c632144286b8b5`; prediction SHA256 is `f9fc60b3434323db53dee85237e6d499e7900b8ea7a7150259aac25b6d92edd8`.

Measured GPU time was 3,290.8 seconds total: 2,315.9 search, 597.6 inner epoch selection and 377.3 fresh full-window refit seconds. Peak allocated/reserved memory was 916,872,192 / 1,528,823,808 bytes. No fit exceeded its wall limit, no OOM/retry occurred, and use was under one GPU hour. The single trainer ran serially; no subagents were used.

Exact fit/resume and verification commands:

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp036_neural_tabular.run --family EXP-036-T --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp036_neural_tabular.verify --family EXP-036-T --run-name finalist --run-dir outputs/neural_refresh_20261022T0033Z
```

EXP-041-tabular was the next work package. Its two-member subset, twelve blend cells, September 9/16 selection tie rule and September 23/30 transfer replay were frozen score-free before opening any blend score; this child does not close the broader EXP-041 study.

## 37. EXP-041 tabular-child registration — score-free

The EXP-041 tabular subset was registered after the independently verified Step 0 controls and complete residual MLP and official TabM finalists. Its immutable preregistration is `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/preregistration.json` (SHA256 `67851fb6cea19739c05342305b136c9c81d4850a65a068bd1835920193abdcda`). Child ID: `EXP-041-tabular-MLP-TabM-SeptemberBlend-v1`. The registration fixes the 12 cells (MLP/TabM × alpha 0.10/0.25/0.50 × P016/P021), average-rank normalization, stable-GitHub-ID tie resolution, P_persist7 k, fresh-source EW7, the September 9/16 selection-only rule, tie tolerance/order and September 23/30 transfer gate. The full six-family EXP-041 remains Ongoing.

The preregistration binds instructor source `2053302ba75751be975b9e779f567557b66f4071`, all eight refreshed table hashes, the Step 0 control file and verification, and the primary MLP/TabM finalist prediction, fit-manifest and verification hashes. Prediction hashes are H2 controls `eeef8440dab47a5b986da765bc454099142acf6ac44b2d5a11547716d7a1ff9b`, MLP `540795e7bdfd2a6873a598dd7a79879de96d2069ad9ee21e87df722d93c99f77`, and TabM `f9fc60b3434323db53dee85237e6d499e7900b8ea7a7150259aac25b6d92edd8`. Input preflight used file/schema/manifest metadata only; no September blend score was read when this registration was written. The first metadata preflight stopped on a lowercase member-to-parameter lookup before writing the preregistration or reading scores. The key mapping was corrected and the score-free preflight rerun successfully; no scientific setting changed.

The selection command reads and scores only September 9/16. It writes `selection.json` and its SHA256 sidecar before the transfer command can materialize September 23/30 rows. The independent verifier reconstructs key/label alignment, every blend score and decision, the frozen tie break, per-origin/all4/mature3 metrics, per-target TP changes, rank correlation/disagreement, official scorer parity, and the stable-user bootstrap. No output from either score phase is yet included in this registration entry.

The first selection invocation stopped at the registered H2-decision parity assertion because the input reader had not loaded the saved Step 0 P016/P021 decision columns. It created no selection score table or frozen child choice and did not materialize September 23/30. The permitted September 9/16 rows were materialized for that attempt; no metric values were printed, persisted or used for a choice. Before retrying, an append-only code repair was registered at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/implementation_amendment_01.json` (SHA256 `afbdbf53c7b2d08bd905796efe486fd7782692cce10105783227bccc042acd79`). It only joins already verified H2 decision columns by `(origin, UserID)` so the original preregistered code can execute its parity assertion. The original scorer and verifier files remain byte-identical to the preregistered hashes; the amendment binds the replay and verification wrappers. No score formula, policy, tie rule, selection rule or transfer gate changed.

The first repair wrapper itself then stopped with a recursion error before invoking the original loader; no rows were materialized and no blend calculation ran in that second failed attempt. This is recorded in append-only implementation amendment 02 at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/implementation_amendment_02.json` (SHA256 `d239d3997c4389cb59a92c44c9081786dec31e320a82229e08f53138c83e1fbd`). The second wrapper retains the original loader reference before adding the H2-decision columns. Both repairs are code-only, each registered before retry, and leave the original scoring runner and frozen candidate-selection procedure intact. The selection scores are still pending; transfer outcomes have not been opened.

The next selection retry exposed that the reader also needed the saved `P_persist7` decisions for its parity check. The process reached the candidate calculation but stopped before writing the score table or selection lock; no metric values were printed or persisted, and no September 23/30 rows were materialized. Implementation amendment 03 (SHA256 `bcf41be968b627b733ae38f4892d4fa74e79079cff1975cd4d70cba4cc2e16de`) binds a final reader adapter that joins H2/P016, H2/P021 and P_persist7 decision columns. It is registered at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/implementation_amendment_03.json`. The frozen original runner and verifier remain unchanged. All three amendment hashes are retained, linked in sequence, and bound into the frozen selection lock, scored manifest and planned verification supplement.

### Frozen selection checkpoint

After the third reader repair, all 12 registered candidate cells scored on September 9/16 only. The maximum mean weekly combined F1 was 0.535833 (`MLP, alpha=0.25, P021`). Three P021 MLP candidates fell within the 0.001 tie band: alpha 0.10 (0.534848), 0.25 (0.535833) and 0.50 (0.535102). Applying the registered lower-alpha tie break selected `EXP-041-tabular-MLP-alpha=0.10-layer=P021`, with Sep 9 combined F1 0.550258, Sep 16 0.519439, and mean 0.534848. This is the only child member/alpha/layer eligible for transfer evaluation; no re-selection is permitted.

The final selection lock, after it was bound to all three repair amendments, has SHA256 `d1f30b755eb1544f5c718737f80c083756d882e49572bcf9a262c21e78c6abb7`. Independent pre-transfer tie/hash validation passed and is saved at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/selection_pretransfer_verification.json` (its sidecar is adjacent). It confirms 12 unique cells, the two permitted selection origins, the tie set/order and all amendment digests. At the pre-transfer lock checkpoint, September 23/30 blend rows had not yet been materialized.

### Transfer replay runtime repair — result pending

After the selection lock was checked, the first full transfer invocation materialized the four registered report populations but exited before writing any report artifact. The captured exit was Windows status `-1066598273`; constraining BLAS/OpenMP/NumExpr threads to one did not resolve it. An input-only probe successfully joined all 22,919 refreshed-source rows, while a bounded synthetic probe isolated the native exit at `np.corrcoef`. No transfer metric was printed or persisted, and the locked child choice was not changed. These attempts and logs are under `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/`.

Report-only implementation amendment 04 was registered before retry (SHA256 `9b7bd76faae975ce19dcbec9f9ac99791bcb52870376e57cf42a1b6ca25e4ec5`). It replaces the failing BLAS-backed Pearson/Spearman diagnostic calculation with equivalent centered sums for correlation and average-rank correlation only. Candidate probabilities, `q`, top-k decisions, AP/F1, selection, transfer gate, and bootstrap remain unchanged. Amendment 04 binds the final transfer wrapper and verifier; it records the already frozen selection hash and does not modify `selection.json`.

Exact reproduction commands, using the accepted HGB/scoring Python environment:

```powershell
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.run --phase preregister --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.run --phase selection --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.run --phase transfer --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.verify --run-dir outputs/neural_refresh_20261022T0033Z
```

The original score-free command strings above are retained as registration history. After input-reader amendments 01–03 and report-only correlation amendment 04, the exact reproduction commands are:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.repair_03 --phase selection --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.repair_04 --phase transfer --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.verify_repair_04 --run-dir outputs/neural_refresh_20261022T0033Z
```

## 38. EXP-041 tabular child — Complete, verified negative

The frozen tabular child `EXP-041-tabular-MLP-TabM-SeptemberBlend-v1` completed its registered 12-cell grid and four-origin scoring. Selection was frozen on September 9/16 before the transfer replay. Transfer consists of the locked September 23/30 pair; it is a development transfer check within an already exposed September study, not independent confirmation. The global six-family EXP-041 study remains Ongoing; this entry completes only its named tabular child.

### 38.1 Full grid results

All twelve cells are shown below. `Selection F1` is mean weekly combined F1 on Sep 9/16. `All4 ΔF1` and Sep 23/30 deltas compare against the same-layer H2 control. `All4 AP` is the six-target AP mean on the fixed all4 origin population.

| Cell | Selection F1 | All4 AP | All4 ΔF1 | Sep 23 ΔF1 | Sep 30 ΔF1 |
|---|---:|---:|---:|---:|---:|
| MLP α=.10 / P016 | 0.526499 | 0.538735 | +0.002311 | +0.002355 | +0.005959 |
| MLP α=.10 / P021 | 0.534848 | 0.553702 | +0.000742 | −0.000186 | +0.001886 |
| MLP α=.25 / P016 | 0.529069 | 0.541045 | +0.005224 | +0.002915 | +0.012440 |
| MLP α=.25 / P021 | 0.535833 | 0.554212 | +0.001397 | −0.001171 | +0.003139 |
| MLP α=.50 / P016 | 0.530465 | 0.539544 | +0.007575 | +0.005396 | +0.016924 |
| MLP α=.50 / P021 | 0.535102 | 0.553784 | +0.001354 | −0.001207 | +0.004502 |
| TabM α=.10 / P016 | 0.525639 | 0.538759 | +0.001753 | +0.002177 | +0.005876 |
| TabM α=.10 / P021 | 0.534195 | 0.553757 | +0.000214 | −0.000155 | +0.001233 |
| TabM α=.25 / P016 | 0.526547 | 0.541826 | +0.003483 | +0.003210 | +0.010610 |
| TabM α=.25 / P021 | 0.534518 | 0.554862 | +0.000673 | −0.001055 | +0.003343 |
| TabM α=.50 / P016 | 0.528324 | 0.543915 | +0.005719 | +0.005546 | +0.014404 |
| TabM α=.50 / P021 | 0.533742 | 0.555577 | +0.000619 | −0.000845 | +0.005092 |

The maximum selection score was 0.535833 for MLP α=.25/P021. Three MLP/P021 candidates were within the registered 0.001 tie band; the lower-alpha rule selected `EXP-041-tabular-MLP-alpha=0.10-layer=P021` (Sep 9 0.550258; Sep 16 0.519439; mean 0.534848). Its Sep 23 delta was −0.000186 and Sep 30 delta +0.001886, mean transfer delta +0.000850. The selected blend's all4 pooled combined-F1 delta was +0.000742 versus both its matched H2/P021 and H2/P021 overall, below the frozen +0.001 gate. The gate therefore **failed**; the child is a completed verified negative.

For the selected blend, stable-user paired bootstrap (1,000 resamples; seed 20261022; all repeated origin rows move together; 6,464 stable-user clusters) estimated all4 combined-F1 delta +0.000742, 95% interval [−0.000291,+0.001629], and six-target AP delta +0.001005, 95% interval [+0.000586,+0.001436]. The interval describes observed-user resampling uncertainty, not temporal generalization. Correlation/disagreement and targetwise true-positive gained/lost tables cover every cell at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/correlation_disagreement.csv` and `prediction_tp_gained_lost.csv`; pooled, per-origin and per-target score tables are adjacent.

### 38.2 Repairs and independent verification

The first transfer replay exited in the Windows native runtime at `np.corrcoef` before persisting metrics. Report-only amendment 04 (SHA256 `9b7bd76faae975ce19dcbec9f9ac99791bcb52870376e57cf42a1b6ca25e4ec5`) replaced Pearson/Spearman diagnostics with centered-sum calculations; candidate scores, decisions, selection, gate and bootstrap were unchanged. The replay then completed and persisted all twelve cells.

While preparing the final score table, a descriptive output defect was found: candidate `layer` labels in two pooled CSVs contained `layer=P016/P021` instead of canonical `P016/P021`. Report correction 05 changed only that column and preserved pre-fix CSVs. The correction changed 24 candidate rows in `cell_pooled_metrics.csv` and 144 in `cell_pooled_target_metrics.csv`; independent comparison confirmed every other field was unchanged. Correction note SHA256 `0a6f3b75962a7e52f41b2c5ba3dc018143821994c2c4bca3749481dc3054b646`; durable metadata verification SHA256 `c3b53914f5ca8886f3ccf9c125177cace85a46ccb8fed0c3fc22ba77728e3477`.

The full independent EXP-041 verifier passed again after the metadata correction. It reconstructed source keys/labels, blend formulas and decisions, all twelve cells, matched controls, official scorer results, pooled/per-target metrics, frozen selection and tie set, gate and stable-user bootstrap. Final verifier SHA256 is `85ed46499f1aa73a3485b8fd5c929036b7c57b69246613b8062cc44e25b13990`; amendment-chain verifier SHA256 is `ed7aa448cf96541275942a98e9398f18cd9977b5b8a2254f2735f99fa00550a2`. Selection lock SHA256 remains `d1f30b755eb1544f5c718737f80c083756d882e49572bcf9a262c21e78c6abb7`.

### 38.3 Exact replay and verification commands

From the repository root, with the accepted HGB/scoring Python environment:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.repair_04 --phase transfer --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.verify_repair_04 --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\nycu-bda-asg2\python.exe' -m experiments.exp041_neural_tree_ensemble.verify_report_repair_05
```

Earlier reader amendments 01–03 and runtime amendment 04 are append-only and retained beside the output. Do not edit the original scoring runner, verifier or frozen selection. Complete artifacts are in `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/`, including blend predictions, all metric tables, gate, bootstrap, preregistration, selection lock and verification files. EXP-037-G is next under the registered one-trainer rule.

## 39. EXP-037-G GRU+F0 — Complete, verified negative

The preregistration was frozen score-free before the eight search fits at `outputs/neural_refresh_20261022T0033Z/exp037/gru/preregistration.json` (SHA256 `389b566244306b9e6efa94022ad6761b7170a3220d3e6e12fa7bcb3f66dc5294`). It binds refreshed source `2053302ba75751be975b9e779f567557b66f4071`, the eight table hashes, sequence-cache definition and manifest, runner/model/core hashes, the four registered GRU cells, two August tuning origins, fit-only normalization, H2 weights, gates, seed schedule, resource limits and all4 September report origins. Search completed 8/8 fits and selected `gru_h64_l1_lr0.0003` (63,686 parameters). The maximum two-fold mean AP was 0.279759 at `gru_h128_l2_lr0.0003`; both low-learning-rate configs were within the frozen 0.0005 tie band, so the fewer-parameter rule selected width 64/layer 1 (mean AP 0.279409). All eight validation files were re-scored from saved labels/probabilities.

The selected model completed 12/12 report-origin × seed inner epoch-selection/full-56-origin refit pairs (24 saved fits total) on CUDA. The ensemble has 22,919 rows across the four registered report origins. The separate verifier rebuilt source populations and labels, training keys/weights and fit-only F0/sequence normalization; checked the 4-config selection/tie rule; authenticated and sampled all 24 checkpoints; matched each of the 12 per-seed output files to the ensemble columns and arithmetic mean; recomputed all operating policies and metrics; and passed official-scorer parity. Maximum sampled checkpoint probability error was `1.79e-7` against tolerance `1e-5`. No October-origin outcome was read.

| Scope | Model | Mean-origin six-target AP | P021 combined F1 | H2 mean-origin AP | H2 P021 combined F1 |
|---|---|---:|---:|---:|---:|
| all4 | GRU | 0.532624 | 0.547042 | 0.535030 | 0.547634 |
| mature3 | GRU | 0.535210 | 0.543173 | 0.537493 | 0.544169 |

The all4 AP delta versus H2 was −0.002406; AP improved on 1/4 origins. P021 combined-F1 deltas versus H2 were −0.000593 all4 and −0.000995 mature3; weekly deltas (Sep 9/16/23/30) were +0.000702/−0.002934/−0.001901/+0.001380. The unchanged standalone development gate failed. The 1,000-replicate paired stable-user bootstrap intervals for P021 delta were [−0.003648,+0.002590] all4 and [−0.004547,+0.002449] mature3 (seed 20261022). Seed-origin AP SD was 0.033718; seed-mean P021 combined-F1 SD was 0.015769. Across 24 fits, cumulative GPU wall time was 772.9 seconds, max allocated/reserved memory 0.84/3.24 GB; no OOM or retry. This is a completed verified negative.

Finalist manifest SHA256 is `0c405faf3f1eea445c77feb77f7e6ee7f8dc128a803b5e2a0831854587355e3d`; ensemble prediction SHA256 is `d5a249742299609d1540ff7df79567b02c508d8e5c0e6273ae7ba33de172cf96`; final verifier SHA256 is `91848f968bd7d2cba2d2d0a40c5b0f3431acd2bf0be9c13fa2f1b74d7334ead6`. Detailed per-origin/per-target, pooled/mature, bootstrap, seed and resource evidence is in `outputs/neural_refresh_20261022T0033Z/exp037/gru/finalist/verification/`.

Exact replay and verification commands (from repository root):

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.run --family EXP-037-G --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.verify --family EXP-037-G --run-dir outputs/neural_refresh_20261022T0033Z
```

## 40. EXP-037-C TCN+F0 — score-free preregistration

After the GRU verification passed, the TCN settings were frozen before the TCN search at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/preregistration.json` (SHA256 `a7583591d2c1b7e08a1693a81e39992c3f61952eb8ca23c5ea4f7ed38d88423f`). The registration binds refreshed source `2053302ba75751be975b9e779f567557b66f4071`, the same authenticated 56×7 sequence cache (manifest SHA256 `946b620f7bd60641feca16e381c35cb3a925116cabb4ddba4b14680f674b0b08`), model SHA256 `c38b78447b35c20402f70fad2506b167ae0f8bda6cb0cbedb73f8ffce9868270`, trainer SHA256 `e3a9d3ceeb8658cade8505ed37c4a709fa95cd7faa519a14ef6eb5211f7943d0`, and common core/normalization and report/refit protocol. Its four fixed configurations are width 32/64 × dropout 0.1/0.3, each with six causal residual blocks, two kernel-3 convolutions per block, dilations 1/2/4/8/16/32, left-only padding, final-time-step representation and learning rate 0.001. The actual causality check is run before training. No TCN candidate report score has been read. Next is its 8-fit search, frozen selection, 12 three-seed full-window refits and verification, followed by the preregistered temporal diagnostic controls.

## 41. EXP-037-C first attempt and compute-only runtime amendment — October 22, 2026

- The original score-free TCN registration remains immutable at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/preregistration.json` (SHA256 `a7583591d2c1b7e08a1693a81e39992c3f61952eb8ca23c5ea4f7ed38d88423f`). Its first search fit exceeded the registered 2,700-second cap at epoch 2. The checkpoint, epoch history, trainer console and process record are preserved under `outputs/neural_refresh_20261022T0033Z/exp037/tcn/`; the partial attempt is incomplete and is excluded from search ranking and selection. No TCN config was selected.
- Before retrying, a synthetic-only benchmark used the registered width-32 TCN, F0 width 59, 2,048-example batches, 56×7 sequences and FP32 BCE/AdamW backward steps on the RTX 3070 Ti Laptop GPU. With cuDNN benchmarking disabled, median step time was 10.9649 seconds for deterministic convolution and 0.02791 seconds for nondeterministic convolution (393.0× ratio). Peak allocated memory was 606,924,288 and 638,168,064 bytes, respectively. The benchmark read no campaign examples, labels, predictions or report scores; full measurements and runtime are saved in `outputs/neural_refresh_20261022T0033Z/exp037/tcn/runtime_benchmark_01.json` (SHA256 `26fa5a6121a81a7e61b7c89b372c2469531f577d3bba9539bfadff43737be8bb`).
- Registered compute-only amendment `EXP-037-C-resource-amendment-01` before any amended fit at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01/amendment.json` (SHA256 `fe4cdd7072fa7fc87d6684377859e850f4fc723291a1ddca45d27f220c842b62`). The amended score-free preregistration is in the same folder; it binds the original registration, benchmark, modified trainer and neural core. `cudnn.deterministic=False` and `cudnn.benchmark=False` are recorded per fit. Fixed seeds, architecture/grid, FP32 precision, preprocessing, batch/optimizer/loss, folds, stopping rules, gates, and the 2,700/5,400-second fit caps remain unchanged. Because cuDNN may select nondeterministic convolution kernels, bitwise repeatability is not claimed; checkpoint sample inference remains checked at tolerance 1e-5.
- All eight TCN search fits and all 24 finalist inner/refit jobs must restart from fresh initialization in the amendment directory. The original partial validation AP is explicitly excluded from every amended result. Exact commands: `C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe -u -m experiments.exp037_temporal_models.run_amendment_01 --phase run --run-dir outputs/neural_refresh_20261022T0033Z`; then `C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe -u -m experiments.exp037_temporal_models.verify --family EXP-037-C --run-dir outputs/neural_refresh_20261022T0033Z --family-root outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01`.
- At registration time the amended TCN search had not started. Temporal same-input diagnostics remain pending until both GRU and TCN finalists have complete independent verification.

## 42. EXP-037-C TCN+F0 — Complete, independently verified negative

The amended TCN run completed all eight registered search fits and all 12 report-origin × seed inner-selection/full-56-origin refit pairs (24 fits total). The frozen search rule selected `tcn_w64_d0.1` (187,462 parameters; two-fold mean six-target AP 0.2749964577). Its search artifact SHA256 is `432432fe35784b668ea4f1f308a487fde5a999fe6b87d1cda698a4f161c128c5`. The finalist manifest SHA256 is `fb81030cf4479e4ab6cc4b9ca11c1c2a7eb8bf35ce26f1e183dde06084474a22`; its 22,919-row prediction artifact SHA256 is `d454ca5f4b793e097ed4c235723475f7ff9c4278142807c580641fdd1ce1631f`.

Independent verification **passed**. It authenticated all eight search checkpoints and re-scored their validation predictions/tie rule, rebuilt source keys, labels, splits and fit-only preprocessing, sampled all 24 finalist checkpoints, reproduced saved per-seed predictions and the ensemble, recomputed report policies/metrics, and confirmed official scorer parity. The TCN causality probe passed (future perturbation prefix error 0.0; left padding only; output shape `[2,6]`). The verifier SHA256 is `035e5617b2f3a99a3469c29ab16069e27e0c9756f2d6930f0162ca11357dc056`.

Checkpoint replay initially exposed output drift when the verifier changed CUDA batch dimensions (CPU batches of five differed by 1.79e-5; same-device, five-row replay differed by 1.12e-5). No tolerance or prediction was changed. The verifier was corrected to reproduce samples within their original CUDA/cuDNN 2,048-row inference batches; the final replay then had maximum error 0.0 against the frozen 1e-5 tolerance. The failed verification attempts and successful same-batch replay log are preserved in the finalist verification directory. A separate GRU recheck with the strengthened search-checkpoint verifier also passed (verification SHA256 `36d56c2d095fbb345c90a74643e4f130e6f6f46864b52137db7aca10405b6658`).

| Scope | Model | Mean-origin six-target AP | P021 combined F1 | H2 mean-origin AP | H2 P021 combined F1 |
|---|---|---:|---:|---:|---:|
| all4 | TCN | 0.535675 | 0.545690 | 0.535030 | 0.547634 |
| mature3 | TCN | 0.542125 | 0.541906 | 0.537493 | 0.544169 |

The all4 AP delta versus H2 was +0.0006452269, with improvement on only two of four origins. P021 combined-F1 deltas were −0.0019442551 all4 and −0.0022628086 mature3. Weekly P021 F1 deltas (Sep 9/16/23/30) were −0.004301/−0.001064/−0.001585/−0.000624. Stable-user paired bootstrap (1,000 resamples, seed 20261022) intervals for P021 combined-F1 delta were [−0.005015,+0.001257] all4 (6,464 users) and [−0.005740,+0.001555] mature3 (6,431 users). The unchanged standalone development gate failed; this is a completed verified negative. Across 24 fits cumulative GPU wall time was 2,584.2 seconds (maximum individual fit 212.5 seconds); no fit hit a cap, OOM or retry.

Exact replay and verification commands (from repository root, accepted CUDA environment):

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.run_amendment_01 --phase run --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.verify --family EXP-037-C --run-dir outputs/neural_refresh_20261022T0033Z --family-root outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01
```

The original capped attempt remains preserved and its partial score was excluded. The compute-only amendment and synthetic benchmark are in §41. Complete TCN evidence is under `outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01/`. The only remaining scoped work is the preregistered same-input temporal diagnostic suite and final reporting.

## 43. EXP-037 same-input temporal diagnostics — score-free registration

After both temporal finalists passed independent verification, the diagnostics were score-free preregistered at `outputs/neural_refresh_20261022T0033Z/exp037/diagnostics/preregistration.json` (SHA256 `c6a0319e393da85c07724aadb4e5a00e66492a324a67b28657d1563324448a95`). No diagnostic fit or diagnostic candidate score has been read. The source and eight refreshed table hashes, GRU/TCN finalist selection and verifier digests, sequence cache, frozen EXP-036-R winner, execution settings and all three diagnostic recipes are bound into the registration. Runner SHA256 `bf85d48fc28b0bfcdf4f8951d40ce47747286ca27dd36c293c98331affd2944a`; verifier SHA256 `1ba19c1927e92a092a364456e21a612e30e7ada49758ba297b451bab93da7922`. Its dependency hashes additionally bind the shared trainer/core, refreshed data reader, sequence reader/cache builder, model implementation and temporal verification helpers.

The temporal architecture was selected by the registered highest all4 mean-origin six-target AP rule: amended TCN `tcn_w64_d0.1` (0.5356748502 AP; 187,462 parameters) exceeded GRU `gru_h64_l1_lr0.0003` (0.5326239942 AP; 63,686 parameters), so the tie rule did not apply. The flat-history control freezes the EXP-036-R width 256 and learning rate 0.0003 winner. The other controls use the selected TCN with (1) deterministic within-example permutation of all 56 bins while keeping the seven channels together, or (2) the F0 branch removed. Each control has four September report origins, three seeds, 12 inner-epoch-selection/full-window-refit pairs (24 fits); controls run sequentially through one trainer. No October outcomes are included.

An initial registration preflight stopped before writing a preregistration because the EXP-036-R selection file exposes `selected_config.config_id` rather than a root `selected_config_id`. This score-free reader adapter was corrected and recompiled; no report candidate scores were read and no fit began. Exact commands:

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -m experiments.exp037_temporal_models.diagnostics --phase preregister --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.diagnostics --phase run --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.verify_diagnostics --run-dir outputs/neural_refresh_20261022T0033Z
```

The pre-registration pass succeeded with `EXP-037-C` selected. The remaining work is the sequential real-data execution of all three controls, independent verification, and final report/audit.

## 44. EXP-037 temporal diagnostics — complete, independently verified

The registered suite completed sequentially with one trainer. Each of the three controls finished all 12 report-origin × seed pairs, each with its inner epoch-selection fit and a fresh full-56-origin refit (24 fits per control; 72 pairs and 144 fits total). The runner exited 0. Per-control manifests and prediction files are bound to the score-free registration SHA256 `c6a0319e393da85c07724aadb4e5a00e66492a324a67b28657d1563324448a95`.

The independent diagnostic verifier **passed** for all three controls. It reconstructed refreshed-source keys, labels, splits, H2 weights and fit-only preprocessing; authenticated every checkpoint; sampled 24 checkpoint predictions per control; checked all seed columns and the arithmetic ensemble mean; recomputed operating decisions and metrics; and confirmed official scorer parity. Maximum checkpoint replay error and maximum seed/ensemble column error were both 0.0 for every control against tolerance 1e-5. Root verification SHA256 is `6acdf6add23342271027e1b5d3968aadbbb64a6ee40fc042e5bba6bd2eb67643`.

| Control | All4 AP | All4 AP Δ vs H2 | All4 P021 combined F1 | All4 ΔF1 vs H2 | Mature3 AP | Mature3 P021 combined F1 | Mature3 ΔF1 vs H2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flattened F0+56×7 history → frozen residual MLP | 0.526354 | −0.008676 | 0.545262 | −0.002373 | 0.540456 | 0.543037 | −0.001132 |
| TCN+F0, all-bin joint seven-channel permutation | 0.531082 | −0.003948 | 0.545842 | −0.001792 | 0.533659 | 0.541182 | −0.002987 |
| TCN sequence only, F0 removed | 0.525250 | −0.009779 | 0.540036 | −0.007599 | 0.530822 | 0.536241 | −0.007928 |

These are preregistered diagnostic contrasts against the same fresh H2/P021 controls (all4 AP 0.535030 / combined F1 0.547634; mature3 AP 0.537493 / combined F1 0.544169). No diagnostic was selected or promoted as a nominee. Stable-user paired bootstrap intervals for P021 combined-F1 differences (1,000 resamples, seed 20261022; 6,464 all4 / 6,431 mature3 stable users) were:

| Control | All4 ΔF1 95% interval | Mature3 ΔF1 95% interval |
|---|---:|---:|
| Flattened history | [−0.005511,+0.001109] | [−0.005084,+0.002466] |
| Temporal order permuted | [−0.004941,+0.001230] | [−0.006485,+0.000458] |
| Sequence only | [−0.011335,−0.003929] | [−0.012248,−0.003761] |

Weekly ensemble P021 F1 deltas versus H2 are in `outputs/neural_refresh_20261022T0033Z/exp037/diagnostics/weekly_p021_deltas.csv`. In Sep 9/16/23/30 order, the flattened-history control scored −0.001552/−0.000204/−0.002264/−0.007410; temporal-order permutation −0.003296/−0.001577/−0.005226/+0.003452; sequence-only −0.009633/−0.003885/−0.011699/−0.005979. The lowest week was Sep 30 for flattened history and Sep 23 for each TCN diagnostic. Targetwise all4/mature3 AP, precision, recall and F1 deltas are saved in `per_target_scope_summary.csv`; the verified per-origin and per-target operating tables remain in each control's verification directory.

Across the three seeds, all4 P021 mean AP (sample SD) and combined F1 (sample SD) were 0.517229 (0.002748) / 0.543684 (0.000975) for flattened history; 0.521602 (0.001675) / 0.544741 (0.001355) for temporal permutation; and 0.513609 (0.003586) / 0.536595 (0.001498) for sequence-only. This seed dispersion is computed from each seed's separately rescored predictions; it is distinct from the arithmetic-ensemble result above.

There were 24 fit-resource records per control. Sum/max reported fit wall seconds, sequential runner elapsed seconds including preprocessing, and peak allocated/reserved CUDA memory were: flattened history 298.6/23.9 seconds, 458.5 seconds, 89.8/107.0 MB; temporal permutation 2,510.7/197.9 seconds, 2,714.0 seconds, 1.249/1.531 GB; sequence-only 3,618.8/267.0 seconds, 3,738.7 seconds, 1.248/1.531 GB. There were no OOM retries or cap failures. Full records are in each `fit_resources.csv` and the corresponding `reporting_supplement.json`.

Final manifest SHA256 values are `52f6427a750ef2a56036c618e65a0df4ac0d312842894db9fba632001a0994b1` (flattened history), `0b18ea11e5f81b8d8c7d2c8c5a6f6deec16668f78a73ac6614208afc6ac5b4b7` (permuted) and `280bab4855f5a6b9d37623305c6f352a9c98e710d5a1e5c04a2023ccc899b812` (sequence-only). Prediction SHA256 values are `2790ebd84a97cb53aa76a2b66615aea0c37736d7fa74fbf583c7e74f24e4134e`, `5f5275cbc34d588eb918e53de8461c56765668a5b07f3d7abcac4a1ba04ba5f1` and `1137a542eda65d928d158a08fa87ba1d27a1ee5ccce2bf8ff551ece621f50b5c`, respectively. Control verifier hashes are `4158c16f9264aa269acc2430987b1d4016c8e1f335fff1a6bfdbec051452c24c`, `41b6c2c72423ec5b0789630670d922cd9545a89a2b56cf62b42e3ff2019805b4` and `2da84f2d2a9bcbea8c1874c7546d18903f83aed561f68bde170671643220ab12`.

Exact execution, verification and reproducible reporting commands (repository root):

```powershell
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.diagnostics --phase run --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.verify_diagnostics --run-dir outputs/neural_refresh_20261022T0033Z
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.exp037_temporal_models.summarize_diagnostics --run-dir outputs/neural_refresh_20261022T0033Z
```

This completes the required temporal controls. No October outcomes were read; all reported model evidence is development evidence on the registered September origins.

## 45. Luna Max final report, artifact audit and scope closure — October 22, 2026

The complete result summary is saved at `reports/neural_refresh_20261022T0033Z/FINAL_REPORT.md` (SHA256 `c2f1bb8200f6429406abd625eab2349c5795023eb664c66f4bf8347ca5adf4d6`). It reports the refreshed-source Step 0 controls, all registered family searches and selected finalist results, the fixed MLP preprocessing contrast, the full 12-cell EXP-041 tabular grid and locked transfer result, the three temporal diagnostic controls, paired uncertainty, resource use, per-origin/per-target scorecard paths and exact replay commands. Row-level evidence remains in the campaign outputs referenced by §§32–44.

Final audit confirms all required Step 0, MLP primary/contrast, TabM, tabular blend, GRU, amended TCN and diagnostic verification artifacts exist. All 118 current `state.json` accepted-output paths exist; all 48 local Markdown links in the final report resolve. The final diagnostic root verifier passed at SHA256 `6acdf6add23342271027e1b5d3968aadbbb64a6ee40fc042e5bba6bd2eb67643`; each control verifier passed. The root check covered 72 checkpoint samples and every control's seed/ensemble columns, all with maximum replay errors 0.0. The summary script is `experiments/exp037_temporal_models/summarize_diagnostics.py`; its report supplement, per-target, seed, weekly and resource outputs are present.

The accepted-output audit reconciled one stale state path: `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/transfer_attempt_03.console.log` is absent. The native transfer failure (`-1066598273`), its diagnosis and the subsequent corrected transfer are documented in §38 and the campaign progress ledger. The actual corrected-transfer console, runtime probes, registered amendments, final predictions and independent verifier are retained. The stale path was removed from the accepted-output list; no synthetic or reconstructed console file was created. This history does not leave a missing required scored fit or verification artifact.

The diagnostic runner exited 0 after three controls had each completed 12 origin/seed pairs and 24 inner/refit fits, with one trainer at a time. Final process inventory: no active Python trainer; campaign `trainer_policy.owned_processes` is empty; no subagents. The state file is `completed_verified`, its scoped `remaining_work` is empty, and it contains all exact registration/build/search/refit/repair/verification/reproduction commands. The user-facing report and campaign ledger both retain exact rerun commands. No October-origin outcome was built or read.

All work requested by this bounded goal is therefore **complete as verified evidence**, with negative gates recorded as negatives and no promotion. The historical studies and reserved October outcomes remain unchanged. The broader six-family EXP-041 study remains Ongoing outside this goal. The final report and audited machine state are the completion records.

## 46. Overnight five execution — October 23, 2026

The execution brief was implemented as a source-frozen, resumable batch. Corrected run `outputs/overnight_five_20261023T0237TW/` passed preflight against source SHA `2053302ba75751be975b9e779f567557b66f4071`, the refreshed table/cache inventory, and the saved H2/P021 control (pooled official F1 `0.5476344735753119`; prediction SHA256 `eeef8440dab47a5b986da765bc454099142acf6ac44b2d5a11547716d7a1ff9b`). No October labels were built or read. Future-row injection checks passed for shared F0, temporal features, relational features and the graph builder (four targeted tests). TabM's official PLE numerical smoke and the RelGNN equation 4–7/hand graph checks passed. The installed CatBoost 1.2.10 package was hashed into preregistration and passed a two-iteration synthetic CPU smoke with the registered options.

An initial runner attempt (`overnight_five_20261023T0217TW`) stopped at the HGB threadpool controller before any HGB estimator fit. The native Windows error was `-1066598273`; that run's preregistration and 98 partial graph-cache files are preserved. After the runtime-only thread-control repair, the obsolete process was interrupted to preserve the one-GPU-trainer rule. No score from that failed attempt is eligible. The fresh, corrected run is tracked separately.

EXP-023-S completed the registered two-fold August selection and four September full-window fits. The locked recipe was `more_rounds` (320 iterations, 15 leaves), mean tuning six-target AP `0.260036` versus `0.255767` for `more_leaves`. H2 runtime matched, so the runtime reference was not refit. Pooled P021 combined F1 was **0.543451** against H2's **0.547634** (Δ **−0.004183**); weekly deltas on Sep 9/16/23/30 were **−0.002944/−0.003986/−0.007298/−0.003496**. Pooled mean six-target AP was **0.480058**. Official P016 and P021 parity passed on every report origin. Six target F1 deltas versus H2/P021 were open PR `−0.001604`, open issue `−0.010852`, issue comment `−0.002462`, PR review `−0.005685`, PR review comment `−0.006521`, and push `−0.001995`. The 24 selected report target fits took 375.29 wall seconds and all reached 320 iterations. The complete scorecard and confusion/resource records are in `outputs/overnight_five_20261023T0237TW/exp023_s/metrics.json` and adjacent outputs.

The EXP-038-S paper-operator smoke and real as-of graph forward/backward verification passed. Initial graph setup took 9.20 seconds; the audited graph has a 90-day lookback and its maximum edge timestamp precedes the `S−4h` cutoff. The first width-128 August fit reached the 2,700-second cap after 2,703.39 seconds and saved three complete epochs (validation six-target AP 0.2852537, 0.2873451, 0.2848851; best epoch 2). It remains incomplete and cannot be used for width selection or interpreted as a predictive result. This is not an adequately trained negative. The authenticated checkpoint, epoch history and cap receipt are preserved. EXP-036-E is now running; EXP-011-R and EXP-041-S follow in the registered order. Interim source/config hashes and current process state are in the corrected run's `preregistration.json`, `preflight.json` and `progress.json`.

## 47. Overnight five execution — final verified results, October 23, 2026

**Correction:** §48 below supersedes this entry's AP interpretation, initialization reproducibility and audit-coverage claims. The observed F1 values remain unchanged.

The authoritative fresh batch is `outputs/overnight_five_20261023T0340TW/`. It supersedes the earlier in-progress snapshot in §46 for final status; both earlier attempts and their outputs remain preserved. Preregistration SHA256 is `9209d0c1e0f5512246f3d7ceb0336e5b2fb0879329bd90359adc82e873807f37`, brief SHA256 is `da4621c4a0685b3bdcaaac7d44da0597a5ad0034d36773e7cf640235385036c4`, and config SHA256 is `16016c0eb9f99976b45552d21c739b298e7c2a2efba041eb151950bdfd0062da`. The refreshed instructor source is `2053302ba75751be975b9e779f567557b66f4071`; TabM source hash is `fe56073ebc9bec642583583182c3ac067e931d6d1ebbdeee26640debdf567520`.

Preflight authenticated the 59-column F0 schema, 100 feature-cache origins, 403 label-cache keys and saved H2/P021 control. H2 reproduced at pooled official combined F1 `0.5476344735753119`; its prediction SHA256 is `eeef8440dab47a5b986da765bc454099142acf6ac44b2d5a11547716d7a1ff9b`. Every scored output uses the same 22,919 user-origin keys. No October-origin labels were built or read. The corrected batch started at 03:41:02 +08 and exited at 04:59:53.891316 +08, after 1:18:51.5; there were no runner failures.

### 47.1 EXP-023-S — HGB capacity, complete negative

The registered two-fold August tuning selected `more_rounds` (320 iterations, 15 leaves), with mean six-target AP `0.26003567190`; `more_leaves` (80 iterations, 63 leaves) scored `0.25576662730`. Selection was locked before September report scoring. Runtime identity matched the saved H2 fit manifest, so the H2 reference was not refit. All 24 report target fits reached the registered 320 iterations. Report-fit wall time was 317.850 seconds; tuning plus report-fit time was 511.292 seconds.

Pooled P021 combined F1 was `0.54345110785` versus H2's `0.54763447358` (delta `−0.00418336573`); mean six-target AP was `0.48005828210`. Weekly deltas (Sep 9/16/23/30) were `−0.002944/−0.003986/−0.007298/−0.003496`. Pooled target F1 deltas versus H2/P021 were open PR `−0.001604`, open issue `−0.010852`, issue comment `−0.002462`, PR review `−0.005685`, PR review comment `−0.006521`, and push `−0.001995`. The complete scorecard and official scorer parity passed. This is a complete negative development screen; no promotion.

### 47.2 EXP-036-E — TabM numerical embeddings, complete matched arms

Both arms used width 128, learning rate `0.001`, three blocks, 16 members and dropout `0.1`. The embedding-on arm used PLE version B, dimension 16, activation disabled, and up to 32 quantile bins per numerical column, fit on permitted fitting rows; all 59 F0 inputs were retained. Selected epochs by report origin were `[9,17,8,14]` off and `[3,7,20,14]` on. There were eight inner/refit fits per arm. The repaired report inference conversion (NumPy-to-torch) passed synthetic smoke checks before the fresh run; the source snapshot remained fixed during this batch.

Embedding-off control pooled P021 F1 was `0.54805800015` (delta `+0.00042352657`) and mean AP `0.46176887988`. Embedding-on pooled P021 F1 was `0.54299029041` (delta `−0.00464418316`) and mean AP `0.49616046236`. Thus embedding-on raised mean AP by about `0.03439` versus the matched control but reduced combined F1 by about `0.00507`. The off arm remains a control and is excluded from the blend pool by design. The embedding-on arm is a complete negative standalone screen; neither result changes the registered design.

### 47.3 EXP-038-S — RelGNN-inspired computation, incomplete at fit cap

The implementation passed the paper equations 4–7 review, hand-calculated tiny graph checks (including empty neighborhoods and missing FK), finite backward gradients, and real graph forward/backward verification. Real graph setup took `7.4641` seconds. The graph used a 90-day lookback with cutoff `2026-06-17T12:00:00Z` and latest edge timestamp `2026-06-17T11:59:53Z`; node counts were user 15,075, repo 59,991, PR 106,393 and issue 54,386. No graph reduction or OOM occurred.

The first width-128 August inner fit reached its registered 2,700-second cap after `2704.9702` seconds. Three epochs were saved with validation six-target AP `0.2868351`, `0.2882364`, and `0.2882110`; epoch 2 is the best partial epoch. This state is incomplete and was not used for width selection, report prediction, or a loss classification. Self-only and pooled-neighbor report contrasts therefore remain unrun. The checkpoint, history, fit budget and incomplete receipt are preserved under `outputs/overnight_five_20261023T0340TW/exp038_s/`.

### 47.4 EXP-011-R — CatBoost, complete negative

The single raw-F0 CPU recipe used the registered 300 iterations, depth 6, learning rate `0.05`, L2 `3`, Logloss, bootstrap `No`, random strength `0`, `rsm=1`, four threads and seed 42, with no class balancing or early stopping. All 24 report target fits reached 300 iterations. Report-fit time was `447.730` seconds and peak recorded RSS was about 595 MB. Pooled P021 combined F1 was `0.54421580066` (delta `−0.00341867292`) and mean six-target AP `0.48663087487`; weekly deltas were `−0.005909/−0.000582/−0.005166/−0.002016`. This is a complete negative screen; stop this one registered recipe.

### 47.5 EXP-041-S — fixed blends, three verified

The registered rank scaling, fixed candidate weight `0.25`, H2/EW7 terms, positive-count top-k and stable UserID ties were unchanged. The eligible pool contained only the complete selected HGB, embedding-on TabM and CatBoost outputs. TabM embedding-off remained a control, and the RelGNN blend was not run because that member was incomplete. The three executed blends passed alpha-zero exact H2/P021 decision parity and official evaluator parity. Report-only blend exports contain 22,919 common-key rows per candidate; their formula/hash and parity receipts are in `outputs/overnight_five_20261023T0340TW/exp041_s/prediction_export_manifest.json`.

The HGB blend scored `0.54619578786` (delta `−0.00143868571`), TabM-on blend `0.54800248633` (delta `+0.00036801276`), and CatBoost blend `0.54725611625` (delta `−0.00037835733`). These are development comparisons only. Retain the small TabM-on blend gain for later review; no policy is nominated or promoted.

### 47.6 Reconciliation and report artifacts

[`reports/overnight_five_20261023T0340TW/summary.md`](../reports/overnight_five_20261023T0340TW/summary.md) contains the comparison table, target-level table link, card decisions and exact commands. The generated tables contain 9 candidate/reference rows, 96 pooled P016/P021 target rows and 64 weekly rows. The independent report audit status is `PASS_WITH_RELGNN_INCOMPLETE`: it checked all 24 frozen source files without mismatch, seven model/blend Parquets with 22,919 unique keys each, finite outputs, independent combined-F1 reconstruction from confusion counts at absolute tolerance `1e-12`, and eight official-scorer checks per candidate. Saved verification is `reports/overnight_five_20261023T0340TW/verification.json`; the full target and weekly rows are in `target_metrics.csv` and `weekly_metrics.csv`.

The frozen batch preflight and independent verifier both passed. A final sandboxed `--preflight-only` re-entry first received `WinError 5` on the CatBoost metadata path; the same no-fit preflight then passed with approved elevated access. It launched no fit. The diagnostic is saved at `reports/overnight_five_20261023T0340TW/preflight_reentry_diagnostic.json`.

The exact recorded re-entry command for the unfinished experiment is:

```powershell
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& 'C:\Users\Davon\miniconda3\envs\asg2-rdl\python.exe' -u -m experiments.overnight_five.run --run-dir outputs/overnight_five_20261023T0340TW --experiment exp038-s --new-batch
```

This re-enters the saved experiment and confirms its cap state; it does not make the partial fit scoreable or continue the same fit. Further scientific work requires a named pre-score resource amendment and a fresh run directory. A preflight-only command is also recorded in the summary and was kept separate from any training invocation. No October outcomes were read and no design or result was silently changed.

## 48. Overnight implementation repairs and RelGNN throughput investigation — October 23, 2026

The three medium-severity review findings are repaired in the separate `experiments/overnight_five_v2/` package. Historical runner bytes, the 0340TW run, checkpoints and original reports are preserved. Future neural construction seeds Python/NumPy/PyTorch before creating either TabM arm or a RelGNN fit/refit, and records that initialization policy. This fixes future execution; the historical neural fits retain uncontrolled-initialization limitations and are not retroactively relabelled as reproducible seed-42 treatments. The new launcher rejects a historical run directory before writing lock/progress state.

AP reporting now distinguishes equal-origin mean six-target AP from pooled-row AP, and labels raw probabilities versus blended policy ranks. Corrected **mean-origin raw AP** is H2 `0.53502962327`, HGB `0.51414879206`, TabM-off `0.53801599314`, TabM-on `0.51985743790`, CatBoost `0.52023558814`. Thus the embedding-on minus off AP effect is **−0.01815855525**, not the pooled-AP increase reported in §47.2. Its F1 effect versus off remains `−0.00506770974`. The opposite AP interpretation is withdrawn; differences between these arms still have the legacy initialization limitation. All seven original standalone/blend F1 results reproduce, including the TabM-on blend's `+0.00036801276` development delta. Blend AP describes the final rank including EW7 and must not be interpreted as a raw-model AP improvement.

The replacement verifier fails on missing expected sources, authenticates 89 checkpoint artifacts, reconstructs policies/decisions/confusion counts directly from the saved probabilities, checks blend export scores and decisions, and executes the official scorer 70 times across the seven outputs (four weeks plus pooled, P016/P021). It also replays 24-row checkpoint samples: both TabM arms on all four report origins, and all six targets of each tree family on the first report origin (20 replay records). Tree replay uses its original CPU runtime. All 24 frozen sources match. Status: `PASS_WITH_LEGACY_INITIALIZATION_LIMITATION_AND_RELGNN_INCOMPLETE`. These are bounded checkpoint checks, not every-row inference replay or a fresh raw-table label audit. Seven focused regression tests passed.

The capped width-128 RelGNN fit used 361,058 rows from 42 fitting origins and 2,048-row globally shuffled objective batches. Reconstructing its seed-42 first-epoch shuffle from cache row metadata gives 177 batches; every batch contains all 42 origins, producing **7,434 separate graph encoder calls per epoch**. The model loops over origins, then two layers and 20 relation routes. This creates many small tensor transfers and operator calls. It is an implementation throughput limitation worth addressing before increasing capacity or runtime.

An August-only, no-optimizer profile of the original checkpoint on the RTX 3070 Ti Laptop GPU held the seed-row count at 256. Forward/backward averaged **0.218 s with one origin**, **0.464 s with four**, and **0.949 s with eight**, a 4.35× slowdown from one to eight. Sampled graph sizes were similar; synchronized sampling stayed near 0.09–0.10 s while encoder/backward time increased. The eight-origin profiled pass invoked 320 route attention forwards and 1,640 `torch.as_tensor` calls. This supports origin fragmentation as a major bottleneck; the small diagnostic is not a measured full-epoch speedup forecast. Three completed epochs at the 2,700-second cap remain incomplete, with no predictive score. The profile changed neither checkpoint nor model weights and read no October outcomes.

Detailed evidence and commands: [repair/profile report](../reports/overnight_five_repair_20261023/REPAIR_AND_PROFILE_REPORT.md), [corrected metrics](../reports/overnight_five_repair_20261023/corrected_comparison.csv), [verification](../reports/overnight_five_repair_20261023/verification_v2.json), [profile](../reports/overnight_five_repair_20261023/relational_profile/profile.json), [batch census](../reports/overnight_five_repair_20261023/relational_profile/batch_census.json). No historical fit was resumed or rescored into eligibility; no new model was nominated.

## 49. Final submission pipeline, sample-week calibration and Oct 7 confirmation — October 23, 2026

`experiments/final_submission/run.py` (score and submit modes) reuses the existing population, F0, label, persistence, EW7 and P021 code. Commits: `d95e120` (pipeline), `cabc397` (six untracked imported modules, unchanged) and freeze `3f705b2` at 14:46:28 Taiwan time ([FREEZE.json](../reports/final_pipeline_20261023/FREEZE.json)). No automatic decision rule is frozen: the user will decide manually from the reported results. Full record: [REPORT.md](../reports/final_pipeline_20261023/REPORT.md).

- **Source preservation:** `2053302b` copied to `assignment2_backup/2053302ba75751be975b9e779f567557b66f4071_20261023/assignment2`; all 8 table SHA256s match the campaign source manifest.
- **R1 parity (Sep 30, preserved `2053302b`):** 5,178/5,178 users. `dec_H2_P021`, `dec_H2_P016` and `dec_P_persist7` have 0 mismatches on all 6 targets; max |Δp_H2| = 0.0. 56 training origins, 425,449 rows, 57 s wall time. Sep 30 combined F1 with labels at S+12d21h: H2/P021 0.554036, P_persist7 0.516449 (same as the saved controls).
- **R2:** passed on every run (feature rows max seen < C; training-label rows max seen and event < A_S).
- **Format check:** submit mode on the 4,785 `sample_output.csv` users passes `evaluate.py --input` for `h2_p021` (51 s) and `persist7` (2 s).
- **Sample week (S = 2026-09-27T13:00Z, `--truth sample_truth.csv`, calibration only):** H2/P021 0.556296, H2/P016 0.540949, P_persist7 0.504091; instructor `sample_output.csv` 0.5615.
- **Release `c221c24d09d70c32533d3f4ad804b4cfd2cbdf1d`** (latest observed 2026-10-22T11:59:23Z; one clone retry after a transport error): cohort 9,000 with identical stable IDs, no schema change, 0 FK orphans across 20 relations. Sep 30 rerun: 0 decision, persistence-flag or label changes; max |Δp_H2| = 0.0. Only Oct 7 is mature.
- **Oct 7 (first confirmation week, after the freeze):** 4,932 proxy users, labels as-of 2026-10-20T13:00Z. H2/P021 0.519876 (macro 0.455607, micro 0.584145); P_persist7 0.479400 (macro 0.413105, micro 0.545694); delta +0.040476. H2/P016 (context) 0.517524. H2/P021 beats P_persist7 on all 6 targets.
- **Reviewer checks:** combined F1 independently recomputed for every week and policy, matching the official `scores()` with difference 0.0; 40 Oct 7 users spot-checked against raw `events()` with 0 mismatches; no October-origin file predates the freeze commit.

Pending: Oct 14 (needs coverage 2026-10-27T13:00Z) and Oct 21 (needs 2026-11-03T13:00Z); adopting the lean endgame plan as plan §51 is the user's decision.

## 50. Revised dataset and policy comparison

**Completed and verified November 7–8, 2026 (Taiwan time).** This section records the revised-course backtests. The instructor's submission target is the future November 22–29 Taiwan-time window. The scores below use reconstructed historical labels.

### 50.1 Data and runner changes

- The revision moved from course-provided Parquet snapshots to hourly GH Archive JSON files and the official `scripts/preprocess.py`. The preprocessor produces eight anonymized Parquet tables. The historical runs used course commit `af9f335301d9badb482dfafa10c4534eabac845a`. The revised submission roster is `submission/test_input.csv`, with 4,465 unique coded IDs in that checkout. The historical comparison populations use the eligibility proxy described below. [Pinned course instructions](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/af9f335301d9badb482dfafa10c4534eabac845a/README.md) · [GH Archive](https://www.gharchive.org/)
- The [final runner](../experiments/final_submission/run.py) now accepts `--official-repo` independently of `--data-dir`. The [loader](../src/data_loader.py) resolves `scripts/evaluate.py` for the revised checkout and supports `evaluation/evaluate.py` for the original layout. The revised preprocessor can yield a partial cohort for short inputs, so submit mode checks every requested roster ID against the loaded cohort. Revised run manifests record the course commit separately from the eight generated table hashes.
- The October 15 smoke build reproduced all eight Parquet tables exactly when run on the compact archive staged using the official extractor. The revised format checker accepted the 4,465-row official sample and a repeatable 321-row persistence smoke prediction for the roster IDs present that day. The full roster check rejected that one-day dataset because 4,144 roster IDs were absent. The legacy format check also passed on 3,960 IDs. All 64 focused tests passed. The broader suite reported 506 passes, one skip, and three temp-directory or import-order failures. Those three tests passed when rerun separately.
- The full-history benchmark used **4,487 hourly files** in `[2026-04-13 12:00, 2026-10-17 11:00)` UTC. The extension added **520 hours** through `2026-11-08 03:00 UTC` exclusive, for **5,007 files**. Staging retained the records selected by the official extractor. Per-hour receipts bind the source and staged SHA256s, the course script and cohort hashes, and counts by raw event type. Interrupted transfers resumed from partial files, and all expected files were verified before the official table build. [Initial coverage audit](../assignment2_new/revision_coverage/hourly_coverage.tsv) · [Validation-week audit](../assignment2_new/revision_coverage/validation_week_hourly_coverage.tsv)

### 50.2 Four versions and the shared protocol

H2 fits one histogram gradient-boosted classifier per activity on the same 59 F0 features. F0 includes personal activity counts, recency, and repository breadth over historical windows. Each full fit uses 56 daily origins. H2 training rows receive recency weights with a two-week half-life, normalized to a mean of one. The registered tree settings are 80 iterations, learning rate 0.07, at most 15 leaves, minimum leaf size 75, L2 penalty 3, seed 42, and early stopping disabled. [Shared configuration and weights](../experiments/neural_research_common/data.py)

| Version | Per-activity ranking before the top-`k` decision |
|---|---|
| H2/P016 | Rank by the raw H2 probability. |
| H2/P021 | 50% H2 percentile rank + 50% EW7 percentile rank. EW7 sums visible events over the preceding 90 days, with each event weighted by `exp(-age_in_days / 7)`. |
| H2-WPCA/P021 | 25% H2 rank + 25% WPCA rank + 50% EW7 rank. The WPCA branch applies a signed log transform to continuous features, scales the 59 inputs and fits PCA retaining 99% of training variance. It then fits one logistic regression per activity with `C=1` and the H2 row weights. Scaling and PCA use uniform weights on training rows. |
| H2-DFS/P021 | 25% H2 rank + 25% DFS rank + 50% EW7 rank. The DFS branch uses the same preprocessing and logistic regression with 84 inputs. Its 25 additional features summarize other users' pushes, issue comments, PR review comments and distinct actors across recently interacted-with repositories over 7 and 30 days. They include mean, max and standard deviation across those repositories, plus the 30-day repository count. |

For each window and activity, all four policies set `k` to the number of users with that activity in the seven days preceding the feature cutoff. They choose the highest-ranked `k` users, with ascending `UserID` breaking ties. Persistence itself predicts those exact prior-week active users. P016 and P021 use the same H2 probabilities and positive counts, which isolates the effect of adding EW7 to the ranking. P016 decisions are exported in the H2 run's `predictions.parquet` as `dec_H2_P016_*`. [Ranking and selection](../src/phase_b_rules.py) · [PCA transform](../experiments/pca_fisher_20261026/models.py) · [Weighted logistic regression](../experiments/pca_followup_20261026/models.py) · [DFS feature definitions](../experiments/dfs_pca_20261026/features.py)

For an origin `S`, features use records visible strictly before `C = S − 4h`. Training labels use events in their own seven-day target windows with an observation cutoff of `A = S − 3h`. Historical evaluation labels normally use `S + 12d21h` as the observation cutoff. The proxy population selects cohort users with at least five visible records in a 90-day window ending 14 days before `C`. Each policy within a window uses the same user IDs and labels. Combined F1 is the mean of macro-F1 and micro-F1. Binary accuracy is the fraction of scored user–activity decisions that match the label. [Runner protocol](../experiments/final_submission/run.py) · [Label construction](../src/labels.py)

### 50.3 Short-history checks and first full-history results

The short-history checks used six training origins and a population of cohort users active in the preceding seven days. On September 27–October 4, WPCA led at **0.592839** combined F1 across six activities on 2,786 users. On October 4–11, WPCA led at **0.438338** across four activities on 2,940 users. The full-history runs use the standard 56-origin fits and eligibility proxy. They provide the matched comparisons below. [September short-history scores](../assignment2_new/revision_coverage/sep27_pilot_scores/scores.json) · [October short-history scores](../assignment2_new/revision_coverage/pilot_scores/scores.json)

September 27–October 4 (both boundaries at 13:00 UTC) had all six activity types in all 168 target hours. The scoring population was **5,313 users**, and each learned fit used **434,402 training rows**. The September 27 boundary was previously used for instructor-sample label reconciliation.

| Policy | Macro-F1 | Micro-F1 | Combined F1 | Binary accuracy |
|---|---:|---:|---:|---:|
| Persistence | 0.437130 | 0.566105 | 0.501618 | 0.886034 |
| **H2/P021** | 0.502081 | 0.609340 | **0.555710** | 0.897390 |
| H2/P016 | 0.481231 | 0.598830 | 0.540030 | 0.894630 |
| H2-WPCA/P021 | 0.501220 | 0.608145 | 0.554682 | 0.897076 |
| H2-DFS/P021 | 0.501512 | 0.608384 | 0.554948 | 0.897139 |

October 4–11 (13:00 UTC) had **40 hours with zero pushes** and **41 hours with zero PR review comments**, so those two activities were excluded from scoring. The four-activity comparison covered **5,065 users**, and each learned fit used **412,846 training rows**. In both benchmarks, the historical training-label windows included 131 hours without recorded pushes and 29 without recorded PR review comments. [Verified full-history scores](../assignment2_new/revision_coverage/full_scores/scores.json)

| Policy | Macro-F1 | Micro-F1 | Combined F1 | Binary accuracy |
|---|---:|---:|---:|---:|
| Persistence | 0.412535 | 0.409889 | 0.411212 | 0.910464 |
| H2/P021 | 0.455467 | 0.453481 | 0.454474 | 0.917078 |
| H2/P016 | 0.453383 | 0.450878 | 0.452131 | 0.916683 |
| **H2-WPCA/P021** | 0.459329 | 0.457385 | **0.458357** | 0.917670 |
| H2-DFS/P021 | 0.456992 | 0.454782 | 0.455887 | 0.917275 |

### 50.4 October–November coverage and full-history scores

The extension refit the same recipes at four origins. A target is excluded from that week's evaluation when its underlying raw event type has zero records in any of the 168 hours. This is an availability screen. Event types can be present while counts are still depressed. Push coverage had already deteriorated in the initial audit, with pushes present in only 8 of 120 sampled hours across five dates from October 15 to November 6.

| Window and clock boundary | Users | Fit rows | Hours with zero records by excluded activity | Labels as of (UTC) |
|---|---:|---:|---|---|
| Oct 11–18, 13:00 UTC | 4,790 | 389,253 | Push 86 | Oct 24 10:00 |
| Oct 18–25, 13:00 UTC | 4,548 | 364,181 | Issue comment 8, push 31 | Oct 31 10:00 |
| Oct 25–Nov 1, 13:00 UTC | 4,373 | 337,741 | Issue comment 1, PR review comment 31, push 146 | Nov 7 10:00 |
| Nov 1–8, midnight Taiwan time | 4,202 | 314,716 | Issue comment 1, PR review comment 13, push 138 | Nov 8 02:00, provisional |

The November target window was complete at scoring time, with labels observed through November 8 at 10:00 Taiwan time. The standard delay reaches November 13 at 21:00 Taiwan time. The November window overlaps the preceding October window by 21 hours. It is reported separately from the mean of the three non-overlapping mature weeks.

The first table uses the activities available in each window. Each column is a matched comparison between policies on that week's users and labels. [Recent reference scores](../assignment2_new/revision_coverage/oct_nov_scores/scores.json) · [Full version comparison](../assignment2_new/revision_coverage/oct_nov_scores/version_comparison.json)

| Policy | Oct 11–18 (5) | Oct 18–25 (4) | Oct 25–Nov 1 (3) | Nov 1–8 (3, provisional) |
|---|---:|---:|---:|---:|
| Persistence | 0.355286 | 0.347159 | 0.356450 | 0.366178 |
| H2/P021 | 0.405129 | 0.412907 | 0.405308 | 0.424849 |
| H2/P016 | 0.406635 | **0.415964** | **0.426351** | **0.449744** |
| H2-WPCA/P021 | **0.412137** | 0.406069 | 0.411004 | 0.432089 |
| H2-DFS/P021 | 0.409054 | 0.409426 | 0.408399 | 0.432089 |

The next table scores the same three activities in every window (`open_pr`, `open_issue`, and `pr_review`). The ranking is the equal-weight mean combined F1 across the three mature weeks.

| Policy | Oct 11–18 | Oct 18–25 | Oct 25–Nov 1 | Mature mean | Nov 1–8 provisional |
|---|---:|---:|---:|---:|---:|
| **H2/P016** | **0.424468** | **0.420988** | **0.426351** | **0.423936** | **0.449744** |
| H2-WPCA/P021 | 0.424309 | 0.413233 | 0.411004 | 0.416182 | 0.432089 |
| H2-DFS/P021 | 0.421021 | 0.415916 | 0.408399 | 0.415112 | 0.432089 |
| H2/P021 | 0.422265 | 0.417153 | 0.405308 | 0.414909 | 0.424849 |
| Persistence | 0.367584 | 0.351022 | 0.356450 | 0.358352 | 0.366178 |

**Submission candidate.** The recent matched results favor **H2/P016** if the submission week has similar activity coverage. Its mature mean was 0.78 percentage points above WPCA and 0.90 above P021. The provisional November score was 0.449744 combined F1 with 0.9329 binary accuracy. The earlier six-activity week favored P021 by 1.57 percentage points over P016. There is no recent six-activity score under this archive coverage, so the two results support different evaluation conditions.

### 50.5 Reproduction and verification

Run these commands from the project root with the pinned course checkout and the verified staged archives. The helpers, datasets, and generated predictions are local under the ignored `assignment2_new/` directory. The run manifests and the results linked above retain the exact inputs and metrics.

```powershell
$b = "assignment2_new/revision_coverage"
python "$b/run_revised_benchmark.py" --archive "$b/training_archive" --output "$b/full_scores_replay"
python "$b/score_oct_nov.py" --base "$b/training_archive" --delta "$b/oct_nov_delta" --output "$b/oct_nov_scores_replay"
```

The recent replay creates the persistence and H2/P021 predictions, historical roster and truth for each window. For each of the four origins above, run the [final runner](../experiments/final_submission/run.py) in submit mode with that window's `population.csv` and the same generated `db/`. Use `--policy h2wpca_p021` or `--policy h2dfs_p021` for the challengers. Export `dec_H2_P016_*` from the corresponding H2/P021 run's `predictions.parquet` under the standard six target headers for P016. Score each CSV against the saved `truth.csv` with the revised evaluator. For the matched comparison, use `--skip issue_comment pr_review_comment push`.

The full-history audit checked all 4,487 receipts and the eight generated table hashes, rebuilt both evaluation label files, and independently recomputed all ten policy/window scores from saved predictions. The recent audit repeated those checks for the extended 5,007-hour archive, reconstructed all four label files and independently replayed the **20 policy/window comparisons** for both activity selections. It verified common user keys, binary output, exact P016 decision exports, and agreement with the official evaluator. The learned-run manifests report the same training row keys within each window and passed the feature and training-label cutoff checks.
