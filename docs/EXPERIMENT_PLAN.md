# Assignment 2 — Experiment Plan & Status Register

**Course:** NYCU — Big Data Analytics Techniques and Applications, Fall 2026  
**Project:** Assignment 2 — Predicting GitHub User Activity  
**Last status review:** 2026-10-14 (Asia/Taipei); updated after EXP-006 user-side full four-model run  
**Related evidence:** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (results and detailed experiment history)  
**Project data source:** Instructor-supplied `assignment2/data/db/*.parquet` only  
**Reference instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`  
**Canonical scope:** `EXP-001` through `EXP-006`

**Repository paths updated October 14:** experiment workflows and saved runs now live in descriptive experiment folders. Shared modules remain in `src/`; configs are named by purpose. See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md). Historical evidence and statuses are unchanged.

> **Purpose:** This file is a living **plan, progress tracker and decision register**, not a substitute for measured evidence. `EXPERIMENT_PLAN.md` says **what has been done, what comes next, and why something is deferred**. `EXPERIMENT_LOG.md` stores the **actual configuration, observations, numerical results, limitations and reproducibility evidence**. Do not copy unverified scores into this plan.

## 1. Status rules and completion gate

| Status | Meaning | Required documentation |
|---|---|---|
| **Pending** | Defined, but implementation or execution has not begun | Objective, prerequisites, acceptance criteria and proposed next action |
| **Ongoing** | Some implementation or execution is underway, but acceptance criteria are incomplete | Progress achieved, remaining work and blockers; no unverified success claims |
| **Deferred** | Intentionally postponed, rather than merely not started | **Explicit reason**, decision date or context, revisit condition and impact |
| **Completed** | All **milestone-specific** acceptance criteria are satisfied and checked against observed evidence | **Mandatory completed result entry in `EXPERIMENT_LOG.md`**, with actual outputs, data/code provenance, metrics where relevant, limitations and a traceable section link |

**Non-negotiable rule:** An experiment **cannot** be marked **Completed** merely because code exists, a ZIP was delivered, synthetic unit tests passed, or a plan was written. The evidence required for that experiment must have **actually been executed** and **recorded in `EXPERIMENT_LOG.md`**. To change a status to Completed, first update the log, then update this plan.

**Experiment ID mapping:** Earlier work used notebook IDs `EDA-01`, `EDA-02`, `EDA-03`; for this six-stage plan they are canonical aliases **EXP-001**, **EXP-002**, **EXP-003**. Preserve those EDA IDs inside notebook names and the existing log—do not rename historical files or rewrite old measurements.

**Status changes:** Use one main status per experiment. A Completed experiment can have further **follow-up tasks** without becoming Ongoing again; new scientific questions belong in an addendum or subsequent experiment. If a completed finding is later invalidated, append a dated correction to the log and explicitly revise this plan with the reason.

## 2. Current overview

| ID | Experiment / milestone | Status | Main observed result or blocker | Mandatory log evidence |
|---|---|---|---|---|
| **EXP-001** | Initial relational dataset and schema audit (`EDA-01`) | **Completed** | 2,552,788 rows, 9,000 cohort users; 0 PK duplicates and 0 orphan references in 20 FK relations | `EXPERIMENT_LOG.md` **§4**, with source/provenance §3 |
| **EXP-002** | Deeper temporal, behavioral and relational EDA (`EDA-02`) | **Completed** | 45 complete weeks; strong regime changes; activity-persistence lifts and repository-breadth association quantified | **§5**, synthesis §6 |
| **EXP-003** | Historical label/split validation and rule-based baselines (`EDA-03`) | **Completed** | 31 user-week cutoff periods; ten integrity checks; prior-week heuristic **0.3504** mean weekly validation combined F1 (eligible proxy) | **§10**, in particular §§10.5 and 10.9 |
| **EXP-004** | Official example label semantics, window and as-of reconciliation | **Completed** | **28,710/28,710** example labels reconstructed exactly under **calibrated** rules; not predictive F1 | **§11**, especially §§11.4–11.11 |
| **EXP-005** | Reusable temporal dataset generator: tests and real-data build | **Completed** | **14 passed, 1 skipped** locally; real sample regression exact; **189k train + 81k validation + 9k sealed holdout features** | **§14**, appended after confirmed local run |
| **EXP-006** | Learned and heuristic baseline model comparison | **Ongoing — results obtained; detailed audit pending** | **Four real model runs scored:** persistence **0.454677** > HGB **0.398767** > logistic **0.381531** > zero **0** on **16,982** primary proxy-eligible user-weeks; per-target/threshold checks not yet supplied | **§15.1–§15.7** actual execution, leaderboard, limitations and completion gate |

**Snapshot status counts:** Completed **5** · Ongoing **1** · Pending **0** · Deferred **0** among the six **main experiments**. Individual actions may be Pending or Deferred within an Ongoing experiment; see §9.

**Next active milestone:** Audit the **already-executed** EXP-006 four-model run's `per_target_metrics.csv`, calibrated thresholds, evaluation checks and manifest; append verified details to log §15 and then decide whether EXP-006 meets the **Completed** gate. **Do not record EXP-004's `1.0` reconstruction as predictive performance, or compare EXP-003 and EXP-006 scores as if protocols were identical.**

---

## 3. EXP-001 — Initial dataset and structural audit (`EDA-01`)

**Status: Completed.**  
**Goal:** Establish whether the eight instructor Parquet tables are loadable, what their shape/coverage is, how the PK/FK relations work, who belongs to the 9,000-user cohort, which timestamps reflect feature visibility, and which fields can leak future information.  
**Executed notebook:** `notebooks/01_data_audit.ipynb` (reviewed executed copy `01_data_audit(1).ipynb`).  
**Evidence:** `EXPERIMENT_LOG.md` **§4.1–§4.12**.  
**Data SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b`.

### Completed findings

- **2,552,788 records** across eight tables and **45,590** `Users` records; **9,000** cohort users and **36,590** referenced non-cohort users. Pushes have **1,661,256** rows (approximately **65.08%** of all rows across these heterogeneous tables).
- **No null or duplicated primary keys**; **no orphan references** among the **20** examined non-null foreign-key relationships. This is *within-release* integrity, not cross-release ID stability.
- All official row-time columns were populated. Latest observed event timestamp for the checked snapshot: **2026-10-13 20:59:29 Taiwan** (Pushes).
- Missingness was material: **55.2%** of all PR rows lack `author_id`/`created_at`; Issues and other tables have optional relations. Feature availability must be distinguished from original action timestamps.
- Identified late-known fields such as `merged_at`, `closed_at`, `state_reason`, `dismissed_at` that must be excluded or time-masked for historical features.

**Exit criterion satisfied:** Structural, missingness, feature-time and cohort audits are written in §4, and limitations are explicit.  
**Deferred inside the finding, not this completed milestone:** no real release-to-release drift was measured because one distinct instructor SHA was available; that is a separate future investigation, not evidence of a failed structural audit.

## 4. EXP-002 — Deeper EDA and behavioral hypotheses (`EDA-02`)

**Status: Completed.**  
**Goal:** Quantify time-regime changes, observation-versus-event timestamp disagreements, 90-day eligibility-proxy behavior, recent-activity persistence and repository-breadth associations.  
**Executed notebook:** `notebooks/02_deeper_data_analysis.ipynb` (`02_deeper_data_analysis(1).ipynb` executed evidence).  
**Evidence:** `EXPERIMENT_LOG.md` **§5.1–§5.10** and §6.  
**Data SHA:** Same initial instructor SHA.

### Completed findings

| Measure | Historical early | Dip (Jun–Aug) | Sep/Oct period | Meaning |
|---|---:|---:|---:|---|
| Weekly median active push users | 3,639 | 3,246 | 2,394 | Push participation did **not** recover in last regime |
| Weekly median issue-comment users | 996 | 161 | 470 | Major decline, partial later rebound |
| Weekly median PR-review-comment users | 605 | 70 | 311 | Major decline, partial later rebound |

Prior-week activity was strongly associated with next-week activity: median recurrence-rate lift **4.985×** (push), **14.793×** (issue comment), **21.243×** (PR review comment). Next-week push prevalence rose across prior-30-day unique pushed-repository buckets: **5.26%**, **31.28%**, **46.36%**, **70.65%** for 0, 1, 2–3, 4+ respectively. These are descriptive associations, **not causal evidence or fitted-model gains**. The prior-90-day `>=5` activity-count eligibility **proxy** fell from **95.18%** (Jul 20) to **50.36%** (Oct 5).

**Exit criterion satisfied:** All EDA comparisons and caveats appear in §5; feature hypotheses are explicit in §6.  
**Unanswered questions:** Source/behavior causes of the dip, effects of controlling for volume, and impact of relational features on held-out prediction scores require later experiments.

## 5. EXP-003 — Historical label/split protocol and initial persistence rule (`EDA-03`)

**Status: Completed** for the original exploratory protocol, **not** final learned-model evaluation.  
**Goal:** Construct temporally aligned historical user-week examples, compare event versus observation times, define chronological partitions and eligibility proxy, use the instructor scorer, and test a transparent nontrained baseline.  
**Notebook:** `notebooks/03_label_and_split_validation.ipynb`, executed saved variant `03_label_and_split_validation(5).ipynb`.  
**Evidence:** `EXPERIMENT_LOG.md` **§10**, especially §10.5 (cutoffs), §10.7 (label disagreement), §10.9 (metrics), §10.10 (integrity).

### Completed findings

- **31 weekly cutoffs**, **279,000 user-week rows** across **22 train**, **5 validation**, and **4 originally sealed holdout** weeks; the original training/validation candidate-label population was **243,000** user–weeks. **10/10** temporal/data integrity checks passed.
- The nontrained **previous-week recurrence** baseline produced **0.5462** mean weekly combined F1 in training and **0.3504** in validation for the *eligibility-proxy* population. Original all-cohort validation mean was **0.3469**. All-zero returned zero in evaluated weeks with positives.
- The **0.3504** value averages **five per-week official combined F1 scores**, rather than pooling user-week confusions. It is not directly comparable to EXP-006's proposed pooled multiweek F1 without recomputation under identical splits.
- **Original September holdout is no longer fully independent for label-policy calibration** because EXP-004 later used official example truth overlapping late September. EXP-005 intentionally creates a replacement Oct 12 holdout.

**Exit criterion satisfied:** The historical exploratory split, rule-based scores, limitations and integrity findings are written in §10.  
**Not part of EXP-003 completion:** a learned ML model, a definitive universal maturation period, or a clean untouched final holdout score.

## 6. EXP-004 — Official sample-truth label reconciliation

**Status: Completed** for exact reconstruction of the **provided sample**.  
**Goal:** Resolve historical action-time and window-boundary ambiguity by comparing candidate labels with `assignment2/evaluation/examples/sample_truth.csv` and the exact official evaluation function.  
**Evidence:** `EXPERIMENT_LOG.md` **§11.1–§11.12**; regression replicated later in §14.3.

### Completed investigation

| Step | Diagnostic outcome | Interpretation |
|---|---|---|
| Initial week beginning Sep 27 **00:00 UTC**, event time | **188** mismatches, score **0.978894** | Starting time was misaligned with likely sample week |
| Move start by **+13 hours** (Sep 27 **21:00 Taiwan**) | **0 push** mismatches | Release-aligned example week strongly supported |
| Reconstruct all six labels without as-of filtering | **2** mismatches, score **0.999639** | Both residual errors were late-observed issue openings |
| Limit row observation to **Oct 9, 20:00 or 21:00 Taiwan** | **0 mismatches; 1.000000 macro/micro/combined** | Exact reproduction of all **28,710** sample labels (4,785 users × 6) |

The two residual issue rows were created during the Sept 27–Oct 4 example week but first observed **October 9, 17:36:51 UTC** and **October 10, 05:11:20 UTC**. The eight additional late-revealed *official positives* were observed by **Oct 9, 05:46:50 UTC**. Both candidate Oct 9 cutoffs fall inside the separating interval. This supports—but does not prove—the hypothesis of an October 9 example-truth dataset release. The exact original truth-generation code was not inspected.

**Exit criterion satisfied:** Exact, reproducible match to the supplied sample, full discrepancy chain, and uncertainty documented in §11.  
**Scientific restriction:** This is **reconstruction/calibration agreement**, **not** predictive performance or an independent check of the chosen cutoff. Do not apply the example's Sunday 21:00 start to the official **Nov 12 00:00 Taiwan** prediction window.

## 7. EXP-005 — Reusable temporal dataset generator

**Status: Completed.**  
**Goal:** Replace exploratory notebook-dependent historical dataset construction with testable Python modules; reproduce EXP-004's example labels in real CLI execution; generate point-in-time features, chronological labels and sealed holdout features.  
**Code:** `src/data_loader.py`, `src/labels.py`, `src/features.py`, `src/temporal_splits.py`, `src/dataset_builder.py`; experiment CLI `experiments/exp005_temporal_dataset/build_dataset.py`; `configs/temporal_dataset.json`; `experiments/exp005_temporal_dataset/README.md`.  
**Evidence:** `EXPERIMENT_LOG.md` **§14.1–§14.8**, particularly user-reported real outputs in §§14.2–14.4.

### Completed acceptance evidence

| Test or output | Observed outcome |
|---|---|
| User-side tests (Windows Conda) | **14 passed, 1 skipped**, 6.19 s |
| Actual official `sample-check --require-perfect` | **28,710/28,710** labels exact; official macro/micro/combined score **1.0** |
| Real `build` output | **21 training weeks, 9 validation weeks, 1 sealed holdout week** |
| Training | **189,000** feature rows + **189,000** corresponding label rows |
| Validation | **81,000** feature rows + **81,000** corresponding label rows |
| Holdout | **9,000** feature rows; **no holdout labels** exported |
| Label audit | **31** cutoff records |
| Actual run folder | `experiments/exp005_temporal_dataset/runs/build-506d860693df-6040f635d4/` |

The Pytest Windows environment initially failed on a stale `platformdirs` auto-plugin entry and restricted default Temp permissions. The successful command used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` plus `--basetemp=.\.pytest_tmp`; this workaround and failure history are preserved in §14.2.

**Exit criterion satisfied:** The defined generator milestone now has both **real source-data execution** and recorded outputs in §14, as well as test results and exact example-label regression.  
**Follow-up quality check still open:** Inspect the actual generated `manifest.json`, `label_audit.csv`, feature missingness and row keys; test new instructor releases separately. These are quality/release follow-ups, not evidence that dataset files failed to generate.

## 8. EXP-006 — Historical machine-learning baseline comparison

**Status: Ongoing — all four real baselines fitted and reported; detailed acceptance audit outstanding.**  
**Goal:** Evaluate all-zero, previous-week persistence, six logistic classifiers and six histogram-gradient-boosting classifiers on consistent historical feature/label windows, calibrating thresholds on earlier weeks and reporting on later unseen-to-threshold weeks.  
**Evidence:** `EXPERIMENT_LOG.md` **§15.1–§15.7** (real Windows/Conda commands and exact primary results). The supplied results are from instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`, EXP-005 folder `build-506d860693df-6040f635d4`.  
**User-side tests:** **25 passed, 1 skipped in 9.51s**. **Dry run:** passed, **59 feature columns**, Oct 12 holdout sealed.  
**Actual final four-model run:** `experiments/exp006_baseline_comparison/runs/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/`.

### Executed models and measured primary leaderboard

These are **historical predictive metrics**, unlike EXP-004's calibrated sample-label agreement. They are evaluated on the **`eligible_proxy` subset (16,982 user-week rows)** across **three reporting cutoffs**: **Aug 31, Sep 7, Sep 14, 2026 (Taiwan time)**. The scores below are the user-reported pooled multiweek macro/micro/combined F1 from the run's JSON.

| Rank | Model | Real execution | Macro-F1 | Micro-F1 | **Combined F1** | Fit time reported |
|---:|---|---|---:|---:|---:|---:|
| **1** | **`persistence`** | **Completed** | **0.451781** | **0.457574** | **0.454677** | 0.0s |
| 2 | `hgb` (histogram gradient boosting) | **Completed** | 0.385487 | 0.412047 | **0.398767** | 7.5s |
| 3 | `logistic` | **Completed** | 0.348296 | 0.414767 | **0.381531** | 6.9s in full run |
| 4 | `zero` | **Completed** | 0.000000 | 0.000000 | **0.000000** | 0.0s |

The `persistence` rule led HGB by **0.055910** and logistic by **0.073146** combined F1 under the measured configuration. The user executed a fast three-model run first (logistic fit **6.3s**) and then a four-model run; all overlapping model scores matched between runs. **This is evidence of model ranking on these development weeks, not proof about later October or November generalization.**

### Recorded temporal protocol and population

| Partition | User-week rows | Weeks | Cutoffs, Asia/Taipei |
|---|---:|---:|---|
| Model fit | **189,000** | 21 | Mar 2 – Jul 20 |
| Threshold calibration | **27,000** | 3 | Aug 3, Aug 10, Aug 17 |
| Maturity gap | Excluded | 1 | Aug 24 |
| Primary reporting | **27,000** all cohort; **16,982** eligible proxy | 3 | Aug 31, Sep 7, Sep 14 |
| Example-overlap exclusion | Excluded | 2 | Sep 21, Sep 28 |
| Independent holdout | **9,000 features only** | 1 | Oct 12; **sealed** |

The EXP-006 dry run confirmed the last threshold-calibration label-as-of timestamp equals the first primary cutoff (**Aug 31 00:00 Taiwan**); week boundaries are Monday midnight Taiwan. The headline score pools target confusion counts over multiple user-weeks, **not** the simple arithmetic mean of EDA-03's per-week scores. The user did **not** provide the per-target confusion table or exact six tuned thresholds, so those cannot yet be checked here.

### Findings and limitations

**Observed:** On these three validation weeks, the simple previous-week heuristic beats the two learned models for the instructor-style combined F1. This does **not** determine why: class distribution drift, probability calibration, threshold choices, feature representation, or other factors remain hypotheses requiring per-target diagnostics.

**Do not claim:** the earlier EXP-003 **0.3504** mean-weekly validation score increased to **0.4547** as a like-for-like comparison; they use different windows, label policies and aggregation. EXP-004's **1.0** is label-reconstruction agreement, not a predictive benchmark. Reported primary validation has now informed model choice, so it is a development/model-selection set, **not** an untouched final holdout.

### EXP-006 completion checklist

- [x] Extract EXP-006 package; run combined tests locally (**25 passed, 1 skipped**).
- [x] Run `python -m experiments.exp006_baseline_comparison.train --dry-run` and verify source EXP-005 SHA, **59 features**, selected cutoffs, holdout seal.
- [x] Execute first real `zero,persistence,logistic` comparison, with full primary JSON scores.
- [x] Execute all four models including HGB; record leaderboard and fit times in `EXPERIMENT_LOG.md` **§15**.
- [x] Record exact run paths and the **189k / 27k / 27k** fit/calibration/report protocol.
- [ ] Inspect/export the **actual six-target** `per_target_metrics.csv` (TP/FP/FN, support, F1, precision/recall, split/population).
- [ ] Inspect actual calibration thresholds, `summary.csv`, `manifest.json`, model settings and per-week official-evaluator consistency checks; record warnings and student project commit if available.
- [ ] Once the complete metric audit is recorded in log §15, change the experiment status to **Completed** in this plan.

**Holdout:** The Oct 12 holdout is a separate deferred task, not a reason to manufacture or reveal immature labels. It may be evaluated only after the required label maturation and the model/threshold selection has been frozen.

---

## 9. Deferred and pending work register

There is **no entire experiment numbered EXP-001 through EXP-006 currently in Deferred status**. The following **subtasks and follow-on checks** are consciously postponed; each has an explicit reason and revisit trigger.

| Parent | Task | Task status | Reason | Revisit / resume criterion |
|---|---|---|---|---|
| EXP-006 | Independent Oct 12 historical holdout evaluation | **Deferred** | As of Oct 14, the seven-day outcome window ends **Oct 19 00:00 Taiwan** and the configured additional seven-day maturation ends **Oct 26 00:00**; labels are sealed. Premature unsealing risks incomplete truth and independence. | After Oct 26, with an appropriate newer instructor snapshot, and **only after** model/split/threshold policy is frozen; open holdout once and record it separately. |
| EXP-006 | HGB real training after quick baselines | **Completed subtask** | The four-model user-side run included HGB; its primary proxy-eligible combined F1 was **0.398767** with reported fit time **7.5s**. | Logged under EXPERIMENT_LOG.md §15.3–15.5; per-target metrics still require the separate audit above. |
| EXP-005 | Cross-release regression and retrospective data drift | **Deferred** | The current completed studies use just one distinct instructor SHA. No valid second-snapshot numerical comparison is possible from the supplied evidence. | After a later distinct authorized instructor release is re-cloned, record both SHAs and diff results. |
| EXP-005 | Final official test-user inference features | **Deferred** | `evaluation/test_input.csv` is not scheduled to exist until **Oct 28**; no authorized final test roster available. | Oct 28 official release or later; run inference on exact released `UserID`s without future labels. |
| EXP-006 | Final November forecast and official test outcome | **Deferred** | Final future events occur **Nov 12–19**, beyond the **Nov 11 23:55** deadline; truth is not present at training time. | Rebuild/infer on final authorized Nov 11 snapshot and submit valid predictions. Report official F1 **only if and when** grading or withheld labels become available. |

**Meaningful distinction:** A Deferred **task** does not automatically make its **parent experiment** Deferred if the rest of the experiment remains actively executable. If EXP-006 as a whole is halted (e.g. computational constraints or deadline reprioritization), change it explicitly to `Deferred`, record the reason and restart condition here, and preserve the current implementation progress in the log.

## 10. Dependencies and next decisions

```text
EXP-001  Dataset/schema/temporal audit                 Completed
   |
   v
EXP-002  Follow-up EDA and behavioral hypotheses      Completed
   |
   v
EXP-003  Candidate labels/splits/heuristic baseline   Completed
   |
   v
EXP-004  Official example label reconstruction       Completed
   |
   v
EXP-005  Reusable point-in-time dataset pipeline     Completed
   |
   v
EXP-006  Real predictive baseline comparison         Ongoing
           |
           +-- [DONE] Tests, dry run and four-model training
           +-- [DONE] Primary leaderboard; persistence 0.454677
           +-- [DONE] Actual scores logged in §15
           +-- [OPEN] Six-target metrics / tuned thresholds / audit
           +-- [DEFERRED] Independent later holdout
```

**Immediate decision:** The real EXP-006 leaderboard **is available** and persistence outperforms both learned models. Before expanding into relational/GNN features, inspect six-target errors and threshold calibration, diagnose the train-to-validation regime shift, and decide whether a targeted feature/threshold change warrants a new controlled experiment. Do not repeatedly tune against the future October holdout.

## 11. Maintenance and evidence checklist

Each experiment row should remain traceable. When creating future work or changing a status, record:

- [ ] Unique experiment ID, exact title and status; for **Deferred**, a concrete reason and revisit trigger.
- [ ] Actual instructor dataset SHA and student code commit; do not treat replaceable internal row numbers as stable identifiers.
- [ ] Target definitions, population, timezone, window boundaries, feature visibility and label maturity rules.
- [ ] Executed command/environment, artifact paths and run configuration; distinguish synthetic tests from actual Parquet execution.
- [ ] For predictive experiments: per-target F1/support/confusion counts, official macro/micro/combined F1, aggregation convention, calibration vs independent validation vs holdout.
- [ ] For **Completed**: exact **`EXPERIMENT_LOG.md` result section** and evidence sufficient to reproduce or audit it.
- [ ] For every new result or correction: append to the log **before** updating Completed in this plan.

### References

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), especially **§§4, 5, 10, 11, 14 and 15**.
- `experiments/exp005_temporal_dataset/README.md` and `configs/temporal_dataset.json`: temporal dataset implementation and observed output contract.
- `experiments/exp006_baseline_comparison/README.md` and `configs/baseline_comparison.json`: baseline training configuration and runbook.
- Instructor evaluator: `assignment2/evaluation/evaluate.py` (grading authority).
- Instructor data schema: `assignment2/data/db/SCHEMA.md`.

**Status reviewed again on 2026-10-14 after the real four-model EXP-006 run.** Main experiment counts remain **five Completed, one Ongoing** because mandatory per-target/threshold checks have not yet been supplied. All actual headline EXP-006 scores have been logged in §15. Later runs require dated amendments and must preserve historical benchmark protocols.
