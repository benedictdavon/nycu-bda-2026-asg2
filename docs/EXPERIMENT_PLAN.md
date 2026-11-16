# Assignment 2 — Experiment Plan & Status Register

**Course:** NYCU — Big Data Analytics Techniques and Applications, Fall 2026  
**Project:** Assignment 2 — Predicting GitHub User Activity  
**Last status/strategy review:** 2026-10-23 (Asia/Taipei). **§47 controls the remaining forward priorities and the screening approach for newly named follow-ups.** The bounded Luna Max campaign is complete (§46): Step 0, MLP/TabM, the EXP-041 tabular child, GRU/TCN and their registered controls have accepted real-data results; all model/blend development gates failed. H2/P021 is the current refreshed-source development reference (combined F1 0.547634), not a confirmed replacement for the submission default. Next: a short evidence check, a one-day tabular block, and an early protected stronger-relational trial. New follow-ups are not executed by this amendment. No October confirmation or policy promotion is claimed.
**Related evidence:** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (results and detailed experiment history)  
**Project data source:** Instructor-supplied `assignment2/data/db/*.parquet` only  
**Historical EXP-001–010 instructor SHA:** `506d860693df60a698b68aa63a8edb4a984c8e1b` (new runs must record their actual source release SHA)
**Canonical scope:** original milestones `EXP-001–015`, continuation `EXP-016–030`, feature follow-ups `EXP-031–035` (§30), and stronger-model research `EXP-036–041` (§31). **EXP-001–012 Completed; EXP-013 Ongoing; EXP-014 Pending; EXP-015 Completed — recorded negative.** EXP-014a is a completed negative sub-experiment; EXP-014b is awaiting Stage 2. Completed studies retain their original protocols and conclusions. HGB/F0 remains the original EXP-013 control; H2/F0 is the stronger-model development control. `P_persist7` remains the submission default until the applicable confirmation gates pass.

**Repository paths updated October 14:** experiment workflows and saved runs now live in descriptive experiment folders. Shared modules remain in `src/`; configs are named by purpose. See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md). Historical evidence and statuses are unchanged.


> **Reader's guide (Oct 23 update):** Start with **§47** for current forward work and its explicit precedence rules; read **§46** and **log §§35–45** for the completed Luna Max campaign. **§31** preserves the original stronger-model registration, and **§32** pins that campaign to instructor release `2053302ba75751be975b9e779f567557b66f4071`. Their clocks, visibility, common-source comparisons and confirmation safeguards remain in force except where §47 explicitly defines a new follow-up's approach. Older pending-status statements in §§33–45 are historical; §46 closes that bounded work. **§2** retains the original EXP-001–015 register; log §29 records the feature-screen results. Completed studies, original EXP-013/015 and confirm-plan v0 retain their scientific rules. The Oct 12 holdout remains a non-independent sanity check under D3. `FEATURE_ENGINEERING.md` is a catalogue, not evidence that all combinations were tested. The exact plan before §47 is archived at the path recorded in §47.7; earlier campaign snapshots remain preserved.

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
| **EXP-006** | Learned and heuristic baseline model comparison | **Completed** | Persistence **0.454677** > HGB **0.398767** > logistic **0.381531** > zero **0**; archive, six-target confusion counts, thresholds and official weekly/pooled scores reproduced | **§15**, audit closure **§17.1–§17.3** |
| **EXP-007** | Frozen-model baseline and threshold diagnostics | **Completed** | Refined calibration does not help consistently; eligible-proxy hindsight gains **+0.118812 logistic / +0.078957 HGB**; HGB open_pr exact maximum still trails persistence | **§17.1–§17.5** and diagnostic run `FINDINGS.md` |
| **EXP-008** | Chronological replay of threshold transfer | **Completed** | Archived control reproduced; rolling one/three mature-week calibration measured on the same three reporting weeks; neither learned model exceeds persistence in pooled eligible-proxy score | [`EXPERIMENT_LOG.md` §18](EXPERIMENT_LOG.md#18-exp-008-chronological-threshold-transfer-replay--october-15-2026) and final run `experiments/exp008_chronological_threshold_transfer/runs/run-frozen-fc2acd6fe5b2-rolling-1-3-final-v2/` |
| **EXP-009** | Controlled HGB training-recency replay | **Completed** | Recent-eight-week HGB improves open_pr ranking and rolling F1 over the frozen model, but open_pr F1 and six-target pooled combined F1 remain below persistence | [`EXPERIMENT_LOG.md` §19](EXPERIMENT_LOG.md#19-exp-009-controlled-hgb-training-recency-replay--october-15-2026); final run `experiments/exp009_hgb_recency/runs/run-hgb-recency-final-v2/` |
| **EXP-010** | Controlled push-target combinations | **Completed** | Calibration selects HGB push at 0.45 for all three origins; selected policy scores 0.465391 pooled eligible-proxy combined F1 vs persistence 0.454677; only inspected development weeks | [`EXPERIMENT_LOG.md` §20](EXPERIMENT_LOG.md#20-exp-010-controlled-push-target-combinations--october-15-2026); final run `experiments/exp010_push_combinations/runs/run-push-combinations-final-v2/` |
| **EXP-011** | Fixed-F0 stronger tabular learner benchmark | **Completed** | Retain HGB: all six archival candidates fail the screen; rolling HGB 0.405331 versus XGBoost 0.402824 and CatBoost 0.398286, pooled legacy-proxy combined F1 | [`EXPERIMENT_LOG.md` §22](EXPERIMENT_LOG.md#22-exp-011-stronger-tabular-learners--october-16-2026); [independent verification](../reports/EXP011_VERIFICATION.md) |
| **EXP-012** | Bounded temporal rhythm/state ablations | **Completed** | Retain F0: state fails reporting stability (median delta −0.000571; 2/4 nonnegative weeks); rhythm fails earlier eligibility | Log **§23.3–23.4**; [independent verification](../reports/EXP012_VERIFICATION.md) |
| **EXP-013** | Original bounded repository-self/neighbor ablations | **Ongoing** | Full source preflight passed; incompatible-runtime controls failed parity; accepted runtime reproduced one fold exactly. Full 12-fold control replay/candidate study remains incomplete; no original candidate fit/result | Log **§29**, final continuation status; bounded closure **§31.12** |
| **EXP-014** | Complete six-target policy integration | **Pending** | Final policy not selected. New bounded neural–tree ensemble research belongs to EXP-041; integration follows applicable frozen-candidate confirmation | **§31.9–31.13** |
| **EXP-015** | Bounded MLP/heterogeneous-GNN pilot | **Completed — negative** | GNN1T AP 0.5126 and GNN1 0.5063 versus REF 0.5312 and H2 0.5375; neither GNN eligible | Log **§28.6**; plan **§29.7** final addendum |

**Main experiment status counts (EXP-001–015):** Completed **13** · Ongoing **1** · Pending **1**. This count excludes suffixed sub-experiments and EXP-016 onward. Development screens are not final policy promotion. The authoritative remaining-work register is **§47.3**; independent November performance remains unestablished.

**Independent verification:** The primary agent independently reran EXP-008–010,
recomputed their scores and calibration decisions, and verified input preservation.
Full regression: **57 passed, 1 skipped**. Evidence is recorded in log §21 and
[the verification report](../reports/EXP008_010_VERIFICATION.md).

**Historical verification checkpoint (EXP-012, October 18):** **114 passed, 1 skipped**; the
independent auditor reproduced 38 saved bundles, 25 threshold searches,
192 official weekly scores and 48 pooled groups. All 122 protected files remain
unchanged. See log §23.4 and [the report](../reports/EXP012_VERIFICATION.md).

**Most recent accepted results:** log **§§35–45** and plan **§46** close the refreshed-source tabular, blend, temporal and temporal-control packages as verified development-gate negatives. Log **§29** retains the seven feature-screen rejections, conditional B/AB skips and original EXP-013's incomplete execution. The earlier test counts above are historical checkpoints. The October 23 strategy amendment updates documents only; it performs no new training or reserved October-outcome evaluation.

**Next milestones (Oct 23):** follow **§47.3**: bounded evidence/trainer checks; one day total for current HGB capacity, TabM numerical embeddings and at most one optional current-protocol tree challenger; an early protected **RelGNN-inspired trial**, with RelGT as a score-free implementation alternative; then one justified follow-up and a shortlist freeze by **November 2**. Count calibration is an optional short test with ready out-of-time inputs. Simple pooling or handcrafted-feature losses do not block the relational trial. Original EXP-013 closure remains secondary and does not block new work.

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

**Status: Completed — four-model comparison and detailed acceptance audit recorded.**
**Goal:** Evaluate all-zero, previous-week persistence, six logistic classifiers and six histogram-gradient-boosting classifiers on consistent historical feature/label windows, calibrating thresholds on earlier weeks and reporting on later unseen-to-threshold weeks.  
**Evidence:** `EXPERIMENT_LOG.md` **§15.1–§15.7** (original training) and **§17.1–§17.3** (executed archive/audit closure). The results use instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`, EXP-005 folder `build-506d860693df-6040f635d4`.
**User-side tests:** **25 passed, 1 skipped in 9.51s**. **Dry run:** passed, **59 feature columns**, Oct 12 holdout sealed.  
**Actual final four-model run:** `experiments/exp006_baseline_comparison/runs/run-build-506d860693df-6040f635d4-fc2acd6fe5b2/`.

### Executed models and measured primary leaderboard

These are **historical predictive metrics**, unlike EXP-004's calibrated sample-label agreement. They are evaluated on the **`eligible_proxy` subset (16,982 user-week rows)** across **three reporting cutoffs**: **Aug 31, Sep 7, Sep 14, 2026 (Taiwan time)**. EXP-007 independently reproduced these archived pooled macro/micro/combined F1 values, all six-target metrics and weekly official checks.

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

The EXP-006 dry run confirmed the last threshold-calibration label-as-of timestamp equals the first primary cutoff (**Aug 31 00:00 Taiwan**); week boundaries are Monday midnight Taiwan. The headline score pools target confusion counts over multiple user-weeks, **not** the simple arithmetic mean of EDA-03's per-week scores. Actual six-target confusion counts and thresholds are now checked and recorded in log §17.3 for both populations, with full calibration/primary and weekly tables in the diagnostic artifacts.

### Findings and limitations

**Observed:** At archived calibration thresholds, persistence beats the learned models' pooled combined F1 on the three primary weeks. EXP-007 finds higher learned-model ranking AP but low non-push recall under a large prevalence shift; hindsight threshold gains exceed 0.02 while finer calibration does not improve consistently. Threshold transfer is the next hypothesis to test chronologically. HGB open_pr also has a target-specific threshold/ranking limitation; see log §17.4–§17.5.

**Do not claim:** the earlier EXP-003 **0.3504** mean-weekly validation score increased to **0.4547** as a like-for-like comparison; they use different windows, label policies and aggregation. EXP-004's **1.0** is label-reconstruction agreement, not a predictive benchmark. Reported primary validation has now informed model choice, so it is a development/model-selection set, **not** an untouched final holdout.

### EXP-006 completion checklist

- [x] Extract EXP-006 package; run combined tests locally (**25 passed, 1 skipped**).
- [x] Run `python -m experiments.exp006_baseline_comparison.train --dry-run` and verify source EXP-005 SHA, **59 features**, selected cutoffs, holdout seal.
- [x] Execute first real `zero,persistence,logistic` comparison, with full primary JSON scores.
- [x] Execute all four models including HGB; record leaderboard and fit times in `EXPERIMENT_LOG.md` **§15**.
- [x] Record exact run paths and the **189k / 27k / 27k** fit/calibration/report protocol.
- [x] Inspect/export the **actual six-target** metrics: archive reproduction and complete both-population/slice/week audit in log §17.2–§17.3.
- [x] Verify calibration thresholds/traces, summary, manifest/config, all official weekly checks; source commit/file hashes and cross-version warnings recorded in log §17.1–§17.3.
- [x] Append complete metric audit to log §17, then mark EXP-006 **Completed** here. Earlier §15 status is preserved as dated history.

**Holdout:** The Oct 12 holdout is a separate deferred task, not a reason to manufacture or reveal immature labels. It may be evaluated only after the required label maturation and the model/threshold selection has been frozen.

---

## 9. Deferred and pending work register

There is **no entire experiment numbered EXP-001 through EXP-010 currently in Deferred status**. The following **subtasks and follow-on checks** are consciously postponed; each has an explicit reason and revisit trigger.

| Parent | Task | Task status | Reason | Revisit / resume criterion |
|---|---|---|---|---|
| EXP-006 | Independent Oct 12 historical holdout evaluation | **Deferred** | As of Oct 14, the seven-day outcome window ends **Oct 19 00:00 Taiwan** and the configured additional seven-day maturation ends **Oct 26 00:00**; labels are sealed. Premature unsealing risks incomplete truth and independence. | After Oct 26, with an appropriate newer instructor snapshot, and **only after** model/split/threshold policy is frozen; open holdout once and record it separately. |
| EXP-006 | HGB real training after quick baselines | **Completed subtask** | The four-model run included HGB; its primary proxy-eligible combined F1 was **0.398767** with reported fit time **7.5s**. | Training in log §15.3–15.5; per-target/threshold/evaluator audit completed in §17. |
| EXP-005 | Cross-release regression and retrospective data drift | **Deferred** | The current completed studies use just one distinct instructor SHA. No valid second-snapshot numerical comparison is possible from the supplied evidence. | After a later distinct authorized instructor release is re-cloned, record both SHAs and diff results. |
| EXP-005 | Final official test-user inference features | **Deferred** | `evaluation/test_input.csv` is not scheduled to exist until **Oct 28**; no authorized final test roster available. | Oct 28 official release or later; run inference on exact released `UserID`s without future labels. |
| EXP-006 | Final November forecast and official test outcome | **Deferred** | Final future events occur **Nov 12–19**, beyond the **Nov 11 23:55** deadline; truth is not present at training time. | Rebuild/infer on final authorized Nov 11 snapshot and submit valid predictions. Report official F1 **only if and when** grading or withheld labels become available. |

**Meaningful distinction:** A Deferred **task** does not automatically make its **parent experiment** Deferred if the rest of the experiment remains actively executable. If EXP-006 as a whole is halted (e.g. computational constraints or deadline reprioritization), change it explicitly to `Deferred`, record the reason and restart condition here, and preserve the current implementation progress in the log.

## 10. Dependencies and next decisions

**Historical sequencing note:** The flow and decision below record the state at
EXP-007 close. Their pending EXP-008 status was superseded by the executed replays
in §§13–15 and log §§18–20; the current status register is the table in §2.

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
EXP-006  Real predictive baseline comparison         Completed
           |
           +-- [DONE] Tests, dry run and four-model training
           +-- [DONE] Primary leaderboard; persistence 0.454677
           +-- [DONE] Actual scores logged in §15
           +-- [DONE] Six-target metrics / thresholds / official audit (§17)
           +-- [DEFERRED] Independent later holdout
           |
           v
EXP-007  Frozen baseline diagnostics                 Completed
           +-- [DONE] Exact archive reproduction; controlled threshold/ranking audit
           +-- [DONE] Decision and limitations documented (§17)
           |
           v
EXP-008  Chronological threshold-transfer replay     Completed
   |
   v
EXP-009  Controlled HGB training-recency replay       Completed
   |
   v
EXP-010  Controlled push-target combinations          Completed
```

**Decision at EXP-007 close (historical):** Prioritize EXP-008 chronological replay before changing training or adding relational/GNN features. EXP-007's measured hindsight opportunity is substantial, but calibration search does not transfer consistently. Investigate HGB open_pr recency/representation separately and carry push complementarity into a later controlled combination experiment. The scope was subsequently executed as EXP-009/010 and documented in §§14–15 below. Do not promote hindsight thresholds or tune against the October holdout.

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

- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), especially **§§4, 5, 10, 11, 14, 15, and 18–20**.
- `experiments/exp005_temporal_dataset/README.md` and `configs/temporal_dataset.json`: temporal dataset implementation and observed output contract.
- `experiments/exp006_baseline_comparison/README.md` and `configs/baseline_comparison.json`: baseline training configuration and runbook.
- Instructor evaluator: `assignment2/evaluation/evaluate.py` (grading authority).
- Instructor data schema: `assignment2/data/db/SCHEMA.md`.

**Historical status note:** This paragraph records the 2026-10-14 EXP-007 review and its then-current count of seven Completed and one Pending. The current count and review date are maintained in §2; EXP-008–010 completion evidence is recorded in log §§18–20.

## 12. EXP-007 — Frozen baseline diagnostics

**Status: Completed.** **Evidence:** `EXPERIMENT_LOG.md` §17.1–§17.5 and `experiments/exp007_baseline_diagnostics/runs/run-frozen-fc2acd6fe5b2-grid001-final/FINDINGS.md`.

The runner reuses EXP-005 loading, EXP-006 splits, compatibility model loading,
model probabilities and shared official scoring. The config pins dataset/model
runs, 27 input hashes, 59 features, cutoffs, tolerances and output location.
No models were fitted or rewritten; October 12 labels remain sealed.

- [x] Exact primary binaries and thresholds; original calibration traces; complete archived metrics/weekly evaluator checks reproduced.
- [x] Confusion audit, both populations, weekly/pooled scores, calibration-to-primary prevalence/operating points.
- [x] AP/ROC-AUC by target/week with explicit undefined cases and prevalence.
- [x] Matched refined calibration/hindsight local searches and exact individual-target threshold maxima.
- [x] Alignment/input/access/immutability tests; final full regression 45 passed, 1 skipped; final findings acceptance suite 17 passed.
- [x] Generated artifacts, source/runtime provenance, limitations and next-experiment decision documented.

Eligible-proxy hindsight scores reach **0.500343 logistic** and **0.477724 HGB**,
versus persistence **0.454677**, but refined calibration changes are only
**+0.001417 / −0.006327** and are not consistently favorable across weeks.
Logistic individual maxima exceed persistence for all six targets; HGB open_pr
still trails (**0.400742 vs 0.454614**). Both learned models have higher weekly AP
than persistence in all 18 eligible target-week comparisons. Push supplies new
true positives in all three weeks with additional false positives; learned push
F1 wins only two of three weeks for each model. These are development diagnostics,
not independent November evidence or promoted production thresholds.

## 13. EXP-008 — Chronological threshold-transfer replay

**Status: Completed.** **Goal:** Test whether thresholds recalibrated from mature
historical activity transfer across the three EXP-006 reporting origins. EXP-007
showed a substantial hindsight opportunity but inconsistent refined calibration.

The replay keeps EXP-006's four models, 59 features, and archived thresholds
unchanged. It compares archived thresholds with deterministic thresholds tuned on
the latest one or three validation weeks whose `label_as_of_utc` is no later than
each reporting cutoff. Tuning retains the EXP-006 eligible-proxy population, grid,
and three-pass combined-F1 coordinate search. The three archived reporting weeks
and both all-user/eligible-proxy scoring populations are identical across modes.
Weeks overlapping the Sep 27 calibrated example are excluded under the original
EXP-006 interval rule. October 12 remains sealed.

**Acceptance met:** Input/config/evaluator hashes are pinned; the mature windows,
origins, thresholds, traces, probabilities, per-origin binary predictions, weekly
and pooled official scores, per-target confusion/ranking metrics, week-to-week
variance, archive reproduction, and limitations are retained under
`experiments/exp008_chronological_threshold_transfer/runs/run-frozen-fc2acd6fe5b2-rolling-1-3-final-v2/`.
No model was fitted or written. The saved archived binaries and all 24 archived
weekly official checks reproduce. The measured results remain development evidence
from three already-inspected weeks and do not justify threshold promotion.

**Scope clarification (October 15):** The user assigned EXP-009 to HGB training
recency and EXP-010 to push-target combinations. This supersedes the earlier
proposal in log §12.2 to use those numbers for release-drift and independent-label
policy checks; that historical proposal is retained in the log as written.

## 14. EXP-009 — Controlled HGB training-recency replay

**Status: Completed for the configured development replay.** **Evidence:**
[`EXPERIMENT_LOG.md` §19](EXPERIMENT_LOG.md#19-exp-009-controlled-hgb-training-recency-replay--october-15-2026)
and `experiments/exp009_hgb_recency/runs/run-hgb-recency-final-v2/`.

The replay holds features and HGB parameters constant while comparing the frozen
21-week archive, a current-runtime 21-week refit, and the predeclared recent-eight-
week refit. All training labels mature by the first calibration origin. Archived,
one-week rolling, and three-week rolling threshold policies use the same three
report weeks; calibration sees only mature prior validation labels. The run keeps
serialized experimental bundles, report probabilities/predictions, per-target and
ranking metrics, training/calibration window audits, variance, search traces, and
input/source hashes. It does not read or create holdout labels.

The full-history refit reproduces the archive's probabilities exactly in the
recorded runtime. For eligible-proxy open_pr, recent-eight-week training increases
average precision from **0.409868** to **0.538654** and rolling-1 F1 from **0.319636**
to **0.394233** versus the frozen model, but its best rolling F1 (**0.394233**) is
still below persistence (**0.454614**). Pooled six-target combined F1 for recent8
with rolling-three is **0.426468**, below persistence (**0.454677**). Keep the
archived model as the control; these three inspected weeks do not support promotion.
Focused tests: **3 passed**. Independent reproduction matched the compact evidence
and direct weekly official scores; model archive bytes can differ after refitting,
while parameters and report probabilities reproduce semantically.

## 15. EXP-010 — Controlled push-target combinations

**Status: Completed for the configured development replay.** **Evidence:**
[`EXPERIMENT_LOG.md` §20](EXPERIMENT_LOG.md#20-exp-010-controlled-push-target-combinations--october-15-2026)
and `experiments/exp010_push_combinations/runs/run-push-combinations-final-v2/`.

The replay preserves persistence on the five non-push targets and compares the
frozen HGB and logistic push scores with persistence. Its fixed 13-policy candidate
set covers learned thresholds, AND/OR rules, and convex blends at weights **0.25,
0.50, and 0.75**. At each origin the policy and eligible-proxy push threshold are
selected from the latest three mature prior validation weeks under the official
combined-F1 objective. The three report weeks are evaluated only after selection;
report labels are not available to selection. The saved run includes all per-origin
candidate/threshold traces, chosen policies, probability and binary predictions,
per-target/weekly/pooled metrics, true-positive overlap, variance, and hashes.

Calibration selects HGB tuned push at threshold **0.45** at all three origins.
The selected policy's pooled eligible-proxy combined F1 is **0.465391**, compared
with persistence at **0.454677**; across the fixed candidates, logistic blend 0.25
and 0.50 reach **0.474577** on the same reports, but those reporting scores are
descriptive and do not replace the calibration choice. Selected push F1/AP is
**0.520121 / 0.575500**. The result is a small development replay, not evidence for
production promotion or independent future performance. Focused tests: **4 passed**;
independent calibration recomputation verified all candidate selections and
threshold-grid objectives. The seven-day label-maturation allowance is a
reconstruction policy, not a completeness guarantee. October 12 remains sealed.

---

## 16. Research continuation (2026-10-16): scope, governance, and decision register

> **Document control.** This continuation is the prospective research protocol for **EXP-011–EXP-015**, written after the executed and independently checked EXP-008–010 replays. Sections 3–15 above remain a historical account of experiments actually performed. No experiment described below has been run merely because it appears in this document. All new numerical thresholds, budgets, and stop criteria are **project decisions or proposed heuristics**, not empirically established optimal values. Recorded evidence belongs in `EXPERIMENT_LOG.md` before a status change here.

### 16.1 Research question, objective, and falsifiable hypotheses

**Primary research question.** Given only the instructor-supplied, incrementally updated eight-table relational record of a 9,000-user GitHub cohort, which reproducible, as-of-valid learning system best predicts six binary activities in the next official seven-day interval for a preselected test-user roster, maximizing the instructor's `(macro-F1 + micro-F1)/2`?

**Unit of inference:** `(stable_github_user_id, target_start_utc)`. **Six outcomes:** `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`. **Prediction is occurrence, not count.** The target interval is half-open. Repeated user-week samples are correlated; 9,000 users observed over 10 weeks do **not** constitute 90,000 independent experimental replications.

**Primary objective:** maximize defensible expected future-window combined F1 at the official test population. **Secondary objectives:** stability across recent time regimes; robust rare-target F1; defensible provenance and leakage resistance; manageable training and final-day inference cost; transparent Q2/Q3 experimental evidence and reproducible assignment submission.

| Hypothesis | Minimal falsifiable comparison | Null/counter-hypothesis | Experiment |
|---|---|---|---|
| H11: a stronger tabular learner better exploits F0 | Same 59 F0 columns, same mature training/folds/threshold policy; compare LightGBM, CatBoost, XGBoost vs HGB | The low-capacity/simple prior-history decision remains better | 011 |
| H12a: personal temporal rhythms carry incremental signal | `F0` vs `F0 + F-TEMP-RHYTHM`; identical model/folds | Features duplicate already captured recency | 012 |
| H12b: weekly states, transitions and cross-activity patterns carry incremental signal | `F0` vs `F0 + F-TEMP-STATE`, then `F-CROSS` | Sparse estimates and shifting prevalence make patterns unstable | 012 |
| H13a: repository specialization adds beyond own event volume | Own-event control + `F-REPO-SELF`, volume-matched comparisons | Repo breadth simply proxies activity count | 013 |
| H13b: other cohort users' activity predicts the focal user's next activity | Repo-self control vs +`F-REPO-NEIGHBOR`, strictly excluding self | Incomplete and sparse cohort network adds noise | 013 |
| H13c: current workflow and collaborator context adds target-specific value | Add individually audited PR/issue/review groups | Historical field availability or missingness overwhelms signal | 013 |
| H14: target-specific choice and sparse ensembles beat any single global classifier | Frozen candidate library with prospective calibration/decision policy | Complexity and threshold transfer cost exceed measured benefit | 014 |
| H15: temporal relational message passing adds value beyond handcrafted relations | Same folds/allowed records, 2-layer relation-aware baseline vs strongest tabular relational system | Sparse, partial graph or extra variance negates benefit | 015 (conditional) |

**No score improvement is assumed.** An experiment that rejects a hypothesis is completed successfully if it executes a valid comparison, logs evidence, and records a reproducible decision.

### 16.2 Nonnegotiable course contract

1. **Permitted data:** the eight Parquet tables in the instructor's course repository only. Background research papers, open-source model implementations and general-purpose libraries may guide modeling; they do **not** authorize adding GitHub API profiles, GH Archive events, repository metadata, external text corpora, GitHub activity or web-scraped labels. Treat external pretrained embeddings as unapproved pending explicit TA clarification.
2. **Correct feature timestamp:** `first_seen_at` for `Users`, `Repositories`, `PullRequests`, `Issues`, `PRReviews`; `created_at` for `IssueComments`, `PRReviewComments`, `Pushes`. For a feature cutoff `c`, retain only records with their official visibility/row time **strictly `< c`**. Also mask values of separately late-known fields until their verified availability. Avoid `Issues.state_reason` until its historical value/availability is independently justified.
3. **Historical target semantics:** event occurrence should use candidate original `created_at`/`submitted_at` where appropriate, with original actor and non-PR issue filtering. PR/Issue/Review first observation is not proof of a new action. The successful EXP-004 sample reconstruction is *in-sample reconciliation*, not an all-weeks completeness proof.
4. **Time splitting:** chronological, never random user-week split. Training *labels*, training statistics, categorical/text encoders, feature selection, calibration and thresholds must all obey simulated as-of fit times.
5. **Official metric and artifact:** exactly one row per test `Users.github_user_id`; six binary outputs; exact header `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`; score from instructor's `evaluation/evaluate.py`, including the special per-target empty/empty F1=1 convention.
6. **Submission timeline:** final data released Nov 11, 21:00 Asia/Taipei, with source coverage through Nov 11, 20:00; target Nov 12, 00:00 to Nov 19, 00:00; assignment deadline Nov 11, 23:55. Maintain a runnable and verified fallback model well before the final release.
7. **Scientific reporting:** measured only, no fabricated scores, no claim of novel causal inference, no conclusion that GNN/RDL is required. Distinguish preliminary development, independent test, and prospective deployment evidence.

**Primary authorities:** `Big Data Course - Assignment 2.pdf` §§1–3, course `data/db/SCHEMA.md`, `data/README.md`, `evaluation/README.md`, and exact `evaluation/evaluate.py`. See §28 for references.

### 16.3 Two specifications, one implementation

- **`FEATURE_ENGINEERING.md` is the authoritative candidate-level feature *catalog*.** It contains 173 **templates**, which may expand into far more than 173 columns; it is not evidence that any new feature is implemented or useful. The correct group register is `F0`, `F-TEMP-BASE`, `F-TEMP-RHYTHM`, `F-TEMP-STATE`, `F-TEMP-MOMENTUM`, `F-CROSS`, `F-REPO-SELF`, `F-REPO-NEIGHBOR`, `F-PR-WORKFLOW`, `F-ISSUE-WORKFLOW`, `F-REVIEW-COLLAB`, `F-GRAPH-2HOP`, `F-REGIME`, `F-META-AUTO`.
- **This plan is the authoritative experiment schedule, protocol, acceptance gate, and status register.** Where `FEATURE_ENGINEERING.md` still describes old proposed experiment numbering, the *actual historical EXP-008/009/010 work and this continuation* override its scheduling labels, **not** its feature formula specifications.
- **`EXPERIMENT_LOG.md` is the authoritative empirical ledger.** Keep append-only historical entries. Its opening status paragraph dated Oct 14 is superseded by its appended §§17–21 and by the current overview above; synchronize its opening summary separately, without altering historical numeric records.
- **Actual code/config, not either document, determines what F0 implements.** First map all 59 frozen columns and original code-defined transformations to catalog IDs; do not count a renamed or trivially transformed column as a novel feature.

### 16.4 Current prospective register and dependencies

> **Historical register:** the following rows describe the October 18 planning snapshot. Current statuses and execution order are **§2 and §30.3**. In particular, EXP-015 is now Completed, EXP-022 has run, and D3 has resolved the holdout's intended use.

| ID | Scope | Status on 2026-10-18 | Depends on | Deliverable / completion criterion |
|---|---|---|---|---|
| **EXP-011** | Strong tabular learner control | **Completed** | Six fixed-F0 configurations executed; all fail the registered screen; HGB retained for EXP-012 | [Execution plan](exp/EXP011_EXECUTION_PLAN.md); log §22.3 results and §22.4 independent acceptance |
| **EXP-012** | Temporal, behavior, state and cross-target feature ablations | **Completed — bounded rhythm/state core** | Registered core executed with Luna Max implementation and primary-agent acceptance; HGB/F0 retained | [Verification](../reports/EXP012_VERIFICATION.md); log §23.3–23.4. Union/removal inapplicable after rhythm fails; expanded groups deferred |
| **EXP-013** | Relational, neighbor and workflow feature ablations | **Ongoing** | Accepted HGB/F0 from 012; Luna Max implementation started; parent evaluation pending | [Execution plan](exp/EXP013_EXECUTION_PLAN.md); log §24. No experiment results claimed yet |
| **EXP-014** | Target-wise model/feature/ensemble and threshold policy | **Pending** | 011–013 candidates, OOF predictions | One complete frozen policy with executable final-inference fallback |
| **EXP-014a** | Recent-regime rolling evaluation and calibrated decision layer (sub-experiment on the F0 HGB control; not the registered EXP-014) | **Completed — negative** | F0 HGB control; five report origins Sep 2–30 (09-30 provisional) | No policy beats 7-day persistence in recent3 (0.5075) or all5 (0.4763); best learned is about 0.44 / 0.42. Log §25; run `experiments/exp014_recent_regime/runs/exp014-recent-regime-v1/`. Persistence retained as reference |
| **EXP-014b** | Raw-p top-k and same-model out-of-time Platt decision layer | **Pending — draft preregistration** | EXP-014a run | [Draft](exp/EXP014B_PREREGISTRATION.md); §29 item 1 |
| **EXP-016–030** | Score-maximization backlog: decision layer, rule baselines, ranking quality, roster alignment, fresh-origin confirmation, final freeze | **Pending** (EXP-026 Scheduled for Oct 28) | See §29.3 per item | §29.3 register, §29.4 cards, §29.5 timeline |
| **EXP-015** | Temporal relational deep learning | **Pending — bounded pilot admitted 2026-10-18 (D9)** (was Deferred — conditional) | EXP-014a protocol, EXP-022 REF/H2, `asg2-rdl` environment; §23.1 gate 1 waived by D9 | Explicit go decision given (§29.7). Deliverable: AP-selected pilot result before freeze #1, replacing the Oct 26 memo. [Draft preregistration](exp/EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md) |
| **VAL-LOCK** | Oct 12 sealed historical holdout examination | **Deferred** | maturity + final freeze + new release | Single documented independent evaluation, not iterative tuning |
| **RELEASE** | Official Oct 28 roster and Nov 11 inference | **Scheduled** | runnable champion, fresh authorized snapshots | Valid submission file, pinned provenance, reproducible code/report |

**Priority order:** integrity/eligible-population alignment → small EXP-011 baseline matrix → EXP-012 temporal groups → EXP-013 relational groups → EXP-014 deployment freeze → EXP-015 only if unusually clear value and time. Temporal feature prototyping may overlap with EXP-011 *after* comparator matrix and fold definitions are frozen; do not block all useful features waiting for every model-family configuration.

### 16.5 Change control and completion rules

**Run identity:** immutable `experiment_id`, `substudy_id`, `data_sha`, `student_code_sha`, `feature_manifest_sha`, `split_policy_sha`, `target_policy_version`, `estimator_config_sha`, `threshold_policy_sha`, `seed`, `environment`, and `run_id`. An input/configuration change means a new run ID. Never overwrite prior `runs/*` outputs.

**Status changes:** Pending → Ongoing when code/experiment actually begins; Ongoing → Completed only after an executed real-data artifact, immutable input manifest, independent metric parity check and corresponding full log entry exist. A suspended whole experiment is Deferred with trigger; rejection of an *individual candidate* does not defer the main experiment.

**Decision vocabulary:** `promote`, `retain_for_next_ablation`, `inconclusive`, `reject`, `blocked`, `not_run`. `promote` means *eligible for subsequent evaluation*, not evidence of official November gains. A holdout-based decision can evaluate the selected pipeline once, but the same holdout cannot be recycled as the next development loop without reclassifying it as development data.

---

## 17. Evidence anchors, experimental threats, and immutable comparators

### 17.1 Verified historical controls (never relabel these as official November performance)

**Shared historical scope:** instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`; `eligible_proxy` **16,982 user-weeks** pooled over reporting cutoffs **2026-08-31, 09-07, 09-14 00:00 Asia/Taipei**. Results are development evidence selected/inspected repeatedly.

| Evidence | Pooled eligible-proxy combined F1 | Provenance and intended role |
|---|---:|---|
| Previous-week persistence | **0.454677** | EXP-006; nontrained fixed control |
| EXP-006 frozen HGB, 59 features | **0.398767** | Original learned-feature comparator |
| EXP-006 logistic, 59 features | **0.381531** | Linear/sanity comparator |
| All-zero | **0.000000** | Historical diagnostic, not a universal F1 guarantee |
| EXP-009 recent-8-week HGB + rolling-3 thresholds | **0.426468** | Different fit/threshold protocol; not a direct representation-only comparison |
| EXP-010 calibration-selected persistence + HGB push (`threshold=0.45`) | **0.465391** | Selected hybrid reference; **+0.010714** over persistence on inspected development weeks |

EXP-007 *hindsight* diagnostic scores and unselected EXP-010 blends must **not** be promoted or treated as prospective production comparisons. In particular the unselected logistic blend reaching 0.474577 on pooled reports was identified **after report outcomes were observed**. The EXP-009 open-PR AP improved from 0.409868 to 0.538654 under recent-eight-week training, but its best rolling-1 `open_pr` F1 0.394233 remained below persistence 0.454614. These mixed findings motivate representation analysis, not a claim that any model upgrade is proven.

### 17.2 Primary threats to internal and external validity

| Threat | Consequence | Required control / falsification |
|---|---|---|
| Late entity discovery | Historical labels or features retroactively change | Filter feature rows by observation time; explicit label-as-of and lag strata; multi-release checks where possible |
| Later-populated entity attributes | Future information enters past | Field-availability audit; conservative exclusions; property tests at many cutoffs |
| Incorrect key namespace | False graph edges / distorted cross-repo counts | Within-release PK/FK join; stable GitHub IDs across releases; compare `head_repo_id` to **`Repositories.github_repo_id`**, not row-number `repo_id` |
| Incomplete non-cohort histories | Spurious neighbor activity rates | Cohort-only observed event denominator; self-exclusion; missing/coverage flags |
| Regime shift / dataset collection variation | Calibration and ranking collapse | Recent forward folds, target prevalence, train/report time stratification |
| Label-maturation assumption | Recent false negatives and optimistic metrics | As-of sensitivity 0/3/5/7/14d where supportable; do not declare 7d sufficient without evidence |
| Eligibility/roster selection | All-9k scores misrepresent official test population | All-cohort + old `eligible_proxy` + **roster-anchor proxy**, later actual test roster |
| Repeated inspection of three development weeks | Selection-induced optimism | Additional chronology where feasible; bounded search and one untouched final holdout |
| Same users repeated weekly | Underestimated statistical uncertainty | Per-reporting-week paired analysis; calendar-block resampling where feasible |
| Metric mismatch / rare classes | Invalid promotion, wrong score | Direct official implementation parity; per-target TP/FP/FN, AP and F1; target support counts |
| Threshold interactions | Individual target F1 improvement can reduce global score | Official six-target joint objective evaluated with calibrated joint thresholds |
| Overly large feature search | Hypothesis fishing and hidden compute cost | Group gates, fixed candidate lists, trial ledger, additive and removal ablations |
| Leakage through encoders / normalization | Future distribution or target information used | Fit transformations on fit-only samples; frozen transform per origin; as-of-safe group statistics |
| Missing field wrongly treated as zero | Systematic missingness bias | NaN + support/coverage indicators; explicitly distinct zero event counts |

### 17.3 Baseline and challenger policy

A new model is compared in two different ways, which must not be conflated:

**Scientific representation ablation:** `same learner + same fitted training sample policy + same tuning policy + unchanged input F0` against `same setup + precisely one new feature group`. This isolates feature contribution. Keep persistence, archived HGB and archived EXP-010 hybrid as context, not substitute the learner in only one arm.

**Full-system challenger:** choose a feature-augmented learner and compare its *pre-registered operational policy* with (i) persistence and (ii) the reproducible, calibration-selected EXP-010 hybrid, using paired fold predictions on the same users and targets. If the hybrid cannot be reproduced at the new reporting origins, label its score `not available` rather than carrying forward 0.465391 as a comparable score.

Set aside a fallback deployment policy (persistence plus any completed, reproducibly calibrated EXP-010 rule) with a known-good end-to-end CSV generator. A newly fit learner cannot replace that fallback merely by producing higher AP on an inspected development week.

---

## 18. Evaluation design (mandatory for EXP-011–015)

### 18.1 Temporal variables and as-of reconstruction

For a hypothetical historical test origin `i`, define:

- `S_i`: target interval start (Taiwan-local), with truth window `[S_i, S_i + 7 days)`.
- `C_i`: **feature observation cutoff**, strictly before `S_i`; for *deployment-matched simulation*, set `C_i = S_i − 4 hours`, echoing the final Nov 11 20:00 vs Nov 12 00:00 gap. This is a **simulation policy**; the instructor's historical Monday-cutoff experiments used their original separately specified cutoffs and must not be silently relabeled as gap-matched replays.
- `A_i`: simulated **decision time**, at or after `C_i` but **strictly before** `S_i`, when model and thresholds must already be selected and any earlier calibration labels must be mature. For a deployment-matched rehearsal, use `C_i = S_i−4h` (source coverage) and `A_i = S_i−3h` (scheduled 21:00 dataset release), provided that release is actually present. Do **not** allow threshold tuning to wait until `S_i` merely to gain more mature labels.
- `M_i`: historical label observation cutoff for a target week. The default **seven-day maturation allowance** is an existing study convention, *not* a proven collection-completeness watermark. If a candidate week cannot be assessed reliably, omit or mark its target as uncertain; do not auto-impute a negative label.
- `T_fit,i`: simulated model-fit time. **Only earlier samples whose labels are available under the documented as-of policy by `T_fit,i` can influence fitting.**
- `T_cal,i`: threshold/policy selection time, before the report target window; only calibration predictions from fits not trained on the corresponding calibration labels are eligible.
- `R_i`: reporting origin, usually identical to `S_i` for the outcome week.

```text
                  older mature fit labels       mature calibration starts    too recent   report
calendar ──────────────────────────────────────[R−35][R−28][R−21]────[R−14][R−7]────[R]───>
                          OOF fits precede calibration target weeks        A=R−3h  S=R
Each calibration origin's features end at (its target start−4h), not target start.
The report features end at C=R−4h; thresholds/policy are finalized at A=R−3h.
```

**Deployment-matched example, conditional on historical coverage:** for a report target beginning `R` and a **seven-day target duration plus seven-day label-maturation allowance**, calibration target starts at `R−35d`, `R−28d`, and `R−21d` (three weekly origins). Weeks starting `R−14d` and `R−7d` are **not yet mature by simulated decision time `A=R−3h`**, and are excluded from calibration under this policy. In general, a calibration target start `S_cal` is usable only when `S_cal + 7d + allowed_maturation <= A`, with separately checked observation completeness; for 7+7 days, this means `S_cal <= R−14d−3h`, hence the most recent aligned weekly target start is `R−21d`. The calibrated-model predictions for a target-start week `S_cal` use features ending `S_cal−4h`. **This is a prospective simulation policy, not a conclusion that any seven-day buffer guarantees truth.** The model used for that calibration prediction must not have fitted its own target label. A later final refit at `A` can use other mature data available by `A`, but requires historical backtests that mimic its refit and calibration transfer.

**Historical EXP-008/010 distinction:** those executed replays admitted labels mature **by the reporting cutoff** (e.g. `label_as_of <= R`), which is a valid *historical diagnostic*. A new deployment-matched study cannot silently use a calibration week whose maturity timestamp is exactly `R` when the real decision must be made at `R−3h`. Log which clock (`R` vs `A`) governs each reported calibration score.

**Fit-time rule for historical labels:** `label_end + maturation_allowance <= simulated_fit_time` is a *necessary scheduling check*, never sufficient evidence of snapshot truth. If a label uses a late-observed issue/PR/review, its available-row filter must be applied at the appropriate historical as-of time. Direct event tables lack independent ingestion-time observability; note this uncertainty in every report.

**Release-level ambiguity:** constructing all folds from a later consolidated Parquet release with `first_seen_at` cutoffs is a partial as-of replay, not exact reconstruction of deleted historical daily snapshots. Where saved distinct releases exist, test retrospective stability and log changed historical fields/targets. Never claim release-to-release stability from one SHA.

### 18.2 Official test roster alignment: avoid conditioning on the wrong recent period

The official test user roster is published **2026-10-28**, selected from recent cohort activity through the instructor's Oct 28 release; the final target begins **2026-11-12 00:00 Taiwan**. Under the published **Oct 28 20:00** data-coverage assumption, the selection anchor precedes the prediction start by **14 days and 4 hours**. This interval must **not** be confused with the final four-hour feature gap.

Construct and score three explicitly labeled historical populations:

1. **`all_cohort`:** all 9,000 cohort users (coverage and sanity checks; not automatically representative of test).
2. **`eligible_proxy_legacy`:** EXP-006's ≥5 activity-observation records in 90 days **at each historical feature/prediction cutoff**; preserves numerical comparability with historical EXP-006–010 studies.
3. **`eligible_proxy_roster_anchored`:** for historical target start `S`, approximate selection at `S − 14d − 4h`, using the previous 90 days of allowable observed activity *before that selection anchor*, with the same ≥5 record rule and explicit caveat that the instructor's exact roster construction is unknown. Compute the anchor without viewing target-period activity. This better approximates **timing**, not necessarily the instructor's selection algorithm.

At official Oct 28 release, profile `test_input.csv` with the *permitted* historical dataset and compare the exact test-user set to approximations (Jaccard, overlap/coverage, source-count distribution, activity-prevalence covariates **only**, no November labels). Predefine which of the two proxies is the primary development population for final selection based on this comparison **before** looking at independent holdout outcomes. Preserve all historical metrics under their stated original population; never retrospectively overwrite old scores.

### 18.3 Nested, forward-only selection architecture

**Three evidence layers:**

- **Exploratory/development:** the already examined Aug 31/Sep 7/Sep 14 report weeks and other historical observations explicitly used to design features. They are *not independent*. Use for debugging and hypothesis generation, report transparently.
- **Inner rolling selection:** additional chronological folds where available, with (a) fit, (b) mature OOF/validation predictions for hyperparameter/feature choice, (c) separately out-of-fold mature predictions for threshold/policy calibration. A fixed finite search and a held-aside reporting segment reduce (but do not eliminate) adaptive selection bias.
- **Outer forward reports:** only folds whose reporting labels have not influenced the candidate decision for that fold; for every outer origin, the **entire model+feature+threshold selection rule** must be determined from earlier information. Prequential / repeated future-origin evaluation is preferred over random CV. Log individual origin decisions to show the policy was executable in chronological order.

Implement an **inner selection runner** and a **prospective-origin replay runner**. For expensive methods allow fewer origins with explicitly broader uncertainty. The Oct 12 sealed holdout is a *separate one-shot gate*, not a development fold. A sequence of after-the-fact retrospective outer folds still becomes development evidence once repeatedly inspected and used to revise the candidate family.

**Selection chronology test:** for each origin, intentionally replace **that origin’s** reporting labels with random noise and verify that model parameters, feature selection, calibration choices and saved predictions **for that origin** remain unchanged. An origin’s labels may legitimately influence *later* rolling decisions only after they have matured; do not demand invariance for those later origins. Reports alone may change; selection must not. Treat this as a test of code isolation, **not** proof of entire label correctness.

**Start-time alignment:** evaluate the existing Monday-midnight protocol for continuity; separately run the deployment-matched four-hour-gap protocol on later folds when data/label maturity permit. Any score difference across those protocols must be interpreted as both feature-window and calendar-alignment changes, not a pure model effect.

### 18.4 Scoring, aggregation and model-quality diagnostics

Let `TP_k`, `FP_k`, `FN_k` be pooled over the *specified evaluated user-week rows* for label `k` (the same cohort/prediction set for both arms). The course scorer defines:

```text
F1_k = 1                          if TP_k + FP_k + FN_k == 0
     = 2*TP_k/(2*TP_k+FP_k+FN_k) otherwise
macro_F1 = mean(F1_k over six targets)
micro_F1 = 2*sum(TP)/(2*sum(TP)+sum(FP)+sum(FN)) if denominator > 0
          = 1 otherwise   # as coded in official evaluator
combined_score = (macro_F1 + micro_F1)/2
```

**Primary metric:** official combined score across a prespecified collection of reporting weeks, by explicit population, using pooled TP/FP/FN. **Secondary:** six target-level F1s and supports; macro, micro and combined *by week*; delta per week; average precision (only if both classes), ROC-AUC (only if both classes), PR curves, prevalence, predicted-positive fraction, precision, recall, TP/FP/FN/TN, score calibration, Brier/log-loss with clipping, relative training/inference time and memory. AP is diagnostic for ranking, not itself the official score. If a week has no positives and/or no negatives, note undefined ranking metrics and use exact official F1 convention, not `sklearn` default silently.

**Population/reporting rule:** do not compare a mean of per-week F1s with a pooled user-week score as if equal. Publish both with names such as `score_pooled_eligible_proxy_legacy` and `score_mean_weekly_all_cohort`. The headline paired delta uses the *same* reporting origins, users, target labels, and score aggregation for control and challenger. No numeric score from EXP-011+ is comparable to historic EXP-006 unless the population/splits and label protocol match.

**Threshold isolation:** For a pure representation ablation, show (a) a fixed frozen threshold policy where meaningful, and (b) a *separately calibrated* threshold policy applied identically to each arm. Report the two results separately; otherwise a feature may appear harmful solely because the old threshold is miscalibrated. Any re-optimized threshold requires earlier calibration labels and is part of the experimental treatment. Never tune threshold on evaluated report labels.

### 18.5 Feature and model attribution: controlled contrasts

**Additive group ablation:** `M(F0)` vs `M(F0 + G)`, same learner/training/splits, only feature group `G` changes. **Removal ablation:** `M(F_selected)` vs `M(F_selected \ G)` with retraining/recalibration; evaluates whether G remains useful alongside correlated groups. **Interaction check:** compare `M(F0+A+B)` against `M(F0+A)`, `M(F0+B)` and `M(F0)` without exploring an unrestricted powerset. **Complexity check:** compare run time and features/variance after deduplication, not just best score.

**Model-family evaluation:** `M1(F_fixed)` vs `M2(F_fixed)` isolates estimator; only after the feature subset is held constant. **Full-system evaluation:** `Policy_A` vs `Policy_B` compares the entire training/calibration/inference rule, and must be identified as such—not described as an isolated feature/learner advantage.

**No universal automatic importance filter.** Tree split-gain or SHAP explanations do not establish predictive contribution, and individual-column permutation importance can mislead when features correlate. Use group-level removal, optional grouped permutation on *unused reporting data*, and repeated stability checks. Investigate high-cardinality ID leakage and categorical surrogate keys rather than celebrating apparent raw-ID importance.

### 18.6 Predeclared promotion/rejection decision rule

Use a **hierarchical gate**, with numeric heuristics explicitly distinguished from scientific confidence:

1. **Validity gate (mandatory, pass/fail):** exact official scorer parity; legal source columns; time cutoff/late-field tests; deterministic input alignment; no holdout reads; fold plan and transformation fit chronology verified. Failure blocks promotion irrespective of score.
2. **Information gate:** feature differs meaningfully from F0; sufficient nonmissing cohort coverage; no accidentally future-derived column; nontrivial user- or calendar-level variance; AP/Brier or confidence of decision boundaries improves for at least the target it was designed to help. AP-only improvement can justify **further investigation**, not final promotion.
3. **Paired score gate (exploratory heuristic):** candidate median per-fold combined delta `>= +0.005` **and** nonnegative paired delta in at least `ceil(0.70*n_reporting_folds)` folds, **with no material collapse** in another target. For system-level release promotion, seek `>= +0.010` pooled combined delta or a strongly supported target-specific net gain, evaluated under prospective-origin policy. These numbers are *project screening tolerances*, not guaranteed statistically significant effects. With only 3 folds, label evidence weak regardless of sign.
4. **Robustness gate:** inspect direct-event targets separately, rare-target F1, per-week prevalence, alternate maturation buffers where source data support them, and anchored vs legacy eligibility. If the benefit exists only on retrospective/proxy labels likely to change, mark `inconclusive`.
5. **Cost gate:** feature creation/fit/rebuild fits the Nov 11 critical path and documented hardware. Reject nominal deltas smaller than reasonable uncertainty if implementation substantially increases failure risk.
6. **Final holdout gate:** one-shot evaluation after maturity with frozen candidate and frozen decision rule; if disappointing, report it as failure and retain safe fallback. If redesign uses holdout findings, reclassify holdout as development and do not claim independence for the redesigned system.

**Uncertainty reporting:** show raw deltas, number of origins, median/IQR and sign count; calendar-week block bootstrap or confidence intervals only with enough quasi-independent blocks and explicit dependence caveats. Avoid a naive `n=16,982` iid t-test or large-sample p-values from repeated observations. Record all attempted feature/model candidates and failed trials; do not cherry-pick positive runs. Avoid claiming a universally valid p-value after many adaptive comparisons.

### 18.7 Mandatory negative controls and implementation properties

- **Future injection:** add synthetic event/entity rows with times `>= C_i`; features at `C_i` must remain unchanged. If fields are late-known, inject future status updates into already-visible records; cutoff-safe features must not change.
- **Label isolation:** randomize reporting labels *after* training/calibration boundary; fitted objects, selection, thresholds and predictions must remain identical.
- **Join multiplicity:** compare distinct primary-key event counts before and after each multi-hop join; reject silent one-to-many amplification. Check self-exclusion for neighbor features.
- **Cross-release ID:** change row numbers while preserving stable GitHub IDs and relational semantics in a synthetic snapshot; stable-ID output remains identical.
- **Sparse users:** zero-history user, one-history event, all-one-class training label, all-negative reporting target, null optional FKs, absent project/organization: predictable finite outputs and correct support flags.
- **Calendar boundaries:** exact `< cutoff`, half-open `[start,end)` target, daylight/timezone handling (Taiwan UTC+08), and four-hour observation gap.
- **Outcome maturity:** compare label construction at saved as-of snapshots if available; late-observed issues must never be backfilled into earlier simulated label snapshots without a declared policy.
- **Feature provenance:** each generated column traces to source tables/fields, transformation code, manifest ID and historical visibility rule.
- **Data source embargo:** all input loaders restricted to eight authorized Parquet tables and instructor-supplied evaluation files for scoring only; no network enrichment.

### 18.8 Reproducible research package standard

Each EXP-011+ completed run must persist at least:

```text
experiments/exp0NN_descriptive_name/runs/<immutable_run_id>/
  manifest.json                    # hashes: data, code, config, feature, splits, dependency versions
  run_config.json                  # hypothesis, treatment/control, cutoffs, population, seeds
  data_coverage.json               # source min/max visible times, row counts, late-label audit
  feature_manifest.json            # F0 and new columns + formula/provenance/coverage
  folds.parquet                    # origin, cutoffs, fit windows, calibration, embargo, roster proxy
  predictions.parquet              # user stable ID, origin, target, y_true (report only), p, yhat
  threshold_trace.json             # earlier-data choice, candidate grid, joint metric
  metrics_by_origin.csv            # each target and official macro/micro/combined
  metrics_pooled.json              # same rows for all compared systems; support
  ablation_summary.csv             # paired deltas, n folds, decision, runtime/memory
  findings.md                      # measured observations, disconfirming evidence, limitations
  tests-and-validation.txt         # commands, actual outcomes, not just planned tests
  figures/                         # generated solely from archived results, optional
```

Use optional file names when a stage lacks some component, and record why. No `results.json` based only on code inspection. **Read-back and reproduce selected metrics** from saved predictions using the exact instructor evaluator. Preserve model artifacts with type/target names, deterministic seeds and environment metadata. Use `UserID` as stable `github_user_id`, never export internal row numbers.

---

## 19. EXP-011 — Controlled stronger tabular learners on frozen F0

**Status:** **Completed** (bounded core executed October 16 and independently accepted October 17). **Observed result:** retain HGB for EXP-012. All six archival alternatives and both earlier-selected rolling alternatives fail the registered improvement screen and target guard. Track B pooled legacy-proxy combined F1: **0.405331 HGB / 0.402824 XGBoost depth 3 / 0.398286 CatBoost depth 4**. Both alternatives improve in only one of four report weeks. Optional second-stage tuning and recency sensitivity remain `not_run` within this bounded scope. See [the execution plan](exp/EXP011_EXECUTION_PLAN.md), log **§22.3–§22.4**, and [independent verification](../reports/EXP011_VERIFICATION.md). The following protocol describes the registered design, not additional completed optional work.

### 19.1 Experimental estimand and controls

- **Treatment:** a change from HGB to a stronger learner, with exactly the same F0 feature matrix, six label definitions, samples and calibration/selection dates.
- **Primary learned comparator:** archived HGB parameters reconstructed and verified through EXP-006/009. Keep logistic as a simple control. **Operational reference:** persistence and prospective replay of EXP-010's calibrator-selected hybrid.
- **Primary scoring population:** same legacy eligible-proxy slice for archival continuity *plus* anchored-roster proxy and all-cohort sensitivity slices. Any new fold/protocol is labeled separately; historical 0.398767 and 0.465391 must not be placed in a new-fold leaderboard as though measured there.
- **Training policy control:** initially mirror the 21-week frozen dataset policy. After selecting an algorithm, run a **separately identified** recent-eight-week sensitivity or a compact 8/12/21-week training-window comparison, not during primary learner isolation. Do not ascribe learner improvements to a different train window.
- **Label/calibration policy:** identical to EXP-006–010 where archival replay is used, and then same prospective-origin schedule on new folds. Do not reuse a frozen model's report probabilities to claim a newly fitted learner passed time-forward fit rules.

### 19.2 Work packages

| Sub-ID | Procedure | Frozen dimensions | Question answered | Required evidence |
|---|---|---|---|---|
| 011-A | Audit/load 59 F0 features, target distributions, missingness, model versions, input hashes | Instructor snapshot, cohort, dates | Are we comparing identical information? | `f0_inventory.csv`, hashes, parity tests |
| 011-B | Refit HGB and logistic controls on same runtime/config | Feature rows, 21-week training, labels | Reproduce known control predictions? | Reproduction delta/tolerance and score trace |
| 011-C | Train LightGBM, CatBoost, XGBoost six one-vs-rest models | F0, fit/cal origins, scoring and seeds | Do alternate learners rank/calibrate better? | Actual model bundles, OOF probabilities, per-target scores |
| 011-D | Small predeclared hyperparameter and class-imbalance study on earlier inner folds | Same learner-family candidate set | Is performance robust to mild model complexity? | Search table and nonwinner trials |
| 011-E | Prospective-origin threshold transfer (archived policy versus independently tuned joint thresholds) | Candidate model probabilities | Are ranking gains usable as official F1? | Matched selected thresholds, report binaries |
| 011-F | Recent training-policy sensitivity after learner selection | Learner, feature set, report folds | Does recency interact with learner family? | Separate treatment label and paired delta |

**Implementation:** Do not assume optional packages exist; inspect pinned environment, add dependencies with explicit version hashes and document licenses/compatibility. HGB remains the always-runnable no-new-package fallback. Models receive numeric F0 features under semantically equivalent missingness rules. If CatBoost uses special native missing/categorical handling while others use imputed numeric columns, the result compares *pipelines*, not pure algorithms; report two arms if necessary. Never introduce raw usernames, late-known entity fields, external training features, user identifier memorization, or external activity data as model inputs.

### 19.3 Bounded search design (candidate matrix is a plan, not fitted outcomes)

**Stage 1:** use deterministic family defaults or two predeclared low-complexity variants per learner; allow no more than **six distinct family/configuration treatments** beyond fixed controls on the initial screen. **Stage 2:** only for the top one or two families, allocate up to four additional configurations each on earlier inner folds. Track total attempted configurations, not just the winner. Default exploration example, adapt only if a config file records the change before seeing report labels:

| Family | Candidate dimensions | Constrained intent |
|---|---|---|
| LightGBM | 31 vs 15 leaves; learning rate `0.05` vs `0.03`; min child samples 30–100; early stopping on an earlier time-ordered slice | Limit capacity vs fitting signal; no random split for early stopping |
| CatBoost | Depth 4 vs 6; learning rate `0.03` vs `0.06`; iterations capped with earlier-fold early stopping; loss `Logloss` | Handle skew and feature interactions with bounded trees |
| XGBoost | Max depth 3 vs 5; min child weight 3–10; learning rate `0.03` vs `0.06`; subsample/column subsample fixed initially | Regularized alternative to HGB |

Values above are engineering starting points, not claims of optimality. To prevent combinatorial growth, encode **full named candidate configurations**, not a Cartesian product of all shown dimensions. Fix CPU/GPU device and thread settings per comparator; log runtime. If no package/environment support, mark the arm `blocked/not_run`, not `failed performance`.

**Target imbalance:** binary cross-entropy/log-loss objective first, no forced class reweighting. Weighting/negative downsampling is a **separate ablation** because it changes probability calibration and class prior; if tried, record correction or calibrated threshold consequences. Do not assume `scale_pos_weight` improves F1.

### 19.4 Acceptance tests and decision

- **Preflight:** 59 exact feature definitions and 6 targets; deterministic shared input row ordering; at least one full real-data fit per executed learner; no illegal columns; basic one-class fallback if a historical training slice lacks a positive label.
- **Metrics:** per-target AP, F1, prevalence/prediction prevalence, per-week/pooled official score, threshold selection trace, fit/inference time and memory; evaluate sparse targets explicitly.
- **Fairness:** equal samples/folds, no post-hoc report-week threshold search, independent control reproduction. Read back saved probabilities and reproduce all metrics.
- **Promotion:** a learner wins as *new fixed-feature screening control* only if it passes §18.6 and is not materially less reliable/cost-effective than HGB. It can remain as a target-specific candidate even when all-six score loses; do not alter the EXP-006 historical baseline archive.
- **Failure criterion:** if no learner exceeds HGB after equally valid calibration or only improves AP while reducing official F1, document the negative result and proceed with HGB for feature attribution. Do not keep tuning indefinitely.

**Required log:** per-arm executable commands, versions, manifest SHAs, fold matrix, full candidate list including nonwinners, threshold policy, paired scoring table, measured decision. **Exit:** an empirically justified fixed learner for EXP-012 (HGB remains acceptable), not necessarily a higher F1 than persistence.

---

## 20. EXP-012 — Temporal behavior, weekly state, momentum and cross-target features

**Status:** **Completed for the registered rhythm/state core**, independently accepted October 18. **Observed decision:** retain HGB with F0 for EXP-013. State passes the earlier screen but fails reporting stability (median weekly delta −0.000571434, 2/4 nonnegative origins); rhythm fails earlier eligibility. Union/removal work is inapplicable under the registered gate; expanded windows, momentum, cross-target and regime groups remain deferred follow-ups. See log **§23.3–23.4**, [the execution plan](exp/EXP012_EXECUTION_PLAN.md), and [independent verification](../reports/EXP012_VERIFICATION.md). The following broader protocol remains a catalog of prospective follow-ups. **Primary question:** which time-based representations improve next-week predictions beyond the existing 59 columns, previous-week persistence, and the currently best fixed-feature tabular learner? **Core hypothesis:** personalization of inactivity, weekly state changes and transitions can discriminate users with similar recent counts but different recurring behavior. **Feature source:** only the authorized Parquet tables. **Definitions:** `FEATURE_ENGINEERING.md` G01–G05 and G12, whose formulas govern implementation.

**October 17 implementation update:** [EXP012_EXECUTION_PLAN.md](exp/EXP012_EXECUTION_PLAN.md) defines a bounded first execution of rhythm and weekly-state groups with fixed HGB, earlier feature-policy selection, paired archival/four-hour-gap reporting, conditional union/removal and independent verification. The user authorized Luna Max subagents as main implementers and the primary agent as orchestrator/evaluator. Broader groups below remain follow-ups; completion requires actual execution and independent acceptance.

### 20.1 Mandatory F0 overlap inventory (before adding any column)

Produce a table for **each of the actual 59 F0 columns**: source table(s), source field(s), formula, visibility filter, windows, actor mapping, feature type, missingness policy, presence in catalog ID, and comparison status `exact_duplicate`, `near_duplicate`, `new`, or `replacement`. Check actual `src/features.py`, `src/dataset_builder.py`, configs and saved EXP-005 manifest; do not infer F0 definitions from the research report. Identify predictors using historical first-observed entity counts versus true event counts; do not silently change established F0 transformations and attribute the resulting difference to a newly added feature group.

**Temporal history discipline:** For `open_pr`, `open_issue`, `pr_review`, first-observation series must have names such as `pr_first_seen_count`, not `actual_pr_open_count`, absent a proven per-field as-of-safe opening-event reconstruction. The official target label generator and historical feature-count generator are different components. Preserve missing/sparse-support indicators.

### 20.2 Staged feature-group study, not one indiscriminate feature dump

| Sub-ID | Feature treatment | Catalog reference / illustrative columns | Predeclared hypothesis | Dependency / unique risk |
|---|---|---|---|---|
| 012-A | **`F-TEMP-BASE`** missing/expanded short horizons | T01–T12; available 1/3/7/14/30/90d counts/flags, active-day counts | Short-window information helps beyond F0 | Explicit F0 duplicates and feature cutoff |
| 012-B | **`F-TEMP-RHYTHM`** | T13/T15/T19/T22/T27: last-event age, median inter-event gap, normalized inactivity, current inactive run, active-day fraction | User-relative inactivity beats absolute recency | Sparse gaps, event vs first-seen semantics |
| 012-C | **`F-TEMP-STATE`** | T29–T34/T39–T41: eight previous week counts/flags, streaks, smoothed transition probabilities | Predictive state recurrence is not captured by a last-week flag | Smoothing fit only on prior mature data |
| 012-D | **`F-TEMP-MOMENTUM`** | T46–T49/T52: 7v21 rate ratio, rate difference, short/long ratio, decayed counts, weekday share | Acceleration/decay helps across regime shift | Ratios/zero denominators, time zones |
| 012-E | **`F-CROSS`** | X01–X05, X10, X13: selected co-presence, push-to-PR proxy, review/reply ratio, latest activity | Cross-activity workflow state adds target-specific signal | Avoid interpreting sequence as causality; censor right edge |
| 012-F | **`F-REGIME`** | C01–C04/C07: contemporaneous cohort rates and relative activity quantiles | Personal propensity is better interpreted against changing cohort baseline | Dynamic population, global preprocessing fit chronology |
| 012-G | **Selected union** | Only candidate groups passing prior screens | Complementary groups combine positively | Redundancy, collinearity, compute cost |
| 012-H | **Removal and interaction** | Drop one winning group at a time; optionally two interacting groups | Is a group still needed after other groups enter? | Refit and recalibrate after each removal |

Do **not** claim catalog template count = generated dimension. For example, a six-target × six-window count template produces up to 36 columns; a 10–25 *new-column* per-screen budget is an engineering heuristic, not a requirement to omit a justified group. Document the instantiated column count and effective number of nonredundant columns.

### 20.3 Exact feature construction and handling rules

- **Counts and flags:** for source activity `a`, observed user `u`, cutoff `c`, `n_a(W)=count(E_a(u) intersect [c−W,c))`; `b_a(W)=1[n_a(W)>0]`. Observed events are deduplicated using within-release source primary keys (or stable IDs where justified), not duplicate multi-hop rows.
- **Days since activity:** `age_a(c)=c−max(t_obs(event))`; undefined for no observed event, accompanied by `has_history_a=0`. Do not assign `age=0` to never-observed users.
- **Weekly panel:** `w_{a,j}` and `z_{a,j}` represent fully elapsed historical seven-day blocks ending strictly before feature cutoff. Do not contaminate with current partial target week; choose alignment to actual target start and feature cutoff consistently.
- **Personal inter-event gaps:** compute on ordered *distinct active days* for event-gap/day features or explicitly event timestamps for event-level gaps. Define units, ties and sample support, with median and CV only above prespecified support. If median gap is zero, use a declared scale/epsilon and coverage flag.
- **Smoothed recurrence:** using historical completed **observed-activity** weeks, `p_on|on=(N11+alpha*p_cohort)/(N1*+alpha)` and `p_on|off=(N01+alpha*p_cohort)/(N0*+alpha)`. Fix `alpha` based on earlier fitting data; no future cohort prevalence or reporting labels. Where a feature instead uses supervised target labels, ensure target history is mature by simulated fit time and that *predicted* week is never included.
- **Rates and momentum:** ratios must define zero-denominator behavior; include raw numerator/denominator or support where meaningful. Compare division-stabilized ratios to rate differences to avoid extreme values for sparse users.
- **Decays:** for past observation age `d`, `EW_tau = sum(exp(−d/tau))`; preregister small `tau` set (7/14/30 days), not all arbitrary decay constants. Ensure all events are earlier than `c` and lookback boundary is consistent.
- **Transitions:** an A→B user-level succession is not evidence they occur in the same PR or repo. Use right-censoring at cutoff and require full allowed follow-up when estimating rates; report support counts. Only build same-PR/same-repo features if link and time visibility rules permit.
- **Cohort context:** leave-one-user-out denominator where own contribution might dominate a local comparator; source population includes 9k cohort with observed activities, not all GitHub users; calculate at each cutoff.

### 20.4 Experimental contrasts and run order

1. **Do not change models while screening a feature group.** Use the fixed HGB control and, if EXP-011 established a clear superior/faster learner, optionally repeat on that *frozen* stronger learner as a prespecified replication. Label learner replication separately.
2. **Single group vs F0:** `F0` vs `F0+G`, for 012-A…F, same chronology and outer reports. No all-six target automatic assumption; capture which targets each feature ought to help.
3. **Best two-group unions:** `F0+G1+G2` only for a small prespecified set of demonstrated promising interactions (e.g. rhythm+state, state+cross). No all-pairs Cartesian expansion.
4. **Removal:** train the selected union, remove one group at a time and refit. If a group improves alone but not in union, classify **redundant/conditional**, not universally useless.
5. **Timing sensitivity:** legacy cutoff and deployment-gap protocol analyzed separately; distinguish gain due to feature count/definition from gain due to newer historical weeks.
6. **Population sensitivity:** all cohort, legacy eligibility and anchored eligibility; maintain exact same rows for treatment/control in every paired calculation.
7. **Simple baselines:** include a calibrated smoothed recurrence/persistence score as a model-free reference where feasible; do not assume an HGB feature alone ought to beat previous-week heuristic.
8. **Negative controls:** shuffle new feature group across users *within earlier training weeks only* to check whether score remains unchanged; use next-week leakage canary. Never use scrambled report labels to guide selection.

### 20.5 Promotion, interpretation, and deliverables

**Minimum completion:** F0 audit completed; at least a low-cost temporal-rhythm and weekly-state family executed on actual authorized data with cutoff tests; all attempted group screens logged; per-target and paired-origin score table; one group-removal test for any selected union; decision on whether temporal features deserve expansion.

**Promote:** only by §18.6 with explicit delta and stability; a higher average precision without prospective F1 is `investigate threshold transfer`, not automatically `promote`. Sparse rare-target estimates with fewer effective examples should receive `inconclusive` and support counts. **Reject:** feature group adds no paired gain or becomes redundant after removal, or it violates field timing. **Blocked:** no as-of-safe definition for an attractive feature (e.g. original PR opening counts known only retrospectively).

**Required artifacts:** actual F0 mapping, per-template manifest (`feature_id`, formula/source/time), cutoff-safety tests, train-only preprocessing trace, ablation result matrix with feature counts, report-week rankings, runtime/coverage diagnostics, and evidence-based decision. **Exit:** a frozen set of justified temporal features, even if that set is empty.

---

## 21. EXP-013 — Relational repository, neighborhood, workflow, and graph-feature ablations

**Status:** **Ongoing — implementation started October 18**. **Primary question:** is there *incremental* prediction value in the eight-table relational structure beyond a strong own-history temporal model? **Feature specification:** `FEATURE_ENGINEERING.md` G06–G11/G13. **Course reporting relevance:** Q2 requires explaining how relationships across tables were enhanced; that requirement is best supported by transparent tested join paths and measured ablations, not merely a schema diagram.

**First-study plan (October 18):** [EXP013_EXECUTION_PLAN.md](exp/EXP013_EXECUTION_PLAN.md) bounds the initial study to repository-self and conditionally cohort-neighbor features from directly timestamped pushes/comments. It fixes HGB with the accepted 59-column F0 control, declares catalog variants, preserves the existing chronological replay, and specifies preflight, selection, reporting and stop gates. Broader workflow/graph groups below remain the prospective catalog. The user authorized Luna Max subagents as main implementers with primary-agent orchestration and evaluation. Implementation has begun; a machine-readable freeze, actual execution and independent acceptance remain required before completion. See log §24.

### 21.1 Control baseline and causal interpretation boundary

The control is **the selected fixed temporal feature set plus the same fixed learner** from EXP-012. Also run `F0+G` where useful to distinguish a relational signal that is only useful without temporal context. **Never interpret a relational feature gain as a causal effect.** Users with more activity naturally touch more repositories and collaborators; raw repo breadth is strongly confounded by count and recency. Include volume, time since activity, active-week count and diversity controls when testing relational increment. As applicable, stratify descriptive comparisons by prior push-count bucket; the definitive test is out-of-time *model* ablation on matched rows.

### 21.2 Research substudies

| Sub-ID | Group | Minimum first implementation | Target hypotheses | Primary risk | Promotion prerequisite |
|---|---|---|---|---|---|
| 013-A | **`F-REPO-SELF`** | R01–R06, R12; distinct repos, max share, entropy, returning/new repo fraction | Push, PR opening, other developer participation | Pure activity-volume surrogate | Own-volume-controlled paired lift |
| 013-B | **`F-REPO-NEIGHBOR`** | R15–R17/R19–R21/R25: excluding-self recent neighbor activity, distinct active cohort users, repo momentum | Push/review/comment opportunities | Incomplete noncohort observations, shared repo dominance | Safe A plus target-specific positive evidence |
| 013-C | **`F-PR-WORKFLOW`** | P01–P03/P05–P06/P12: branch/tag/new-ref push behavior; remaining PR fields conditional | `open_pr`, `push`, `issue_comment` | PR author/creation field availability, ID namespace | Specific as-of field provenance |
| 013-D | **`F-ISSUE-WORKFLOW`** | I01/I04–I07/I10: visible issue/PR-conversation distinctions, observed involvement | `open_issue`, `issue_comment` | Late-discovered issue creation; status fields | Event vs discovery time audit |
| 013-E | **`F-REVIEW-COLLAB`** | V01–V05/V08/V12/V14: repeat reviews, pair frequency, linked review comments/replies | `pr_review`, `pr_review_comment`, `issue_comment` | FK missingness, delayed PR author attribution | Pair coverage and unique event counts |
| 013-F | **`F-GRAPH-2HOP`** | G01/G02/G05/G08 (later only): typed bounded two-hop cohort counts/trends | Same targets with neighbor exposure | Graph sparsity, fan-out, indirect self paths | B/E feature gain and compute gate |
| 013-G | **`F-META-AUTO`** | M01/M03/M06 lightweight text/status or M08–M09 bounded DFS | Metadata beyond activity counts | Text truncation; missing/late fields; unsafe joins | Earlier groups insufficient; provenance verified |

**Start with cohort-safe repo-self and neighbor aggregates**, then narrowly selected workflows. PR/issue/review features should be target-specific if their genuine event/field availability is uncertain. Do not build broad free-form multi-hop text embeddings before the Q2 relational contribution can be established with basic joins.

### 21.3 Valid relational joins, atomic events, and no false completeness

- **Within one snapshot:** join `Users.user_id` to event actor FKs; `Repositories.repo_id` to event `repo_id`; `PRReviews.pr_id` and review comments to `PullRequests.pr_id`; Issues/IssueComments to PR conversations as documented. A joined PR can have many reviews/comments, so group by the *original event primary key* before counting user events. Explicitly assert join cardinalities and audit missing FK references.
- **Across snapshots:** internal row indices can be rewritten each release. Remap by `github_user_id`, `github_repo_id`, `github_pr_id`, and other stable source IDs only when that entity has a stable original-ID field. Never persist a raw `user_id` or `repo_id` as cross-release identity.
- **Cross-repository PR detection:** `PullRequests.head_repo_id` is source GitHub repo ID, whereas `PullRequests.repo_id` is a row-index FK. The supplied `is_cross_repo` derivation compares incompatible representations; audit the field and, if justified, recompute `head_repo_id != Repositories.github_repo_id` through the correct repo join. Do not silently trust unvalidated `is_cross_repo`.
- **Temporal FK target availability:** a source event visible at `c` can reference an entity first visible later than `c` in a joined table. Never pull future entity metadata through a historical join; require the joined target row itself visible at `c`. If an FK's row is unavailable as-of, retain event-level information that is independently visible but treat related attributes as missing.
- **Field-level author/metadata time:** `PullRequests.author_id`, `title`, `draft`, `created_at` may have been populated through later observation; `first_seen_at < c` alone is not proof every such field was known then. Exclude ambiguous attributes, require an independently verifiable first-availability event, or label an explicitly sensitivity-tested weak-as-of proxy. The same caveat applies to mutable issue state and review status.
- **Late lifecycle:** for `merged_at`/`closed_at`/`dismissed_at`, use only when the event timestamp is < `c` and the entity was already observable, subject to schema assurances; for `merged_by_id` require corresponding visible merge evidence. Avoid `Issues.state_reason` unless historical value versioning can be proven.
- **Cohort graph:** when computing neighbor features, use only complete *collected* cohort event streams and explicitly exclude focal user `u` from the numerator and relevant denominator. A referenced non-cohort user's identity does not imply their full activities were collected.
- **Edge semantics:** PR reviewer–author pairing requires both actor identities genuinely available at `c`. Repeated interaction counts aggregate verified observations, not presumed friendship/collaboration. For graph aggregates containing the same source event along multiple paths, deduplicate appropriately and expose path support.
- **Windowing:** define repo set `R_u(30)` using only user interactions before `c`, then neighbor activity *also before `c`*. Do not construct an all-time graph containing edges from after the cutoff, even if the final numeric aggregate is time-filtered afterward.

### 21.4 Required controls and falsification

1. **Repository breadth residual:** compare `F0+own count+recency` vs `+distinct repos/entropy` over identical users and folds; optionally stratify within push-count buckets. A rising conditional future-push rate across repository counts alone is insufficient.
2. **Self-exclusion check:** neighbor metrics with focal user's activity artificially inflated must remain unchanged, except where the experiment intentionally defines a combined all-activity metric.
3. **High-degree repo stress:** cap/sampling decisions must be predetermined and logged; test whether a handful of heavily used repositories dominate the aggregate. Sensitivity test excluding the highest-degree repo is diagnostic only, not arbitrary removal of actual data.
4. **Late entity/incomplete FK stress:** selected PR/issue interactions joining to an entity discovered after cutoff cannot import its title/author/status; test null coverage and that no orphan/null path silently drops all valid event-level evidence.
5. **Event count conservation:** pre/post join event IDs and distinct actors/repositories must reconcile; no extra counts from multiplied review-comment edges.
6. **Popularity confounding:** compare graph signal after conditioning on user's past event volume and repo concentration; no causal claims.
7. **Ablation direction:** measure group-alone increment, increment on top of temporal winner, and group removal in the combined selected union; if graph depth two helps only on previously inspected weeks, classify it inconclusive.

### 21.5 Evaluation, acceptance, and stop criteria

- Run cohort/legacy-proxy/anchored-proxy populations; maintain matched treatment/control rows. Track *coverage strata* separately (no repo, one repo, many repos; PR authorship known/unknown; no observed cohort neighbors).
- Track target-specific ranking, calibrated official F1, macro/micro contributions and error examples, but no exposure of sealed Oct 12 future labels during feature design.
- **Explicit incremental hypothesis:** `score(temporal_control+relational_group) − score(temporal_control)`; don't conflate a new learner and new features in this primary comparison.
- **Promotion:** mandatory as-of/join QA plus §18.6 group gates and target-aligned improvement. Partial groups can be retained for target-specific EXP-014 even if pooled six-target result is neutral, provided the selection policy is validated without hindsight.
- **Stop/skip:** if repo-self has no stable independent signal, neighbor/2-hop work needs a new specific hypothesis or is deferred. Stop text/DFS after a bounded pilot if labels/memory/provenance are uncertain; do not consume Nov 11 critical path for unvalidated graph complexity.

**Exit:** reproducible relational feature aggregates and a tested Q2 story (including negative findings, if applicable), plus a defensible temporal+relational candidate bank for EXP-014. **Artifacts:** join-path provenance, FK/as-of tests, coverage table, before/after counts, large-repo diagnostics, paired ablation/removal tables, single-source run manifests and evidence-based keep/reject decisions.

---

## 22. EXP-014 — Target-specific decision system, ensembles, and joint F1 optimization

**Status:** **Pending**. **Primary question:** can the learned models contribute selected incremental true positives beyond persistence without sacrificing rare-activity precision or becoming miscalibrated under drift? **Starting observation:** the EXP-010 *selected* HGB-push hybrid scored 0.465391 vs persistence 0.454677 on development reports; some unselected variants retrospectively scored higher, but are **not** eligible as independently selected winners.

### 22.1 Freeze candidate pool before policy selection

Only admit learners/features that passed earlier validity gates. Candidate bank per target `a` may include:

1. `PERSIST_a`: fixed previous-week rule or its explicitly tested smoothed counterpart;
2. `TAB_F0_a`: HGB/selected EXP-011 learner using frozen 59 features;
3. `TAB_TEMP_a`: selected temporal subset from EXP-012;
4. `TAB_REL_a`: selected temporal+relational subset from EXP-013;
5. `BLEND_a`: a **small, predeclared** convex blend of one learned probability and persistence (`persistence` is a binary score, not calibrated probability unless separately modeled); or AND/OR combination if supported by an earlier specific hypothesis.

A candidate need **not** exist for every target. Avoid training all model×feature×ensemble combinations. Record complete target-wise candidate matrix and rejection reasons before calibrating on the reporting origins.

### 22.2 Correct six-target joint objective

For a vector of six decisions `ŷ(u)=(ŷ_1,…,ŷ_6)` the objective is the official:

\[
J(\theta)=\tfrac12\left(\tfrac16\sum_{k=1}^6 F1_k(\theta_k) + F1_{micro}(\theta_1,\dots,\theta_6)\right)
\]

The six thresholds are **coupled through the micro-F1 component**, so optimizing each target F1 separately is not necessarily optimal for `J`. Do not assume threshold `0.5`, especially after weighting or when sparse targets dominate the macro term. Use exact official TP/FP/FN arithmetic and its empty/empty convention.

**Candidate policy space:** target-wise hard switches between persistence and a selected learned classifier; a *limited* convex blend (`w ∈ {0.25,0.50,0.75}`) for one or two promising targets; per-target threshold grid that includes earlier known candidates plus score quantiles. Enumerate candidate thresholds **only on out-of-fold calibration predictions**, not inspected reporting-week labels. Use deterministic coordinate descent with multiple fixed starting vectors (including persistence-like/no-positive where meaningful), bounded passes, explicit tie-breaking and archived search traces. A search algorithm that always improves calibration objective does **not** prove future transfer.

### 22.3 Calibrator design and deployment transfer

**OOF requirement:** calibration `p` for week `W` must come from a model fit at a simulated earlier time without W's labels. Once weekly OOF calibrations are ready, choose model/ensemble/thresholds using earlier **mature** windows only. If the final model is subsequently refit on a larger history, the threshold policy must be validated for that refit pattern via analogous historical backtests; don't assume old thresholds transfer unchanged to a new score distribution.

**Two time-safe candidate policies to compare** (under matched reporting weeks):

- **Policy P1, frozen fit:** fit each target model on a prescribed mature training history; predict subsequent calibration and reporting weeks without fitting again; select thresholds at simulated report decision time from matured earlier calibration labels.
- **Policy P2, rolling refit:** refit on the allowed mature history at every reporting origin; obtain calibration prediction distributions through previous rolling *out-of-fold* fits that mimic that refit schedule; calibrate the refit model without using its current report labels.

Choose only after measuring both prospective performance and feasibility. A single historical calibration table generated by a model that was fitted using the very same calibration labels is **invalid** for this comparison. All `T_fit`, `T_cal`, as-of and included weeks must be serialized.

**Optional probability calibration:** Platt/sigmoid or isotonic calibration is itself a model trained on earlier held-out probabilities and labels, and isotonic may overfit sparse classes. Compare uncalibrated probabilities + tuned decision thresholds versus a simple calibration stage, using earlier inner folds. Do not insert calibration by default.

### 22.4 Paired policy experiments

| Sub-ID | Control | Treatment | Main concern |
|---|---|---|---|
| 014-A | Persistence all six | F0-trained model on one target, persistence on five | Added TP vs FP/FN tradeoff |
| 014-B | EXP-010 prospective hybrid recreation | Better selected push model/features, same other five | Verify actual incremental value over controlled selected policy |
| 014-C | Single learner all six | Target-wise model and feature subset chosen from earlier folds | High-dimensional policy overfitting |
| 014-D | Fixed target policy | Joint threshold selection | Does combined objective transfer? |
| 014-E | Selected hard switches | Small selected AND/OR/blend candidate set | Does complementarity survive time shift? |
| 014-F | Frozen fit | Rolling refit with true OOF calibration | Data drift vs threshold instability |

**Error analysis:** for every target, list new TPs, rescued FNs, lost TPs, extra FPs relative to persistence on *report data for analysis after predictions are fixed*. Repeated user behavior and shared PR/repo events must not cause duplicate scored rows. Evaluate predicted-positive-rate drift, probability distribution quantiles and target prevalence across origins.

### 22.5 Final decision, fallback and acceptance

- A **single reproducible policy object** specifies the six sources (`persistence`, `model`, `blend`), feature-manifest version per source, preprocessing, fit window/recency, model artifacts, target ordering, threshold vector, calibration origin policy, deterministic seed/ties, and snapshot compatibility rule.
- **Operational guardrail:** both the selected challenger and a simple fallback can generate a valid final CSV on a fresh permitted snapshot; no manual notebook-only inference step.
- **Go criterion:** valid prospective-origin replay, observed paired gain following §18.6, stable enough policy selection, sensible predicted-positive rates, documented uncertainty, and no breached resource/security restriction. An improvement only on previously inspected reporting labels is not a go.
- **No-go criterion:** learned candidates do not show transfer reliability or final-day execution risk is high; freeze persistence/verified simple hybrid as the official output choice. Negative results are scientifically valid.
- **Completion:** selected policy *and fallback* can be rebuilt end-to-end, archived predictions reproduce official metrics, independent holdout plan is frozen, and exact final-output format check passes in a rehearsal with authorized user IDs/synthetic roster without November labels.

**Outputs:** target selection matrix, joint-objective threshold traces, prequential reporting predictions, error-overlap matrix, policy JSON, trained artifacts, inference entry point, full log and figure-ready measured tables for presentation Q4.

---

## 23. EXP-015 — Conditional relational deep learning feasibility and comparison

> **Historical conditional design:** D9 later admitted a bounded pilot; it ran and finished negative (log §28.6; plan §29.7). Do not schedule this older conditional branch or an October 26 no-go memo as unfinished work. Further GNN tuning is deferred under §30.

**Status:** **Deferred, conditional research branch (not a required assignment prerequisite)**. The course permits GNN/RDL, but no execution or advantage over a strong tabular relational system is established. The reference is the *best validated EXP-013/014 tabular relational policy*, not EXP-006's weaker HGB alone. Initiating this before a stable, valid deployable fallback threatens the Nov 11 deadline.

**Amendment (2026-10-18, user decision D9):** a **bounded pilot** is admitted ahead of the gates below. Gate 1 is waived; gates 2–5 still apply. The pilot is scoped and timed in **§29.7**. Its draft preregistration is [EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md](exp/EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md). The text below is kept unchanged as the original conditional plan.

### 23.1 Explicit entry gates

Enter only after all of the following are recorded:

1. EXP-013 shows **stable positive value from relational features** under own-activity controls, or an independently reasoned typed relational hypothesis requires learned neighborhood interactions beyond aggregations.
2. EXP-014 (or an acceptable simple fallback) has a **tested final-day inference path**; GNN cannot delay it.
3. GPU/CPU memory, library versions, projected graph-construction time and available development days satisfy a documented compute budget (propose a 1–2-day proof-of-concept cap after gate; do not assume this is universally sufficient).
4. All node/edge types and dynamic attributes have explicit as-of semantics; no full graph is constructed using future rows before splitting by origin.
5. Predeclare minimum tabular comparator, actual folds/population/metric, and a **stop condition**. A visually interesting graph or training loss decline is not an acceptable result by itself.

If any gate fails, record a **dated no-go** and maintain Deferred status; Q2/Q3 can still be fully satisfied through tested temporal/relational features.

### 23.2 Minimal model specification (pilot before architecture search)

- **Nodes:** typed Users, Repositories, PullRequests, Issues, PRReviews (and, where useful, comment/push records as typed event nodes or timestamped edges). **Edges:** exact observed PK/FK relationships; typed, directional and timestamped when meaningfully associated with observable events. No assumed missing edge for noncohort users.
- **As-of graph `G(C)`:** include only nodes and edges whose source event/row and referenced entity were observable by `C`; restrict dynamic attributes to version available by `C`, not modern latest-state fields. Stable IDs across snapshots; internal row-number FKs remain within one snapshot. Ensure no future edges in neighbor sampling.
- **Features:** same historic own-activity F0/selected temporal features at cutoff; relation and time features visible before C; categorical encodings from training only. For event nodes, avoid treating post-event truth as pre-event node attributes.
- **Architecture:** first test shallow **two-layer relation-aware heterogeneous message passing** with fixed relation-specific transformations, small bounded hidden dimension (e.g. 32 or 64), dropout, multi-output/six independent BCE heads. Preserve valid negative examples among 9k cohort users and target-wise prevalence masks. Limit neighbors or aggregate high-degree repos with reproducible sampling and diagnostic coverage.
- **Loss:** masked binary cross-entropy on *historically mature* six-label user-week samples. Optimize neural training loss on inner fit periods, use earlier-fold validation for early stopping, and apply the **same official combined-F1 thresholding protocol** as tabular models. A multitask model may share backbone but must still output all six probabilities/binaries in the exact order.
- **Pilot experiments:** (a) node-only MLP, (b) relation-aware 1/2-layer graph, (c) graph+temporal edges/relative time. Test one relation family at a time; keep a typed-edge ablation. HGT/TGAT/TGN/RelGNN are **follow-on candidates only if** the simple graph passes initial safety and performance gates. No obligation to implement a complex architecture for completeness.

### 23.3 Evaluation, competing explanations and stop rules

- Identical valid feature/target cutoffs, reporting users, label maturity and official scorer as EXP-013/014.
- Compare **strong tabular F0+temporal+relational**, node-only MLP, and shallow typed GNN, with fixed parameter/compute budgets and at least two seeds if time permits. Report model size, preprocessing/inference wall time, memory, and invalid-edge/missing-feature coverage.
- Verify relational gain beyond own activity: drop all graph edges (node-only), drop neighbor relations, and optionally randomize neighbor assignments within **training** times while preserving degree statistics. Randomization cannot introduce report-week labels into training.
- Evidence hierarchy: passing leakage and split tests → additional AP/PR behavior → prospective-origin official F1 → cost/stability; do not promote solely because graph attention weights appear interpretable.
- **Stop** on any validated future-edge leakage, irreconcilable per-cutoff node metadata, inability to reproduce the tabular benchmark, major memory failure under real dataset scale, or no paired evidence of useful added signal within the prespecified pilot budget.
- **Exit when run:** an auditable benchmark plus promote/reject/defer decision in the log. **If not run:** a reasoned `Deferred` or `No-go` record is sufficient; do not invent empirical GNN results.

---

## 24. Independent evidence: sealed October 12 holdout and release-to-release verification

### 24.1 Status and intended use of the Oct 12 holdout

> **Controlling amendment:** user decision D3 (§29.6; log §26.4) downgrades this holdout to a **non-independent sanity check**. The original timing and one-shot hygiene below remain, but references to independent evidence or allowing its score to select/override a champion are superseded by **§30.9**.

**Status:** **Deferred — sealed.** EXP-005 exported **9,000 unlabeled feature rows** for historical cutoff **2026-10-12 00:00 Asia/Taipei**; no holdout labels were exported. Its outcome week ends **Oct 19 00:00 Taiwan**. The configured seven-day maturation allowance ends **Oct 26 00:00 Taiwan**. This **does not certify** completeness, especially for late-discovered issue openings. The first practical post-maturation official update is scheduled **Oct 26 21:00**, assuming it is obtained and covers the required time. Snapshot QA and as-of reconstruction must still pass.

**Important mismatch:** this historical holdout was defined under the legacy Monday-midnight feature/target design, not the proposed deployment-matched `C=S−4h` protocol. Preserve its original design and caveat. Do not reinterpret it as a direct test of the new four-hour-gap predictor unless a *separately named*, predeclared four-hour-cutoff holdout feature set and compatible as-of labels can be built **without** adapting to holdout outcomes.

### 24.2 Lock/unlock protocol

1. **Before opening:** pre-register the one selected candidate policy, model/feature configuration, source-replay procedure, target definitions, maturity/as-of rule, official scorer, target population and a fallback comparator. Record hashes and date of decision in `EXPERIMENT_LOG.md` **without reading holdout outcomes**. Freeze plotting/analysis templates as feasible.
2. **Data gate:** obtain a distinct authorized instructor release after the maturation date, store SHA/date, check event coverage, label-as-of consistency, and compare label stability against any earlier available authorized snapshots. For unreliably observed targets, document an uncertainty/sensitivity analysis instead of silently claiming ground truth. If a full six-target trustworthy label cannot be constructed, make the scope explicit and do not call the result an official-score estimate.
3. **One execution:** materialize holdout labels only after the gate, execute frozen challenger and comparators on identical holdout users, output official pooled score, six target confusions, predicted-positive fraction, AP when defined, and generalization gap. Record **every** score and predeclared subgroup; do not run dozens of alternative thresholds against revealed labels.
4. **Interpret:** a positive result increases confidence but does not guarantee November performance because of label reconstruction and distribution shift. A negative result is evidence to prioritize fallback. If the team changes features/policies based on the holdout, acknowledge loss of independence for that revised system.
5. **Data hygiene:** protect label artifacts from feature constructors and model selectors. Tests verify that reading a holdout-label file is impossible before an explicit evaluation command/gate; no public repo copy of sensitive result artifacts is necessary.

**Alternative when label maturity is disputed:** postpone independent six-target comparison; rely on uninspected later mature development origins where possible and release a fully documented uncertainty statement. Do not invent a better holdout merely by sampling random user-weeks from already inspected blocks.

### 24.3 Distinct-release drift experiments

The completed historical studies reference **one** instructor SHA. As new daily releases become available, preserve at least two distinct authorized snapshots where storage allows, keeping dates/SHAs separate. Compare, keyed by stable GitHub IDs:

- table counts; earliest/latest official visibility times; null/coverage distributions; FK integrity;
- direct-event row additions and any changes to old event records;
- backfilled PR/issue/review entities whose candidate original creation/submission date predates the earlier release;
- historical feature vectors built at a *fixed past cutoff* under old and new releases (differences can reveal changing field availability or retroactive table changes; investigate, do not automatically label drift as a bug);
- historical label positives/negatives and delay profiles under consistent label-as-of policy;
- train-time and report-time class prevalence, OOF score distribution and threshold transfer for real release pairs when sufficiently mature.

Distinct-release checks are a source reliability audit, **not** permission to access data after the Nov 11 official cutoff for the final forecast. Never merge cross-release row-number foreign keys.

---

## 25. Time-boxed execution roadmap and project management gates

> **Superseded for scheduling (Oct 20):** the dates below are the Oct 16 plan; §29.5 is the intermediate Oct 18 plan. The current schedule is **§30.12**. Validity and fallback principles still apply subject to §30's explicit amendments.

**Planning date:** **Oct 16, 2026**, Asia/Taipei. The project has less than four weeks until the **Nov 11 23:55** submission deadline. Dates below are *targets contingent on available compute and student time*, not records of executed work. A fallback reproducible model and deliverables take priority over marginal speculative score gain.

| Date (Taiwan) | Milestone / planned work | Evidence / deliverable | Go/no-go gate |
|---|---|---|---|
| **Oct 16–17** | Sync plan/log; inventory actual F0; pin evaluator/source; instantiate additional fold and roster-anchor protocol; protect holdout | `f0_inventory`, validation manifest, reproducibility regression | No expansion until validity tests pass |
| **Oct 17–20** | EXP-011 compact learner matrix using F0; choose learner for feature ablations | Real fit logs; common-feature scores; runtime | Prefer small champion; HGB valid fallback |
| **Oct 19–25** | EXP-012 groups in order: rhythm/state → momentum/cross → selected union and removal | Paired temporal ablations, feature manifest, tests | Stop unsupported/high-cost groups |
| **Oct 24–31** | EXP-013 repo-self → cohort-neighbor → selectively PR/issue/review | Join QA, target-specific gains, Q2 evidence | Two-hop/DFS only if simple relations justify |
| **Oct 26 earliest** | Oct 12 holdout becomes *eligible to consider* maturity gate; **do not automatically unseal** | Source release check, independent gate memo | One-shot only after selected policy freeze |
| **Oct 28** | Official `test_input.csv` publication; study *input roster* using permitted records only | Roster overlap/coverage, anchored-proxy comparison | Choose intended evaluation population before holdout results |
| **Nov 1–5** | EXP-014 target-specific policies, OOF thresholds, limited blends; compare against persistence and selected hybrid | Champion/fallback policy, actual chronological scores | Freeze once robust and executable |
| **Nov 5–7** | End-to-end dry-run on latest authorized release; optional conditional EXP-015 if all gates met | Build→train→predict→format-check rehearsal | No risky model experiments after fallback freeze |
| **Nov 7–9** | Final prospective-style audit; one-shot historical holdout when mature and policy frozen; draft figures and Q1–Q4 presentation | Holdout report, figures from real results, draft slides/video plan | If failure, select documented fallback; do not retune holdout |
| **Nov 9–10** | Finalize public reproducibility code, README, slide/report and recording; reserve compute contingency | Clean checkout test; PDF/MP4/CSV naming check | Code and submission all locally tested |
| **Nov 11 before 21:00** | Prepare last-rebuild script, fallback artifacts, environment and submission files | Preflight checklist and tested commands | No new library/architecture installations |
| **Nov 11, 21:00–23:55** | Re-clone final instructor release, record SHA, rebuild/infer, validate exact test-user set and scorer, submit | Final `output_{studentID}.csv`, source commit, slides and video | **Submission-first**; use fallback if new pipeline fails |

**Parallel work:** Q1 schema/entity graphs and reproducible presentation can be prepared using existing files without waiting for final models. Q2/Q3 result figures must be regenerated from **actually executed** ablations; mark expected figures as placeholders until measured. Code must be pushed to the public repository **before** the deadline; don't postpone recording/video until the final dataset release.

### 25.1 Resource budget and stopping policy

A proposed engineering resource cap (adapt to actual hardware; do not call it literature-backed):

- **Feature screens:** initially 10–25 nonduplicate additional columns per group and at most six temporal group trials plus a handful of predeclared group unions; one fixed learner/fit policy for attribution.
- **Model-family search:** 2 quick configurations per family, then up to 4 additional for only the top one or two; record every candidate.
- **Relational cost:** audit cached user–repo–time aggregates before any full multi-hop joins; cap one-feature-group pilot wall time according to hardware and deadline.
- **Threshold/policy search:** small predefined policy pool with deterministic grid and coordinate-descent stops; always log calibration vs report.
- **DL:** low-priority and strictly gated; if not completed by an agreed stop date, write a no-go rather than risk deliverables.
- **When results are inconclusive:** retain safe comparator rather than extend a broad unconstrained search because a single week improved.

### 25.2 Critical-path dependency map

```text
                         EXP-001..010 COMPLETED
                                   |
                        F0 inventory + test/fold QA
                                   |
                       EXP-011 fixed F0 learner benchmark
                                   |
                           fixed learner for ablations
                                   |
                  EXP-012 temporal groups + removal tests
                                   |
                      selected temporal representation
                                   |
                 EXP-013 relational/workflow groups + QA
                                   |
                selected verified target-specific candidates
                                   |
                   EXP-014 joint policy + OOF thresholds
                                   |
                  frozen champion + always-valid fallback
                                   |
                 independent holdout / final dry-run / QA
                                   |
                     NOV 11 FINAL RELEASE + SUBMISSION

      EXP-015 optional branch: only after EXP-013 relational+compute gates;
          MUST NOT block champion/fallback or submission deadline.
```

---

## 26. Engineering interfaces and release-readiness acceptance tests

### 26.1 Planned (not currently verified) module interfaces

Reuse `src/data_loader.py`, `src/labels.py`, `src/features.py`, `src/temporal_splits.py`, `src/dataset_builder.py` and existing experiment runners rather than rewriting a parallel data pipeline. Interfaces below describe **requirements**, not claims about existing public APIs:

| Component | Inputs | Outputs | Mandatory invariants |
|---|---|---|---|
| Release adapter | Eight Parquet paths, instructor SHA | Mapped tables, stable-ID joins, provenance | No external data; FK/PK audit; time naive UTC attached explicitly |
| Temporal sample builder | `(UserID, feature_cutoff, target_start, simulated_fit_time)` | Point-in-time feature panel + versioned six-target labels | One row per stable ID/origin; source row time < cutoff; no future field values |
| Feature registry | Catalog IDs + frozen F0 + source version | Immutable feature manifest and Parquet columns | Formula/window/missingness/visibility for **every** new column |
| Chronological evaluator | Fit/calibration/report origin schedule | As-of checks, OOF predictions, fold metrics | No current or future label in model/selection state |
| Model trainer | Selected feature view + candidate config | Six fitted estimators or multihead model, fit manifest | Train-only preprocessors, deterministic fallback on one-class cases |
| Threshold policy | Earlier matured OOF predictions and labels | Six target decisions, policy trace | Exact official combined-score objective, no report leakage |
| Comparator/reporter | Frozen predicted binaries + labeled report weeks | Per-target and pooled exact scores + statistical diagnostics | Same users/labels for compared treatments |
| Final inference | Final authorized instructor snapshot + official `test_input.csv` + frozen policy | Valid `output_{studentID}.csv` | Exactly test roster; deterministic 0/1; independent official format check |

**Recommended data flow:** separate `raw snapshot` → `canonical time-safe observations` → `cached user-week / user-repo-week aggregates` → `feature views` → `train/cal/report splits` → `OOF probabilities` → `selection/calibration policy` → `frozen inference`. Cache keys include instructor SHA, feature group version, cutoff schedule, and labeling/as-of policy. Protect chronological boundaries at the data access layer, not only in README prose.

### 26.2 Testing matrix, verification and failure recovery

| Test class | Examples | Minimum evidence |
|---|---|---|
| Unit math | moving-window count, decay, Laplace-shrunk activity probability, streaks, entropy, ratio guards | Synthetic exact-value assertions and empty-support cases |
| Temporal leakage | equality at boundary, event after cutoff, late-known attribute updated after cutoff, out-of-order future entity FK | Frozen earlier feature vector unchanged |
| Data-model contract | 8 tables, 20 FK relations, 9k cohort, stable user ID, no raw cross-release FK | Real snapshot audit plus synthetic remapping |
| Label semantics | non-PR issue only, PR author known/null, submitted review vs comment, source-specific event vs visible times | Truth-construction fixtures and official sample regression, with calibrated-sample caveat |
| Split/maturity | no training label from future, embargo, OOF model independence, test roster anchored correctly | Serialized origin table and forbidden-label injection |
| Score parity | random 6-label arrays, all negative, one positive, per-week pooled vs weekly-mean | Exact `evaluate.py` matching numeric assertions |
| Model determinism | same seed/config/data, unchanged preprocessing | Probability tolerances, artifact manifest, warning log |
| Join conservation | multiplicity, self-neighbor exclusion, null FK, high-degree repo | Event ID/count accounting, sparse-coverage audit |
| Feature selection integrity | group present/absent, duplicate removal, only previous folds control selection | Manifests + threshold traces + label permutation isolation |
| Deployment rehearsal | clean clone, build, fit/load, infer, validate CSV header/IDs/type; fallback mode | Real run on latest authorized instructor release and *known* input roster or declared dummy roster |

For every failed test, preserve the first observed traceback, affected input SHA, diagnosis, fix, and rerun result. Do not report “tests pass” after skipped execution, mock-only tests, or an unavailable optional live instructor sample. Existing EXP-008–010 verified regression (**57 passed, 1 skipped**) is a **historical checkpoint**, not a guarantee that new feature code preserves all guarantees.

### 26.3 Final release runbook and rollback

1. Verify public source revision and `README.md` instructions; preserve existing instructor clone and its SHA or back it up before replacing; delete/re-clone only the designated disposable instructor directory. Keep all personal code/artifacts outside it.
2. At/after Nov 11 21:00 Taiwan, download **only the final instructor repo release**, record full instructor commit SHA, file hashes and source coverage, and run row schema + stable-ID consistency checks. Use actual data coverage field/maximum time, not an assumed event at precisely 20:00.
3. Load official test IDs (released Oct 28). Confirm unique IDs, that all belong to cohort, row count and identity equality; identify any missing actor or empty-history user before inference.
4. Rebuild feature matrix at the correct feature observation cutoff (data available through approximately Nov 11 20:00). Freeze target start **Nov 12 00:00**; do **not** include any unobserved four-hour interval or event after the deadline. Apply the preselected preprocessing/model-fit/calibration policy with source-as-of and mature-label safeguards.
5. Run selected six-target policy and fallback policy; never choose based on unknown November labels. If selected policy fails parity or runtime, use documented and previously tested fallback **without** editing thresholds on the fly.
6. Export `output_{studentID}.csv` with exact header, all and only official `UserID` entries, no duplicates, six integer 0/1 columns, stable order. Run the *instructor's actual validator*, e.g. `python assignment2/evaluation/evaluate.py --pred output_{studentID}.csv --input assignment2/evaluation/test_input.csv` from the appropriate project root.
7. Inspect command exit codes and final file; archive checksum, inference manifest, model policy version, release SHA and selected fallback status. Commit/push code to a **public** GitHub repository before Nov 11 23:55; submit CSV, slide PDF and independently recorded MP4 to E3. Instructor PDF has conflicting stems `Asg2_` and `Asg_` for PDF/MP4, so confirm accepted naming from E3/TA ahead of time.
8. **Never report the November evaluation score before truth or official grader feedback exists.** Archive only reproducible predictions and selection rationale.

**Rollback:** on any late unexpected package dependency, release schema mismatch, OOM, model nonconvergence, or inference parity failure, fall back to last known-good validated policy; prioritize deadline. Do not undertake an untested model retraining experiment between 21:00 and 23:55 on submission day.

---

## 27. Experiment preregistration, log templates and decision reporting

### 27.1 Mandatory preregistration card (fill *before* running)

```yaml
experiment_id: EXP-0NN
substudy_id: EXP-0NN-X
status: Pending
research_question: "..."
hypothesis: "..."
null_hypothesis: "..."
primary_outcome: official_combined_f1_pooled
secondary_outcomes: [macro_f1, micro_f1, per_target_f1, average_precision, support]
source:
  instructor_dataset_sha: "..."
  schema_version_or_sha: "..."
  official_evaluator_sha: "..."
  student_code_sha: "..."
  allowed_input_tables: [Users, Repositories, PullRequests, Issues, IssueComments, PRReviews, PRReviewComments, Pushes]
samples:
  target_start_utc: ["..."]
  feature_cutoff_rule: "historical S-4h OR explicitly documented legacy cutoff"
  label_as_of_policy: "..."
  maturity_allowance_days: 7  # scheduling assumption; validate separately
  fit_times: ["..."]
  calibration_times: ["..."]
  reporting_origins: ["..."]
  populations: [all_cohort, eligible_proxy_legacy, eligible_proxy_roster_anchored]
feature_manifest_sha: "..."
control_policy: "..."
treatment_policy: "..."
selection_rule: "earlier mature OOF only"
fixed_hyperparameters: "..."
candidate_budget: "..."
random_seeds: [42]
promotion_gate: "see EXPERIMENT_PLAN.md §18.6"
expected_runtime_budget: "..."
negative_controls: [future_event_injection, label_isolation, output_parity]
artifacts_directory: "experiments/exp0NN_name/runs/<new_run_id>/"
observed_results: null   # populated ONLY after real execution and audit
```

This is a **schema example**, not a command to create new files for every entry. For research comparison, a single machine-readable manifest per run plus a concise Markdown decision record is adequate if it contains the same fields.

### 27.2 Executed experiment log entry template

```markdown
### EXP-0NN-X — concise title — YYYY-MM-DD Asia/Taipei
- **Status:** Completed / Failed / Invalidated / Blocked
- **Question and falsifiable comparison:** ...
- **Source:** instructor SHA, student SHA, full source/config manifest and environment
- **Time:** feature cutoffs, target starts, fit/calibration/report and maturity rules
- **Population:** all cohort / legacy proxy / roster-anchored proxy; counts by fold
- **Treatments / frozen controls:** full features, estimator and threshold policy
- **Executed commands and actual exit codes:** ...
- **Tests:** pass/fail/skip with traces; evaluator parity and leakage controls
- **Observed metrics:** each six-target TP/FP/FN, per-target F1/AP, weekly and pooled official metrics
- **Paired comparison:** same user-week rows, each origin delta, pooled delta, runtime
- **Stability:** across folds, rare-target supports, missingness/coverage, label-as-of caveats
- **Candidate search accounting:** tried/failed configs and reason for selection
- **Interpretation:** what evidence supports; what is unproven; counter-explanations
- **Decision:** promote / continue / reject / inconclusive / defer, with exact criterion
- **Artifacts:** immutable run path, code/data/config hashes, reproduction instructions
- **Next step:** ...
```

**No result, no completion.** A fully specified experiment that has not been executed remains Pending; a code stub, synthetic test suite or attractive graph does not count as experimental evidence. Preserve prior log numbers; append an explicit invalidation/correction if a later defect changes conclusions.

### 27.3 Per-feature-group decision table (must be produced in EXP-012/013)

| Column | Meaning |
|---|---|
| `group_id`, `catalog_template_ids`, `instantiated_columns` | Exact treatment and dimensionality |
| `novelty_vs_F0` | Exact/near duplicate/new/replacement counts |
| `source_and_visibility_audit` | Raw fields, joined tables, as-of checks |
| `training_support`, `reporting_support` | Nonmissing counts and class positives by target |
| `learner_and_fit_policy` | Identical treatment/control unless explicitly pipeline-level |
| `delta_combined_pooled`, `delta_weekly[]` | Paired official-score effect and its variation |
| `target_F1/AP_deltas` | Per-target ranking and classification distinction |
| `train_runtime`, `inference_runtime`, `feature_build_runtime` | Complexity tradeoff |
| `calibration_policy`, `selected_thresholds` | No reporting-label selection |
| `removal_delta`, `interaction_delta` | Redundancy and conditional signal tests |
| `decision`, `reason`, `date`, `run_id` | Reproducible promotion or rejection |

### 27.4 Stop criteria and final research report checklist

The project is ready for final freeze when it can truthfully answer all of the following:

- Was the final six-target modeling choice fixed **without** using November activity or independent holdout outcomes to tune it?
- Can a clean machine/environment reproduce processing of the *instructor's eight tables*, the exact selected feature set, six fitted target predictions and the final CSV from a documented `README.md`?
- Can each positive feature-engineering or learner-performance claim be traced to actual saved OOF/report predictions, code/data hashes, score convention and chronological rows?
- Does Q2 show a real relational mechanism (tables/foreign keys, tested feature-group gains **or null findings**) and Q3 a real temporal mechanism, not just a feature list?
- Does Q1 include the full eight-table PK/FK schema graph and entity graphs for user IDs `22996444`, `28983449` on `2026-09-15 00:00`–`2026-09-17 00:00 UTC`, using only instructor data?
- Are Q4's pipeline and the slides/video aligned with actual implementation, real measurements and the selected deployment policy?
- Have uncertain label maturity, collection regime shifts, ambiguous PR/issue field timing, non-cohort incompleteness and test-roster mismatch been disclosed instead of silently assumed solved?
- Is the fallback explicitly tested and can it produce a valid official CSV inside the Nov 11 release-to-deadline interval?

**The stopping rule is not “all catalog features tried.”** Stop when the next candidate's expected score gain is too small/uncertain relative to validation overfitting and release risk, or when the planned finalization deadline is reached.

---

## 28. Evidence basis, research references, and source hierarchy

### 28.1 Evidence tiers

| Evidence tier | Applies to | Permitted inference |
|---|---|---|
| **Course contract** | Official PDF, schema, `data/README.md`, evaluator | Governs rules, timestamps, source columns, output and metric |
| **Verified prior execution** | EXPERIMENT_LOG §§14–21, saved run outputs and verification reports | Measured results **only** for reported snapshot, protocol, population and folds |
| **Methodological literature** | Below: temporal validation, nested selection, feature importance, F1 | Motivates test design; does not establish performance here |
| **Feature catalog and research report** | `FEATURE_ENGINEERING.md`, supplied Deep Research report | Candidate hypotheses/formulas; no completed experiment implied |
| **Project design decision** | Numeric gates, search/budget caps, planned timetable and architecture shortlist | Adjustable preregistered heuristic; not externally validated statistical fact |

### 28.2 Primary assignment/repository and internal sources

1. Instructor course PDF: `Big Data Course - Assignment 2.pdf`, Oct 14, 2026, §§1–3 (the strict allowed-data rule, chronological split, six labels, submission and 15% assignment grading).
2. [Instructor repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2), [data README](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/README.md), [schema](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/db/SCHEMA.md), [evaluation README](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/README.md), [official scorer](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/evaluate.py). **Fetch a fresh version/record hashes** for real executions; these links are reference documents, not substitutes for the pinned instructor SHA used by EXP-001–010.
3. Existing `EXPERIMENT_LOG.md` (the supplied `EXPERIMENT_LOG(2).md` reviewed in preparing this continuation); §17 audit closure; §18 threshold replay; §19 training-recency; §20 push-target hybrid; §21 independent verification.
4. Existing `EXPERIMENT_PLAN.md` (the supplied `EXPERIMENT_PLAN(2).md`), historical EXP-001–010 evidence and original completion rules, preserved in §§1–15 of this updated plan.
5. `FEATURE_ENGINEERING.md` (supplied finalized catalog, 173 feature templates, exact group IDs/formulas and feature-evaluation cautions); source of candidate **definitions**, not performance evidence.
6. User-provided Deep Research report, `Pasted text(3).txt`, for previously proposed EXP-011–014 and literature; older proposed IDs have been intentionally renumbered to prevent collision with the **actually executed** EXP-009/010 scope.

### 28.3 Methodological references checked for relevance

- **Cerqueira, V., Torgo, L., & Mozetič, I. (2020).** “Evaluating time series forecasting models: an empirical study on performance estimation methods.” *Machine Learning* 109:1997–2028. [DOI 10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7). Evidence for repeated out-of-sample temporal evaluation under nonstationarity; the exact fold design in this document remains a project choice.
- **Cawley, G. C., & Talbot, N. L. C. (2010).** “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.” *JMLR* 11:2079–2107. [Original](https://www.jmlr.org/papers/v11/cawley10a.html). Evidence that repeatedly maximizing a noisy model-selection criterion can bias the apparent winning score.
- **Varma, S., & Simon, R. (2006).** “Bias in error estimation when using cross-validation for model selection.” *BMC Bioinformatics* 7:91. [Original](https://doi.org/10.1186/1471-2105-7-91). Nested selection/evaluation motivation; not a mandate to use random iid CV for time series.
- **Roberts, D. R., et al. (2017).** “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.” *Ecography*. [DOI](https://doi.org/10.1111/ecog.02881). Dependence-aware validation principles relevant to repeated user-weeks and shared repositories.
- **Guyon, I., & Elisseeff, A. (2003).** “An Introduction to Variable and Feature Selection.” *JMLR*. [Original](https://www.jmlr.org/papers/v3/guyon03a.html). Feature-group selection rationale and distinctions between feature relevance, redundancy and classifier-dependent value.
- **Nogueira, S., Sechidis, K., & Brown, G. (2018).** “On the Stability of Feature Selection Algorithms.” *JMLR*. [Original](https://jmlr.org/papers/v18/17-514.html). Motivation to assess stability across folds rather than celebrate one selected subset.
- **scikit-learn User Guide.** [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) and [correlated-feature caveat](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html). Importance is model-relative and individual correlated features can appear unimportant; group removal is the principal decision tool here.
- **Lipton, Z. C., Elkan, C., & Narayanaswamy, B. (2014/2015).** “Thresholding Classifiers to Maximize F1 Score.” [arXiv 1402.1892](https://arxiv.org/abs/1402.1892). Threshold optimization matters; the **assignment's joint combined objective** must be implemented exactly rather than imported unchanged from single-label F1 theory.
- **Saito, T., & Rehmsmeier, M. (2015).** “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.” *PLOS ONE*. [DOI](https://doi.org/10.1371/journal.pone.0118432). Justification for AP and PR diagnostics on rare targets, not as substitutes for official F1.
- **Fey, M., et al. (2024).** “Relational Deep Learning.” *ICML/PMLR*. [Original](https://proceedings.mlr.press/v235/fey24a.html); **Robinson, J., et al. (2024)**, [*RelBench*](https://arxiv.org/abs/2407.20060). Background for the conditional typed temporal GNN/RDL branch; the assignment's dataset may behave differently.

### 28.4 Interpretation limitations and final decision rule

No published result above demonstrates that any catalog feature or temporal GNN will improve **this** six-target assignment's official score. EXP-011's executed fixed-F0 LightGBM/CatBoost/XGBoost screen did not justify replacing HGB; completion is supported by log §22. EXP-012's completed rhythm/state core retains F0 after independent acceptance in log §23; expanded temporal groups remain deferred. EXP-013–015 remain pending or deferred. Historical EXP-006–012 gains and rankings remain valid **for their logged experimental configurations**, not for November. The Oct 12 independent holdout remains sealed, and no claim is made that an arbitrary seven-day maturation period guarantees the true action record.

**Final operating principle:** prefer the simplest demonstrably time-safe, stable and reproducible *complete six-target policy* that is competitive under prospective-origin official scoring, can survive the final instructor release, and can be submitted before the deadline. Optimize reproducible future-week score—not feature count, architecture complexity, or the largest hindsight number on previously inspected development weeks.

---

## 29. Score-maximization backlog and revised timeline (2026-10-18)

> **Historical planning section:** the initial register and cards below are preserved with their dated status addenda. **§30** supersedes the remaining forward schedule and priorities. The screens in log §§26–27 and the completed EXP-015 pilot in log §28.6 have actually run; the original "plan only" statement below describes this section when first written.

**Status of this section:** **Plan only — nothing below has been run.** It was written after reading log §§15–25, `FEATURE_ENGINEERING.md`, the EXP-014a run and [EXP014_VERIFICATION.md](../reports/EXP014_VERIFICATION.md). It reorders the remaining work toward the official combined F1. Where it conflicts with the dates in §25, **§29.5 supersedes §25** for scheduling. The governance rules in §§16.5, 18 and 27 still apply to anything that is promoted.

### 29.1 Why the priorities change

Four recorded findings drive the reordering:

1. **Ranking is not the bottleneck; the decision layer is.** HGB and logistic beat persistence's average precision in **all 18 target-week comparisons** (log §17.4). They lose on F1 because their positive *count* transfers badly across regimes: the EXP-014a Platt layer over-predicted **2.65× / 4.05×** at 09-16 / 09-23 ([EXP014_VERIFICATION.md](../reports/EXP014_VERIFICATION.md) F1).
2. **Label-free top-k on uncalibrated HGB probabilities reached parity with persistence.** This was a hindsight diagnostic, so it is not a selection result: recent3 **0.514** vs **0.508**, all5 **0.483** vs **0.476** (same report, F1).
3. **Feature and learner screens under the old threshold policy showed no gain.** EXP-011 found no learner that beat HGB, and EXP-012 found no temporal group that passed. Both were scored through the threshold-transfer layer that EXP-014a showed is broken. Their negative results say little about ranking value under a working decision layer.
4. **Persistence is weak exactly where macro-F1 is decided.** Recent3 pooled persistence F1 is about **0.40–0.45** on the five non-push targets and **0.67** on push (EXP-014a `per_target_metrics.csv`). Macro-F1 weights each target equally, so the rare targets matter most.

**Consequence:** first fix the decision layer, reusing the saved EXP-014a predictions where possible. Next, improve ranking, selecting on AP rather than F1. Then confirm on fresh October origins. Stop spending time on new learner families, more rhythm/state variants and GNNs.

### 29.2 Two evaluation tiers

| Tier | Origins | Use | Gate |
|---|---|---|---|
| **Screen** | `all4` = Thursday 00:00 Taiwan origins 2026-09-09, 09-16, 09-23, 09-30 (09-30 provisional). 09-02 is flagged as a transition week and never used for decisions | Fast iteration; may reuse `experiments/exp014_recent_regime/runs/exp014-recent-regime-v1/predictions.parquet` | Pooled Δ vs `P_persist7` > 0 **and** ≥ 3/4 weekly wins **and** leave-one-origin-out minimum ≥ −0.005 (same margins as the EXP-014b draft) |
| **Confirm** | Fresh origins 2026-10-07, 10-14, 10-21 as each matures (§29.5) | Only for frozen candidates; no edits after the freeze | Pooled Δ > 0 over the confirm origins **and** no origin Δ < −0.010. Otherwise `P_persist7` stays the default |

All September origins have already been inspected many times, so they are **development** data. Only confirm-tier results count as evidence of future performance. Ranking experiments (EXP-022–025) are screened on **mean per-target AP** and on score under the decision policy frozen from EXP-014b/016–018. They are not screened on F1 under thresholds tuned for them alone. Heavy independent audits (§18.8) are reserved for the final candidate set (EXP-030), not for every screen.

### 29.3 Backlog register

Priority is ordered by expected score value per day of work. "Refit" means new model fits are needed; "No" means the experiment runs on saved predictions or features only.

| # | ID | Experiment | Phase | Refit | Depends on | Status |
|---|---|---|---|---|---|---|
| 1 | **EXP-014b** | Raw-p top-k and same-model out-of-time Platt (existing draft [EXP014B_PREREGISTRATION.md](exp/EXP014B_PREREGISTRATION.md)) | A — decision | C3 only | EXP-014a run | **Pending** (draft, not frozen) |
| 2 | **EXP-016** | Persistence-count top-k: k from persistence, ranking from HGB | A — decision | No | EXP-014a predictions | **Pending** |
| 3 | **EXP-017** | Persistence-anchored edits: add top-m non-persistent users, drop low-p persistent users | A — decision | No | EXP-014a predictions | **Pending** |
| 4 | **EXP-018** | Cohort positive-count forecast for k | A — decision | No | EXP-014a predictions | **Pending** |
| 5 | **EXP-019** | Per-target policy selection across all six targets | A — decision | No | 014b, 016–018 | **Pending** |
| 6 | **EXP-020** | Label-aligned and multi-window persistence rules | B — rules | No (feature recompute) | Source tables | **Pending** |
| 7 | **EXP-021** | Smooth-persistence score and blended ranking | B — rules | No | EXP-016 | **Pending** |
| 8 | **EXP-022** | Training window and recency weighting under daily origins | C — ranking | Yes | Decision policy from A | **Pending** |
| 9 | **EXP-023** | HGB capacity tuning, selected on AP | C — ranking | Yes | EXP-022 | **Pending** |
| 10 | **EXP-024** | Short-horizon features (1d/3d, hours since last event) | C — ranking | Yes | EXP-022 | **Pending** |
| 11 | **EXP-013** | Finish relational ablations; rescore under the new decision layer | C — ranking / Q2 | Yes | Audit blockers in [EXP013_CODE_REVIEW.md](../reports/EXP013_CODE_REVIEW.md) | **Ongoing** |
| 12 | **EXP-025** | Seed bagging and HGB + logistic rank blend | C — ranking | Yes | EXP-023 | **Pending** |
| 13 | **EXP-026** | Official roster alignment (Oct 28 `test_input.csv`) | D — deployment | No | Roster release | **Scheduled** |
| 14 | **EXP-027** | Fresh-origin confirmation harness | D — deployment | Per candidate | New instructor releases | **Pending** |
| 15 | **EXP-028** | Release drift and late-label sensitivity | D — deployment | No | Two or more distinct release SHAs | **Pending** |
| 16 | **EXP-029** | Final-day end-to-end dry run | D — deployment | Yes | Frozen candidate | **Pending** |
| 17 | **EXP-030** | Champion freeze, final refit and fallback package | D — deployment | Yes | 026–029 | **Pending** |

**Explicitly not in the backlog:** further LightGBM/CatBoost/XGBoost sweeps (EXP-011 found no gain), further rhythm/state variants (EXP-012), the EXP-014a M_cal→M_final Platt layer, `F0R` regime columns, 2-hop graph/text/DFS features, and **EXP-015 (GNN)**. For EXP-015, record a dated no-go memo unless EXP-013 shows a stable relational gain by Oct 26.

### 29.4 Experiment cards

#### Phase A — decision layer (highest expected value, mostly no refits)

**1. EXP-014b — raw-p top-k and same-model OOT calibration.** Run the existing draft as written: candidates C1 (expected-score k on raw p), C2 (rate-match k on raw p) and C3 (Platt fitted on M_final⁻'s last-week holdout), all against `P_persist7` on `all4`. C1 and C2 are contaminated by the hindsight diagnostic, so only Stage 2 (confirm tier) can promote them. *Deliverable:* the S1/S2 outcome recorded in the log.

**2. EXP-016 — persistence-count top-k.** For each target t, set k_t to the number of users that `P_persist7` predicts positive, then predict the top k_t users by HGB `p_raw`. Variants: k_t = α · persist count with α ∈ {0.8, 1.0, 1.25}, with α chosen on the two origins before each report origin. *Hypothesis:* persistence's count adapts to the regime automatically, while HGB's better ranking should convert the same budget into more true positives. *Label-free:* yes. *Q3 relevance:* separates "how many" (temporal regime) from "who" (user ranking).

**3. EXP-017 — persistence-anchored edits.** Start from the `P_persist7` positives. (a) Add the top-m non-persistent users by `p_raw`. (b) Drop persistent users whose `p_raw` is below quantile q. (c) Do both. Choose m and q per target from the previous two origins. *Hypothesis:* a conservative edit keeps persistence's robustness and fixes its clearest errors, such as one-off events last week and reactivations.

**4. EXP-018 — cohort count forecast for k.** Forecast next week's positive count per target with: last week's active count; the mean of the last 2 weeks; EWMA (half-life 1 or 2 weeks); and Σ`p_raw` from M_final. Score each method's count error on `all4` first, then plug the best into EXP-016's top-k. *Q3 relevance:* an explicit temporal model of cohort regime.

**5. EXP-019 — per-target policy selection.** For each target, choose among {`P_persist7`, the best of 014b/016/017} using the two most recent mature origins before each report origin. This generalizes EXP-010 (push only) to all six targets. Keep the joint combined-F1 objective, because micro-F1 couples the targets through push volume.

#### Phase B — better rule baselines (cheap, robust fallbacks)

**6. EXP-020 — label-aligned and multi-window persistence.** (a) For `open_pr`, `open_issue` and `pr_review`, count rows whose `created_at`/`submitted_at` falls in the last 7 days *and* whose `first_seen_at` is before the cutoff. This matches label semantics instead of discovery time, and every row used is still visible before the cutoff. (b) "Active in the last 7 days OR active in ≥ 2 of the last 4 weeks." (c) 7-day window vs 10-day window. Every variant is label-free and needs no model. *Value:* a stronger fallback if no learned policy confirms.

**7. EXP-021 — smooth persistence and blended ranking.** Rank by an exponentially weighted event score (τ ∈ {3, 7} days), and by w · rank(HGB) + (1 − w) · rank(EW score) with w ∈ {0.25, 0.5, 0.75}. Use EXP-016's k. *Hypothesis:* the blend inherits persistence's recency signal and HGB's breadth.

#### Phase C — ranking quality (refits; select on AP)

**8. EXP-022 — training window and recency weighting.** Under the EXP-014a daily-origin protocol, compare 28/56/84/112-day windows; exponential example weights with half-lives of 2 and 4 weeks; and Thursday-only vs all-weekday training origins. EXP-009 tested recency only under the old weekly protocol and old thresholds.

**9. EXP-023 — HGB capacity tuning.** The fixed HGB settings (80 iterations, 15 leaves, minimum leaf 75) were never tuned. Search a small grid: iterations {80, 200, 400} with early stopping on a time-held-out week, leaves {15, 31}, minimum leaf {25, 75}. Fit per target and select on inner-fold AP.

**10. EXP-024 — short-horizon features.** F0 has 7/30/90-day windows only. Add 1-day and 3-day counts per target, hours since last event, and an "active in the final 24 h" flag. *Hypothesis:* with a 4-hour gap before the target week, the last few days carry signal that 7-day aggregates blur. These were not part of EXP-012's rhythm/state groups.

**11. EXP-013 — finish relational ablations.** Close the three audit blockers in [EXP013_CODE_REVIEW.md](../reports/EXP013_CODE_REVIEW.md), run the registered study, then additionally rescore the selected arm under the Phase A decision policy. This experiment is required for presentation Q2 whatever its result; a negative finding is acceptable evidence.

**12. EXP-025 — seed bagging and rank blend.** Average 5 HGB seeds (variance reduction), and blend HGB with logistic ranks (logistic blends scored highest on push in EXP-010, though unselected). It is cheap and usually gives a small, stable AP gain.

#### Phase D — population, confirmation and deployment

**13. EXP-026 — official roster alignment.** When `evaluation/test_input.csv` is published (Oct 28), compare the roster with the anchored eligibility proxy using permitted records only: overlap, activity levels and target prevalence. Redefine the evaluation population to match the roster, and rescore all frozen candidates on it before the Nov 4 decision.

**14. EXP-027 — fresh-origin confirmation harness.** Build a script that takes a frozen candidate config and a new instructor release, rebuilds features and labels for each newly mature origin, and appends confirm-tier rows. It must not change any candidate. Origins: 10-07, 10-14, 10-21 (dates in §29.5). The Oct 12 VAL-LOCK holdout can run as one more one-shot check under §24.

**15. EXP-028 — release drift and late-label sensitivity.** Re-clone the instructor repository; the pinned snapshot `506d860…` ends 2026-10-13. Compare consecutive releases: how many September/October labels change, by target (`open_issue` is expected to be the most sensitive). Use the result to set the training-label maturity buffer for the final refit, and to decide whether nearly mature recent weeks may enter training.

**16. EXP-029 — final-day dry run.** On a past Thursday origin, simulate the Nov 11 sequence end to end: re-clone, feature cutoff 20:00 Taiwan, fit, decision policy, CSV, then validate with the official `evaluate.py`. Time each step and keep a persistence-only fallback script that runs in a few minutes.

**17. EXP-030 — champion freeze and final package.** Choose the champion by the confirm-tier gate in §29.2; if nothing passes, the champion is `P_persist7` (or the best confirmed EXP-020 rule). Refit on all labels mature by the final release, freeze code/config hashes, run the §18.8 independent audit on this one policy only, and log the result.

### 29.5 Revised timeline (Oct 18 – Nov 11, Asia/Taipei)

> **Superseded by §30.12.** October 25 is a provisional shortlist checkpoint, not a course deadline. Source coverage determines label maturity; calendar dates alone do not authorize confirmation. New feature candidates and unchanged confirm-plan v0 have distinct exposure/eligibility rules (§30.9).

Fresh origins mature at label-window end + 5 d 21 h (EXP-014b convention). The release dates below are the first daily release (~21:00 Taiwan) expected to contain each mature week, assuming the daily schedule holds.

| Dates | Work | Exit criterion |
|---|---|---|
| **Oct 18–20** | EXP-014b Stage 1; EXP-016, 017, 018, 021 on saved EXP-014a predictions; EXP-020 rules; start EXP-027 harness; re-clone for EXP-028 | Screen-tier table for every Phase A/B candidate |
| **Oct 21** | **Origin 10-07 matures** (release ≈ Oct 21). Run the harness on all Phase A/B candidates that passed the screen | First confirm-tier row |
| **Oct 20–24** | EXP-022 → EXP-023 → EXP-024 (ranking, AP-selected); close EXP-013 audit blockers and run it | Best ranking model chosen on Sept AP |
| **Oct 25** | **Candidate freeze #1:** at most 3 complete six-target policies plus `P_persist7` and the best EXP-020 rule | Frozen configs and hashes |
| **Oct 26** | EXP-015 go/no-go memo (expected: no-go); EXP-019 per-target selection on frozen candidates | Memo logged |
| **Oct 28** | **Roster published → EXP-026.** **Origin 10-14 matures** (release ≈ Oct 28) → confirm run | Roster-aligned population fixed; second confirm row |
| **Oct 29 – Nov 2** | EXP-025 bagging/blend; rescore on the roster population; **candidate freeze #2 by Nov 2** (only confirm-passing or fallback policies survive) | Final candidate list |
| **Nov 4** | **Origin 10-21 matures** (release ≈ Nov 4) → final confirm run; **choose champion vs fallback** | Decision logged with all three confirm origins |
| **Nov 5–7** | EXP-029 dry run; EXP-030 freeze, final refit and audit; optional one-shot VAL-LOCK | Tested final commands; fallback ready |
| **Nov 7–10** | Slides (Q1–Q4) and video from logged results only. Origin 10-28 matures ≈ Nov 10–11: sanity check only, **no retuning** | Deliverables drafted |
| **Nov 11, 21:00–23:55** | Re-clone final release, record SHA, run frozen pipeline, validate CSV, submit; use the fallback if anything fails | Submission |

**Stop rules:** no new candidates after Nov 2; no code changes after Nov 7 except bug fixes that leave predictions unchanged on the dry-run origin. If the confirm tier is inconclusive, submit `P_persist7` or the confirmed EXP-020 rule.

### 29.6 Status update (2026-10-18, session s29-unattended-20261018; append-only, supersedes the §29.3 status column)

Evidence: log §26 and `reports/unattended_sessions/s29-unattended-20261018/`. None of the statuses below means
"Completed" in the promotion sense. Screen results are development evidence only.

| ID | status |
|---|---|
| EXP-014b | **Stage 1 complete**: C1 nominated, awaiting Stage 2 (10-07/10-14). C2, C3 dropped |
| EXP-016 | **Screen complete**: P016_a100 and P016_sel `screen_pass` (provisional) |
| EXP-017 | **Screen complete**: add/both `screen_pass`, drop `rejected` |
| EXP-018 | **Screen complete**: `rejected` |
| EXP-019 | **Screen run**: `screen_pass`, but contaminated by construction and dominated by its sources; not carried forward |
| EXP-020 | **Screen complete**: multi and w10 `screen_pass`, aligned `rejected`. Best rule: P020_multi |
| EXP-021 | **Screen complete**: ew3, ew7 and blends `screen_pass`. Leading candidate: P021_blend50 |
| EXP-022 | Draft preregistration (`docs/exp/EXP022_PREREGISTRATION_DRAFT.md`); not frozen, not run |
| EXP-027 | **Harness built**, older-fold self-test exact; awaiting a release with mature 10-07 labels and the D3 decision |
| EXP-013 | Ongoing (unchanged) |
| EXP-028 | Blocked: only one local release SHA |
| others | unchanged |

Confirm-tier plan v0: `reports/unattended_sessions/s29-unattended-20261018/CANDIDATE_MANIFEST_confirm_v0.json`.
Freeze #1 (Oct 25) may add EXP-022–024 ranking variants. Those can be confirmed only on origins after their freeze.

**§24 status amendment (2026-10-18, user decision D3):** the Oct 12 holdout is now a **non-independent sanity check**.
The 10-07 and 10-14 confirm label windows cover its outcome week, and those confirm results will inform selection.
The confirm tier (§29.2) is the primary future-performance evidence. See log §26.4.

**§29.6 status addendum (2026-10-18, after log §27):** EXP-022 — **screen complete**. It was frozen before any fit
(config `f6a8b510…`) and run as `exp022-ranking-window-v1`. The registered AP rule selects **H2** (2-week-half-life
sample weights): all4 mean AP +0.0063 vs REF, 4/4 wins. Under `P021_blend50` it adds +0.0041 official score vs the
same layer on REF (4/4). This is development evidence only, with no promotion. Whether H2 enters the confirm candidate
set is decided at freeze #1 (Oct 25). If it does, it can be confirmed only on origins after that freeze. Confirm-plan v0
(10-07) is unchanged.

### 29.7 Amendment: bounded EXP-015 RDL pilot admitted (2026-10-18, user decision D9; append-only)

**Decision.** The user admitted a **bounded relational deep learning (RDL) pilot** as EXP-015. This amends the
§29.3 line "Explicitly not in the backlog … EXP-015 (GNN)" for this pilot only, and waives §23.1 gate 1 (stable
EXP-013 relational gain). Nothing has been run; the preregistration is a draft and is not frozen.

**Why.**

1. The assignment brief invites models that "directly analyze relational databases", naming GNNs and RDL (Lecture 5).
   The graded presentation asks how cross-table relations (Q2) and temporal information (Q3) were modeled. It also
   grades additional experiments and analysis. A tested GNN gives direct evidence for Q2/Q3 whether it wins or loses.
2. §29.1 point 3 applies here too. EXP-011/012 were scored through the broken threshold layer, so they say little
   about ranking value. Under §29.2 a GNN can be judged purely as a **ranker** on mean per-target AP and plugged into
   the frozen `P016_a100` / `P021_blend50` layers, exactly as EXP-022 was. It does not need its own decision layer.
3. The lecture's RDL argument (learned JOIN + aggregation instead of hand-chosen windows; temporal sampling to avoid
   leakage) is a testable alternative to the handcrafted F0 aggregates. The node-only MLP arm isolates the
   contribution of message passing.

**Scope (bounded).** One pilot with three arms: MLP0 (node-only, F0 inputs), GNN1 (2-layer heterogeneous GraphSAGE on
the as-of relational entity graph) and GNN1T (GNN1 + edge recency). The protocol, population, origins and decision
layers are those of EXP-014a/EXP-022. Selection is on all4 mean AP vs REF **and** vs MLP0. Draft preregistration:
[EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md](exp/EXP015_RDL_PILOT_PREREGISTRATION_DRAFT.md). Still out of scope:
RelGNN, RelGT, HGT, RT, architecture search, text features, and handcrafted 2-hop features.

**§23.1 gates after D9.**

| gate | status |
|---|---|
| 1. EXP-013 relational gain | **Waived (D9).** EXP-013 stays Ongoing and is still required for Q2; the two results are reported side by side when both exist |
| 2. Tested final-day fallback | `P_persist7` and `P021_blend50` run from existing code. EXP-029 (dry run) is still required before Nov 11. The pilot cannot delay it |
| 3. Compute budget | New environment `asg2-rdl` (`configs/exp015_rdl_environment.yml`, `requirements-exp015.txt`), RTX 3070 Ti Laptop 8 GB. Work cap: 3 days |
| 4. As-of semantics | Per-origin graph built only from rows visible before the feature cutoff, before any sampling; asserted in tests (preregistration) |
| 5. Comparator, folds, metric, stop rule | Fixed in the preregistration; must be frozen with a config SHA-256 before any fit |

**Register row (appended to §29.3).**

| # | ID | Experiment | Phase | Refit | Depends on | Status |
|---|---|---|---|---|---|---|
| 11b | **EXP-015** | Bounded RDL pilot: MLP0 / GNN1 / GNN1T, AP-selected | C — ranking / Q2–Q3 | Yes | EXP-014a protocol; EXP-022 REF/H2; `asg2-rdl` env | **Pending** (draft preregistration) |

**Timeline change (supersedes the Oct 26 memo in §29.5).** Freeze the preregistration and build the graph code Oct 19;
fit and screen Oct 20–21; record the result before **freeze #1 (Oct 25)**. The 10-07 confirm run (Oct 21) takes
precedence over pilot work on that day. The pilot competes with EXP-023/024 and EXP-013 for the Oct 20–24 slot. If it
runs late, EXP-013 keeps priority, because Q2 needs it.

**Promotion path.** Unchanged from §29.2. A selected arm is only eligible for freeze #1 consideration. It can be
confirmed only on confirm-tier origins after its freeze (10-14 / 10-21). `P_persist7` stays the default. A negative
pilot result is acceptable and is recorded as such.

**§29.7 status addendum (2026-10-18, after log §28.4):** EXP-015 is **Ongoing**. The preregistration is frozen (config
`89f2b3c0…`, amendment A1). The as-of graph builder `src/rdl_graph.py` passes its leakage tests and the real-data
truncation-invariance check. There are no fits and no scores, and the runner is not written yet.

**§29.7 status addendum (2026-10-19, after log §28.5):** Amendment A2 is applied (structural-edge time fallback and
runner fill-ins). The runner is built. The no-fit preflight on real data passed 322/322 checks. EXP-015 is **ready to
fit, pending the user's approval**. No fit has run.

**§29.7 status addendum (2026-10-19, after log §28.6):** EXP-015 is **Completed — recorded negative**. The single
user-approved fit `exp015-rdl-pilot-v1` ran under the frozen config (`89f2b3c0…`; A1, A2), 75 min, all full-batch,
with no OOM and no blocked origin. all4 mean AP: GNN1T 0.5126 and GNN1 0.5063, vs REF 0.5312 and H2 0.5375. GNN1T
beats REF on 2/4 origins, GNN1 on 0/4. Both GNNs beat MLP0 (0.4666) on 4/4 origins. No arm is eligible, so no arm
enters freeze #1 from this pilot. The REF-vs-H2 choice stays with freeze #1 (§29.6), and `P_persist7` stays the
default. The orchestrator verified the result independently: AP, selection, decisions, official parity, the gate,
the bootstrap and the MLP0 reproducibility (Δ = 0.0). The deferred ablations are not run. This is development
evidence on inspected September weeks; the EXP-013 side-by-side for Q2 is still open.

---

## 30. Historical feature-campaign protocol — October 20–21, 2026

> **October 22 status amendment:** The forward research program is now §31. Preserve the formulas, frozen budgets and gates below when interpreting/reproducing the completed feature campaign. Log §29 and the continuation report record seven accepted negative results, two conditional skips, no feature nominee and an incomplete original EXP-013. Prospective instructions below to start S/M/SM again, stop architecture research, or end new research at the October 25/26 shortlist are superseded for the new phase. Original confirmation manifests remain unchanged.

### 30.1 Authority, scope, and what this registration does

**Planning dates:** drafted 2026-10-20 and finalized 2026-10-21, Asia/Taipei. The `20261020` suffix in planned paths and seeds identifies the campaign's start, not an earlier executable freeze. **Status:** prospective research/implementation specification; no new feature panel, candidate model, score, confirmation, or deployment is claimed by this amendment. The user requested a detailed update after reviewing current results and `FEATURE_ENGINEERING.md`. This section records the intended work and resolves routine design choices so the next implementer does not have to invent the experiment.

**Precedence:** this section controls remaining work where it conflicts with older prospective instructions. Completed results and their original acceptance rules remain unchanged. The original EXP-013 freeze remains a distinct historical-protocol study. The existing `CANDIDATE_MANIFEST_confirm_v0.json` and its pinned source/config files remain unchanged. New follow-ups receive new IDs rather than being presented as additional completed EXP-012/015 work.

**Implementation freeze still required:** this is the human-readable scientific registration. Before fitting a new arm, materialize the exact config, ordered columns, expanded origin/user membership, code/data/environment hashes, output schedule, and this section's gates in a machine-readable freeze. Record the freeze in the log before scoring. A discovered ambiguity or implementation impossibility must be resolved in a dated, score-free amendment; never silently choose a formula after observing candidate scores. Planned paths below do not assert that those files already exist.

**Research objective:** improve the complete six-target prediction policy's future-week official combined F1 while obtaining defensible temporal and relational ablations. The objective is not to exhaust 173 feature templates, maximize feature count, or select the largest September score.

**Alternative considered:** implement the catalogue broadly, tune several learners, and choose the largest feature union. Rejected for this campaign: it entangles representation, training policy, model capacity and binary decision changes; creates many adaptive comparisons on four reused weeks; and increases final-release failure risk. Chosen design: one fixed learner/training policy, a finite set of coherent feature groups, four explicitly allowed combination slots, and separate label-blinded confirmation.

**Scope of this edit:** update this plan only. Model execution, data refresh, code implementation, Git commits/pushes, publication, and submission are not performed by this edit. This statement records what was done, not a new requirement for repeated permission questions during later authorized work.

### 30.2 Evidence motivating the change, with limits

| Observed evidence | Source | Consequence for the next study | Inference that is not supported |
|---|---|---|---|
| Original F0 has 59 columns, including 7/30/90-day counts, last-seen age, 30/90-day repo breadth, prior-week activity, count change, and eight-week active-week count | `src/features.py`; frozen EXP-011/012 inventories | Start from exactly those columns and audit overlap before adding features | F0 already represents every temporal or relational mechanism |
| Persistence 0.5166; REF + fixed 50/50 rank blend 0.5454 on all4 | Log §26.2; session report | Retain persistence's positive count and the existing blend as fixed decision rules | The blend is already confirmed for November |
| H2 AP 0.5375 versus REF 0.5312; H2 + same blend 0.5495 versus 0.5454 | Log §27; EXP-022 findings | Use H2 as a provisional development control, without changing confirm-plan v0 | H2's development selection is independent future evidence |
| The blend adds useful decayed-activity information to model ranking | Log §§26–27 | Test whether short-horizon and smooth-activity features let HGB exploit that information directly | Short-horizon features are guaranteed to beat the blend |
| EXP-012's 24-column rhythm and 24-column state additions fail their original gates; union not run | Log §23.3–23.4 | Preserve that negative result; allow a separately named, finite reevaluation under current chronology/policies | Every template in G02/G03 or their union has been disproved |
| EXP-013 has frozen score-free panels and zero candidate fits | Log §24; frozen run; preservation record | Close audit readiness and obtain the registered relational comparison | Code existence or unit tests establish relational predictive value |
| EXP-015 completes negatively against HGB, although both GNNs beat MLP0 | Log §28.6 | Close the pilot; defer more architectures; prioritize auditable aggregates | Relations contain no incremental information, or every GNN would fail |
| All4 contains only four repeatedly inspected weeks; 09-30 is provisional on the pinned release | Log §§26–28 | Bound the number of trials and require later confirmation | User-cluster intervals quantify uncertainty across future weeks |

The score differences in this table are comparable only within their stated all4 protocol. They are not comparisons with EXP-006's earlier 0.454677 population/schedule. The fixed blend remains useful even if adding its decay score as a model input fails.

### 30.3 Current register and revised execution order

The order below is scientific priority, not a claim that separate compute workers or agents have been assigned. Execute sequentially unless later execution instructions explicitly permit parallel work. A mature, already-frozen confirmation run takes priority over development fits.

| Priority | ID / work item | Current status | Exact next action / dependency | Completion or exit |
|---|---|---|---|---|
| 0 | Source/exposure ledger and baseline preflight | Pending work package, not a new experiment ID | §30.4 and §30.10; preserve old release; identify observed labels; reproduce REF/H2 | Frozen, authenticated campaign inputs and parity report |
| 0 | EXP-027 unchanged v0 confirmation; EXP-014b Stage 2 | Harness implemented; confirmation pending source maturity | §30.9.1; D3 is resolved, so it is not an outstanding user-decision blocker | Logged mature-origin results or an explicit coverage/access blocker |
| 0 | EXP-028 release/late-label audit | Pending; only one local release observed at review | Obtain a second permitted release without destroying the first; §30.10.4 | Stable-ID release comparison; no silent change of training buffer |
| 1 | **EXP-024 short horizons** | Pending; not run | Exact S18 group, §30.5.2; fixed H2 control | Five-origin paired study and recorded screen outcome |
| 1 | **EXP-013 original registered study** | Ongoing; score-free preflight only | Reconcile older audit review with current code; §30.7.3 | Original frozen-protocol result and independent acceptance, including a negative exit |
| 2 | **EXP-031 smooth activity/momentum** | New, Pending | Exact M12 group, §30.5.3 | Five-origin paired study; no decay/window search |
| 2 | **EXP-035 SM interaction slot** | New, Pending | Fit S18 and M12 single groups first; union is scheduled regardless of their individual screen verdicts | Exact F0+S18+M12 comparison and removal contrasts |
| 3 | **EXP-033 relational ranking follow-up** | New, Pending | Same A9/B7 formulas as EXP-013, new H2/daily-origin protocol; §30.7.4 | A9, conditional B7/A9+B7, with coverage and joins independently checked |
| 3 | **EXP-032 rhythm/state reevaluation** | New, Pending | Reuse exact R24/W24 definitions, not old unmatched predictions | Two single-group studies under current protocol |
| 3 | **EXP-035 RW interaction slot** | New, Pending | R24/W24 single groups fitted first; union scheduled even if individual gates fail | Exact union comparison; no claim it ran in EXP-012 |
| 4 | **EXP-034 selected cross-target representation** | New, Pending and conditional | §30.5.6; only if a temporal candidate qualifies under §30.8 | One X6 group; no pair/lag search |
| 4 | **EXP-035 TR / TX interaction slots** | New, Pending and conditional | §30.6 and §30.8 determine parents without October outcomes | At most one temporal-relational and one temporal-cross union |
| 5 | **EXP-023 capacity tuning** | Deferred from the former EXP-022→023→024 critical path | Only after the feature shortlist and core deliverables are ready; §30.7.7 | Bounded follow-up or explicit not_run; cannot delay confirmation |
| 5 | **EXP-025 bagging / HGB-logistic blend** | Deferred | §30.7.7; seed changes must actually alter predictions before bagging is justified | Bounded follow-up or explicit not_run |
| D | EXP-026 roster alignment | Scheduled, contingent on Oct 28 release | §30.9.5; retain fixed primary population and separate roster sensitivity | Coverage, deterministic membership adaptation, valid inference test |
| D | EXP-014 integration; EXP-029 dry run; EXP-030 final package | Pending | §30.9–30.12 | Selected/fallback complete policies, real timed rehearsal, frozen release recipe |
| Closed | EXP-011 / EXP-012 / EXP-015 / EXP-022 | Completed bounded 011/012/015 studies; 022 screen complete only | Preserve artifacts and conclusions; no rerun relabeling | No new scientific work charged to a closed study |
| Closed | EXP-016–021 / EXP-014b Stage 1 | Development screens executed | Carry forward only existing registered candidates; no wider grids or new per-target choices | Confirmation or recorded rejection/deferment |

**Reordered dependency:** EXP-024 now precedes capacity tuning. EXP-013 closure remains high priority for Q2. EXP-031 and the SM interaction come before broad cross-target work. EXP-032 is a bounded reevaluation, not an unrestricted reopening of the catalogue. More GNN work is not on the remaining critical path.

### 30.4 Shared campaign contract: data, time, population, models, and binary decisions

#### 30.4.1 Input provenance and permitted information

1. Development source is the pinned instructor release `506d860693df60a698b68aa63a8edb4a984c8e1b`, latest observed row `2026-10-13T12:59:29Z`. Verify the actual SHA and the eight Parquet hashes before use. If unavailable or different, stop the paired comparison and register a source-version amendment rebuilding both control and treatments; do not mix old reference predictions with new labels.
2. Targets, always in this order: `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`. All arms receive the same six-stream F0 view. No raw user/repository IDs, origin timestamp, label, eligibility flag newly created by this campaign, or result-derived variable becomes a new model column. Preserve the accepted 59-column F0 order, including any existing F0 mask columns; changing them is a different study.
3. New temporal observations use `OfficialData.events`'s visibility-time contract: PR/issue/review entities use `first_seen_at`; comments, review comments and pushes use `created_at`; issues exclude PR rows. These are observed-record counts, not guaranteed actual opening/submission counts. Retain source PK namespaces and validate unique source rows before counting. Do not deduplicate distinct same-timestamp events.
4. New relational groups use only the three direct-event streams and endpoint contract in §30.5.5. They do not silently expand to retrospectively known PR/issue metadata. No external activity data, text extraction, live GitHub queries, or later-known lifecycle fields.
5. Parse source timestamps as UTC. All elapsed windows below are durations measured from the exact UTC cutoff, not locale-dependent calendar dates. Features require visibility strictly `< C`; a row at `C` is excluded and one at the left boundary is included.

#### 30.4.2 Report origins and population

| Role | UTC origin S | Taiwan target-week start | Decision-bearing in development? |
|---|---|---|---|
| Continuity / transition diagnostic | 2026-09-02T16:00:00Z | 2026-09-03 00:00 | No; never selects a feature or breaks a tie |
| Screen 1 | 2026-09-09T16:00:00Z | 2026-09-10 00:00 | Yes |
| Screen 2 | 2026-09-16T16:00:00Z | 2026-09-17 00:00 | Yes |
| Screen 3 | 2026-09-23T16:00:00Z | 2026-09-24 00:00 | Yes |
| Screen 4, provisional | 2026-09-30T16:00:00Z | 2026-10-01 00:00 | Yes, disclosed; nonprovisional sensitivity is mandatory |

`all4` is screen 1–4. `mature3` is screen 1–3, not EXP-014a's differently defined `recent3`. `all5` includes continuity and is diagnostic only. Report every pool by its name and explicit origin list.

For **every** training and report origin `s`, feature cutoff is `C_s = s - 4 hours`. The population anchor is `B_s = s - 14 days - 4 hours`. Include a cohort user iff at least five visible source records across the six streams lie in `[B_s - 90 days, B_s)`. Sort by stable `UserID` ascending. Apply the rule per origin, not once at the report date for all training rows. Do not filter users using any future outcome, new feature support, or candidate prediction. Expected report populations on the pinned source are 7,417 / 6,382 / 5,862 / 5,497 / 5,178; assert exact membership against the reference, not only these counts.

#### 30.4.3 Training and label clocks

For report origin `S`, decision time is `A_S = S - 3 hours`. Eligible daily training origins are exactly `s = S - 63 days, S - 62 days, ..., S - 8 days` at 16:00 UTC: **56 origins**, each satisfying `s + 7 days <= A_S`. Each training origin uses its own `C_s` and population. The report target window is `[S, S + 7 days)`.

Training labels are built with `label_as_of = A_S`; report labels use the pinned development source's latest observed timestamp. Reuse the existing target builder and its audited occurrence/visibility semantics; do not replace it with the feature observation stream. Do not train on report labels. The report-label maturity rule used for new October confirmation is **not** an extra training buffer in this EXP-022-matched replay. These are separate clocks; changing the training buffer requires a distinct full-system comparison.

H2 origin weight is `v_s = 0.5 ** (((S - s) / 7 days) / 2)`. Repeat `v_s` over that origin's actual eligible rows, then divide all training-row weights by their arithmetic mean. Do not normalize each origin to equal total weight; populations differ. Compute the same weights for every feature arm. There is no target class reweighting, oversampling, new downsampling, or additional calibration split in this campaign.

#### 30.4.4 Fixed learner and reference reproduction

Fit six separate `HistGradientBoostingClassifier` models with learning rate **0.07**, max iterations **80**, max leaf nodes **15**, minimum samples per leaf **75**, L2 **3.0**, `early_stopping=False`, and seed **42**, using H2 sample weights. Freeze all other resolved estimator defaults from the selected runtime; record `get_params()` and library versions. Use existing feature-matrix dtype/ordering behavior, without new scaling, imputation, feature importance filtering, or categorical encoding. A one-class training target uses the existing constant `DummyClassifier` behavior, is logged, and does not trigger an alternative learner.

Reproduce two controls before candidate fitting: **REF** = F0 with unit weights, **H2** = F0 with the weights above. Compare their populations, labels, F0 values, raw probabilities, fixed-layer decisions, per-target AP and official scores with EXP-022 at all five origins. Numeric probability/metric tolerance is `1e-12`; keys, labels and binary decisions require exact equality. A different environment may not be excused by a wider tolerance: resolve the runtime or register a separate source/runtime comparison before fitting candidates.

#### 30.4.5 Fixed decision layers

For target `a` and report population, let `k_a` equal the number of users with at least one visible record in `[C_S - 7 days, C_S)`. This is exactly the `P_persist7` positive count on those users.

- **P016_a100:** predict positive for the top `k_a` users by candidate raw HGB probability. Higher score sorts first; ties use `UserID` ascending. If `k_a=0`, predict all zero; if it equals population size, predict all one.
- **P021_blend50:** compute `EW7` from all visible target records in `[C_S - 90 days, C_S)` as `sum(exp(-age_days/7))`. Rank candidate probabilities and EW7 separately across the same users using average ranks for ties, divide each rank by population size, and average the two ranks with weights **0.5/0.5**. Select top `k_a` using the same stable-ID tie-break.
- **P_persist7** and the existing **P020_multi** are fixed comparators. The latter is active in the last seven days OR active in at least two of four trailing complete seven-day blocks, as implemented in the frozen rule code.

Every new learned arm is scored through **both** layers. The campaign's primary deployable layer is P021_blend50; P016 is the fixed diagnostic that isolates model ranking. Do not choose blend weight, alpha, threshold, decay constant, or decision layer per candidate/target/week using these results. No transferred Platt layer is introduced. Applying a fixed top-k rule to a newly supplied roster may recompute counts/ranks on that roster; it does not permit tuning its parameters.

### 30.5 Exact feature groups, formulas, counts, and exclusions

#### 30.5.1 Common notation, support, and output ordering

For user `u`, stream `a`, feature cutoff `C`, let `E_a(u,W)` be the visible source rows in `[C-W days,C)` and `n_a(W)` their count. `age_days(e)=(C-visibility(e))/86400 seconds`. Use all six target streams in target order unless a group explicitly restricts sources. New temporal column prefixes must distinguish entity discovery: `observed_pr_first_seen`, `observed_issue_first_seen`, `observed_issue_comment`, `observed_review_first_seen`, `observed_pr_review_comment`, `observed_push`.

Construct temporal groups on the same `(origin_utc, UserID)` keys as F0; append in target order, then suffix order as listed below. Counts use int32 after checking range, binary flags int8, and real-valued new columns float32 with float64 intermediate arithmetic. All retained new columns must be finite. Do not clip valid outliers or add missingness columns after inspecting scores. Zero observations means zero count/decay. Undefined statistics in R24/W24/A9/B7 retain their already registered support/sentinel rules. Metadata/coverage tables can contain missing values without being inserted into model inputs.

The dtype rule above applies to newly defined S18/M12/X6 columns. Reused R24/W24/A9/B7 retain their exact existing registered dtypes, including int8 weekly-state counts where specified. Group-union order is F0 first, then S18, M12, R24, W24, A9, B7, X6 for the groups actually present; within a group use its registered order. No alphabetical resorting of all model columns.

Before model fitting, classify each proposed column as `new`, `exact_duplicate`, `near_duplicate`, or `replacement` against F0. A deterministic transform still has to be disclosed even if it is not byte-identical. The groups below have fixed membership: discovery of a true exact semantic duplicate or an invalid formula requires a score-free amendment recording the final count/order, rather than silently dropping it mid-run. No univariate score-based filtering, automatic SHAP selection, or feature-importance cutoff.

#### 30.5.2 S18: EXP-024 short-horizon activity, 18 columns

For each of six streams append these three columns, in this order:

| Suffix | Exact value | Purpose / catalogue mapping |
|---|---|---|
| `count_1d` | `n_a(1)` | T01; activity in the final observed 24 hours |
| `count_3d` | `n_a(3)` | T01; activity in the final observed 72 hours |
| `active_bins_3d` | Number of nonempty bins `[C-(j+1)d,C-jd)`, `j=0,1,2` | T03 complete-bin variant; separate one burst from repeated daily presence |

This is a **complete 24-hour-bin variant**, not distinct UTC calendar dates. At the deployment clock, the final bin ends at the available 20:00 Taiwan feature cutoff, not midnight. Preserve this definition at every origin. Zero-history users get three zeros. Do not append an additional `active_1d` flag or hours-since-last-event: the former is determined by `count_1d`, and the latter rescales F0's elapsed-days recency. No 2/5/14-day window search. F0+S18 has **77 columns**.

Hypothesis: users with similar seven-day counts differ in immediate activity and continuity; S18 can improve ranking near the fixed positive budget. Report separately whether S18 improves model-only P016, the blended layer, both, or neither.

#### 30.5.3 M12: EXP-031 smooth activity and momentum, 12 columns

For each stream append these two columns, in this order:

| Suffix | Exact value | Purpose / catalogue mapping |
|---|---|---|
| `ew_count_tau7_90d` | `sum(exp(-age_days(e)/7))` over `E_a(u,90)` | T49; identical numerical definition to the fixed EW7 ranking input, now exposed to HGB |
| `rate_difference_7v21d` | `n_a(7)/7 - (n_a(28)-n_a(7))/21` | T47; recent per-day activity minus the preceding 21-day per-day activity |

Use arithmetic difference rather than a ratio to avoid an unregistered small-denominator treatment. Counts at exactly `C-28d` enter the prior interval; those at `C-7d` enter recent7 only. No supervised normalization, fitted personal baseline, epsilon, or extra decay grid is needed. A zero-history user gets zero for both. M12 has **12 new columns**, so F0+M12 has **71**.

The presence of EW7 in the fixed blend does not make this trial pointless: the question is whether HGB can condition its use on the user's other history. A gain in P016 without a gain in the blended policy is recorded as representational value but insufficient deployment increment. Adding M12 does not change the external blend or its weight.

#### 30.5.4 R24 and W24: EXP-032 exact temporal follow-ups

Use the definitions already frozen in `configs/temporal_feature_ablations.json` and `src/temporal_features.py`; copy their hashes and complete ordered columns into the new campaign freeze. Do not change them to the broader G02/G03 catalogue while reusing the same group label.

- **R24**: four suffixes per stream: `median_observation_gap_90d`, `normalized_inactivity_90d`, `gap_supported_90d`, `active_day_fraction_28d`. Use the last **10** visible records in the preceding 90 days, stable handling of equal timestamps, minimum three records and two distinct timestamps, unsupported gap sentinel **-1**, normalization floor **1/24 day**, and the original support logic. Active-day fraction uses 28 complete cutoff-relative 24-hour bins. It is not a calendar-day seasonal feature.
- **W24**: four suffixes per stream: `active_weeks_4w`, `active_streak_8w`, `inactive_streak_8w`, `state_switches_8w`. Weekly blocks are `[C-7(j+1)d,C-7jd)`, `j=0..7`; streaks start at `j=0`; switches count the seven adjacent binary-state differences. A no-history user has zero active weeks/streak/switches and an eight-week inactive streak, following the accepted builder.

Each single-group arm has **83 columns**. The union has **107**. There are no additional transition probabilities, survival models, weekly slopes, alternative record caps, or seasonal windows in EXP-032. The proposed union is EXP-035 RW and was not executed by EXP-012. These runs change the training/replay/decision system relative to EXP-012, so they are separate follow-up studies; they do not annul its original result.

#### 30.5.5 A9 and B7: EXP-033 relational groups

Reuse the exact ordered feature arrays and formulas in `configs/relational_feature_ablations.json`. **A9** has nine columns; **B7** has seven. Join only `Pushes`, `IssueComments`, `PRReviewComments` to cohort actors and repositories visible before C. Use each source table's own PK namespace, many-to-one FK joins, stable public IDs in saved output, and the original null/unknown/not-yet-visible/missing-stable-ID categories. Exclude focal-user events before every neighbor aggregation. F0 retains valid observations even when a repository endpoint is unresolved.

Let `R_u(W)` be the valid linked direct-stream repository set and `n_ur(W)` the count of linked direct rows. A9 is:

1. `direct_repo_distinct_7d = card(R_u(7))`.
2. `direct_repo_distinct_30d = card(R_u(30))`.
3. `direct_repo_distinct_90d = card(R_u(90))`.
4. `direct_repo_top_share_30d = max_r n_ur(30)/(1+sum_r n_ur(30))`, zero for empty exposure.
5. `direct_repo_entropy_30d = -sum_r p_r*ln(p_r)`, `p_r=n_ur(30)/sum_r n_ur(30)`, zero for empty/one repo.
6. `direct_repo_returning_fraction_30d`: fraction of recent30 repos also observed in `[C-90d,C-30d)`, zero for empty recent set.
7. `direct_repo_mean_activity_types_30d`: unweighted mean over recent30 repos of distinct allowed direct-source types on each focal user/repo, zero when empty.
8. `direct_repo_has_history_30d = 1[card(R_u(30))>0]`.
9. `direct_repo_unresolved_events_30d`: focal visible direct rows in recent30 with any of the four disjoint unresolved endpoint reasons, each row counted once.

B7 fixes the exposed repo set at `R_u(30)`. Let `m_-u,r(W)` count valid direct events by **other cohort actors** and `q_r=(m_-u,r(last7)/7)/(m_-u,r(previous23)/23+1e-6)`, where previous23 is `[C-30d,C-7d)`. Append:

1. `neighbor_mean_events_7d`: unweighted exposed-repo mean of other-cohort last7 direct counts.
2. `neighbor_distinct_users_7d`: union count of other cohort users with last7 direct activity on exposed repos, deduplicated across repos.
3. `neighbor_mean_rate_ratio_7v23d`: unweighted exposed-repo mean of `q_r`.
4. `neighbor_mean_push_rate_7d`: exposed-repo mean of other-cohort last7 push count divided by seven.
5. `neighbor_mean_issue_comment_rate_7d`: same for issue comments.
6. `neighbor_weighted_rate_ratio_7v23d`: sum of `q_r` weighted by focal `n_ur(30)/sum_r n_ur(30)`.
7. `neighbor_repo_coverage_fraction_7d`: fraction of exposed repos with at least one other-cohort valid direct event in last7.

All empty-exposure outputs follow the frozen zero convention. Retain support/coverage companions and the original uncapped formula; do not silently trim high-degree repos or winsorize q. Verify these summary formulas against the hash-pinned config before implementation; the original config governs its exact field/endpoint handling. Any discrepancy is an amendment, not discretion for the implementer.

F0+A9 has **68**, F0+B7 **66**, F0+A9+B7 **75** columns. No organization, author-pair, PR/issue-state, text, all-time graph, two-hop, or non-cohort activity-completeness assumption is added. A negative result concerns this direct-stream aggregate recipe only.

#### 30.5.6 X6: EXP-034 bounded cross-target representation, six columns

The six ordered columns below are a new, explicitly limited G05 representation study. They derive from already permitted observations, with no new entity metadata or supervised target encoding:

| Column | Exact value |
|---|---|
| `cross_push_to_pr_first_seen_ratio_30d` | `n_push(30)/(1+n_open_pr(30))` |
| `cross_reviews_to_review_comments_ratio_30d` | `n_pr_review(30)/(1+n_pr_review_comment(30))` |
| `cross_push_pr_joint_weeks_8w` | Sum over eight complete cutoff-relative seven-day blocks of `1[push block count>0 AND PR-first-seen block count>0]` |
| `cross_review_comment_joint_weeks_8w` | Same joint-block count for review-first-seen and PR-review-comment streams |
| `cross_active_types_3d` | Number of the six streams with `n_a(3)>0` |
| `cross_activity_entropy_7d` | Natural-log entropy of six `n_a(7)` shares; zero when all six counts are zero |

The first ratio does not measure real push-to-PR conversion, and joint weeks do not establish same-repository or causal workflow. Use float64 ratio/entropy arithmetic then float32; joint/type counts are int32. F0+X6 has **65 columns**. No other pair, lag, one-hot action encoding, transition horizon, or target-specific subset may be added to this study. This group tests whether a compact representation helps HGB use interactions; it does not claim that all six columns supply new raw information.

### 30.6 Finite arm matrix and fit budget

Arm labels refer to **F0 plus the listed groups**, always using H2 weights unless labeled REF. Fit groups on all six targets together; no per-target feature selection. Every fitted arm produces predictions at all five report origins. Compute all4 and mature3 screens from those same predictions.

| Experiment / arm | Added groups | Total columns | Trigger before fitting | Role |
|---|---|---:|---|---|
| Control REF | none; unit weights | 59 | Mandatory parity preflight | Frozen historical/deployment comparator |
| Control H2 | none | 59 | Mandatory parity preflight | Development feature comparator |
| EXP-024 S | S18 | 77 | Validity preflight passes | Single-group short horizons |
| EXP-031 M | M12 | 71 | Validity preflight passes | Single-group smooth activity/momentum |
| EXP-032 R | R24 | 83 | Core S/M work does not lose its reserved budget | Exact rhythm follow-up |
| EXP-032 W | W24 | 83 | Same as R | Exact weekly-state follow-up |
| EXP-033 A | A9 | 68 | Relational feature QA passes | Repository-self increment |
| EXP-033 B | B7 | 66 | A is screen-eligible under §30.8 | Neighbor-only comparison |
| EXP-033 AB | A9+B7 | 75 | Same A eligibility; B need not pass first | Relational complementarity |
| EXP-034 X | X6 | 65 | A nonempty temporal winner T* is screen-eligible | Single cross-target representation |
| EXP-035 SM | S18+M12 | 89 | S and M single-group fits complete, regardless of their screen verdicts | Predeclared possible synergy |
| EXP-035 RW | R24+W24 | 107 | R and W single-group fits complete, regardless of their screen verdicts | Test the union omitted by the earlier individual gate |
| EXP-035 TR | groups(T*) + groups(R*) | 59 + actual unique added columns | Both T* and R* exist and are eligible | One temporal-relational union |
| EXP-035 TX | groups(T*) + X6 | 65 + columns(T*) | T* exists and X is eligible | One temporal-cross union |

There are **12 maximum new H2 feature arms**, not 12 mandatory runs. REF/H2 add two control arms. At five origins, the upper bound is **70 six-target bundles / 420 binary estimators**, including controls; valid immutable cache reuse may reduce actual fits. Constant-target fits still count as estimators. Conditional arms not triggered must appear in the trial ledger as `not_run` with the rule that skipped them. Original EXP-013's registered fit schedule is separate and must not be squeezed into this count or changed to obtain a positive result.

**T*** is selected, after their scheduled results exist, from eligible S, M, R, W, SM, RW using §30.8's fixed ordering rule. It is one group set, not a new mixture of the best columns from each. **R*** is selected from eligible A, B, AB using the same rule. If no candidate qualifies, the respective symbol is `none`; do not substitute the largest failed delta. T* and R* are locked before fitting TR/TX. Eligible X is determined against H2 before TX is fitted. TR and TX cannot become parents of another union; no T+R+X, S+R, S+W, arbitrary pair grid, or all-feature dump is in this budget.

Each completed union reports deltas against H2 **and both exact parent arms**, and leave-group-out comparisons. For SM, dropping S is exactly M and dropping M is S; for RW the parents are W and R; for TR they are R* and T*; for TX they are X and T*. Reuse an already fitted exact parent only if data, user membership, weights, estimator, column order and preprocessing hashes are identical. Otherwise refit the exact parent and charge it as a reproducibility/control fit, not a new scientific arm. Do not simulate removal by zeroing columns in a fitted model.

**Resource ceiling:** use the available CPU environment after REF/H2 parity succeeds, with four compute threads, one fit process, and a six-GiB measured process-RSS ceiling. Reserve at most **12 wall-clock compute hours** for the new campaign, including feature construction, control reproduction and scoring, excluding separately registered original EXP-013 and unchanged v0 confirmation. Set a two-hour limit per arm across its five origins and a 30-minute limit per origin; check budgets between fits and stop without a verdict if a hard overrun occurs. Record actual limits and thread environment in the freeze. A resource failure is `blocked`/`not_run`, never a negative predictive result. A resumed execution may finish only unexecuted cells under the same freeze; it cannot refit and select a luckier seed.

**Deterministic fit order:** after all shared validity checks and the score-free original EXP-013 readiness review, run S → M → SM → original EXP-013 (its own registered schedule) → A → triggered B → triggered AB → R → W → RW → lock T*/R* → triggered X → triggered TR → triggered TX → choose the single feature nominee. Within each new campaign arm, fit report origins in ascending UTC order and targets in the fixed six-target order. Service a newly mature already-frozen confirmation run between arm blocks. If original EXP-013 is blocked solely by its historical auditor/lock compatibility, record that blocker and continue only work whose separate raw-feature/source validity is established; EXP-033 requires its own passing relational QA and cannot inherit a claimed pass. Scientific priority does not permit reordering individual report origins to observe a favorable subset first.

### 30.7 Work packages and exact handling of existing experiments

#### 30.7.1 Preparation and error analysis, before new fits

Create `reports/feature_campaign_20261020/` as the planned compact evidence directory. Record source/evaluator/code hashes, runtime versions, current Git status, baseline parity and the exposure ledger. On existing September predictions only, produce a fixed error table for persistence, REF/P021 and H2/P021 by target and origin: TP, FP, FN, precision, recall, positive count, and AP. Split errors into persistence-positive/persistence-negative users and into cutoff-relative last-observation bands `<1d`, `[1d,3d)`, `[3d,7d)`, `>=7d or no history`. Use only feature-time information for strata. Also report linked-repo support `0`, `1`, `>=2` from the registered direct-event builder.

The purpose is diagnosis and presentation, not another feature-selection search. These strata are fixed here; do not invent favorable subgroups after seeing candidate results. Do not open October labels for this analysis. Any change to the specified feature arms suggested by it must be registered before those arms are fitted and counts against an amended finite budget.

#### 30.7.2 EXP-024 / EXP-031 implementation order

Implement S18 and M12 in a shared point-in-time feature builder with group names supplied explicitly. Keep F0 construction and the original EXP-014/022 sources unchanged. Tests cover math, cutoff boundaries and key preservation (§30.10). Build all required training/report features without labels; validate manifests and cache provenance; freeze before candidate fitting. Run S, then M, then SM. This first block gives the highest-priority answer without waiting for every optional family. Do not tune HGB during it.

#### 30.7.3 Original EXP-013: audit reconciliation, then unchanged registered execution

The October 18 review predates `verify_exp013.py` changes that contain `verify_report_schedule`, `verify_selection_and_gates`, expected fit identities, and exact calibration/matched-F0 checks. This observation does **not** close the blockers by inspection alone. Required sequence:

1. Compare all three review items with current verifier code and `tests/test_exp013_decision_audit.py`; record a blocker-to-function-to-test map. Check exact missing-output detection, independent conditional A/B/union selection, and calibration-policy completeness.
2. Run the relevant auditor/runner tests in an environment with the required dependencies and a new workspace basetemp. Preserve failures. Authenticate the existing `exp013-protocol-freeze-v1` and its preservation record; the recorded score-free package is not blanket approval for fitting.
3. Establish whether any proposed repair changes a frozen scientific rule, expected schedule or protected hash. Audit-only corrections need a recorded code/freeze revision; scientific changes need a dated preregistration amendment **before fitting**. Never edit the protected v1 artifact in place.
4. Run a fresh score-free preflight and independent preflight audit. Reproduce the original F0 probability/threshold references and protect all original inputs.
5. Execute the **original** A→conditional B→conditional A+B, shuffle and reporting schedule in its own new run directory, with the original learner, features, populations, inner choice and gates. Do not inject H2, S18, the new decision layer, or §30.8 into that historical study.
6. Independently verify fitted models, complete schedules, thresholds, labels, selection and metrics; log the actual result and limitations; only then mark EXP-013 Completed. A valid early A failure can close its bounded scientific question without conditional B/union work, as its original protocol permits.

Any ranking follow-up uses EXP-033. If original execution is delayed by an auditor problem, leakage-safe feature construction/QA may proceed, but neither study is accepted by borrowing the other's test count. Q2 can discuss both measured negative and positive results; do not claim it requires a positive relational gain.

#### 30.7.4 EXP-033: same relational features, current development system

This is a new replay with H2 weighting, daily origins, anchored proxy, and the fixed P016/P021 layers in §30.4. It is not a rescore of unmatched old weekly probabilities. Run A after its raw-source and join QA. If A is screen-eligible, run B and AB; AB does not require B to pass alone. If A fails, B/AB are `not_run` under this bounded protocol, not evidence that neighbors cannot help. Report A on all five origins even if it loses. R* comes only from eligible executed arms. Original EXP-013's outcome neither replaces the new gate nor provides independent confirmation for this follow-up.

#### 30.7.5 EXP-032 and EXP-034

Run exact R24 and W24 on the current shared replay, then the unconditional scheduled RW union. Preserve old EXP-012 findings verbatim. No expanded transition/hazard group is hidden in the word "state". X6 runs only after an eligible T* is identified, with F0 alone as its feature control under the shared H2 setup. TX subsequently compares X6's increment on T*; do not fit X only on a favorable target or select pairs from its feature importance.

#### 30.7.6 EXP-035 parent locks and interpretation

Store `parent_selection.json` with the evaluated pool, all eligible arms, gate values, sorting keys, selected T*/R*, ordered groups, and hashes before fitting TR/TX. For each union, record whether it beats both parents. A union that qualifies versus H2 but fails to exceed its stronger parent is redundant for final selection under §30.8.4. A union may help despite one failed individual group only in the **already registered SM/RW slots**; this is not permission to generate new pairings after seeing a failure.

#### 30.7.7 EXP-023 and EXP-025 are moved out of the required campaign

Neither is required before freeze #1. Revisit only if the core feature block, original EXP-013, source audit, and confirmation preparation are complete **and** there are still at least two uninspected eligible October confirm origins for any resulting new policy.

For EXP-023 the only allowed initial follow-up is a new registered comparison on one fixed selected feature view: the four Cartesian settings `max_iter={80,200}` × `max_leaf_nodes={15,31}`, with min leaf 75 and all other settings unchanged, `early_stopping=False`. The existing 80/15 result is a reusable control, so at most three new configurations. No per-target capacity choices, 400-iteration branch, min-leaf grid or retrospective early-stopping rule. Freeze its own exact gate before fitting; if timing does not allow confirmation, record `not_run` rather than running it for a leaderboard entry.

For EXP-025, first inspect whether fixed-data HGB seed changes produce different probabilities in this runtime. Do not spend five fits averaging identical deterministic models or introduce row subsampling under the name of a seed change. Any HGB/logistic blend requires its own bounded preregistration with identical feature inputs and earlier-only blend selection. No new blend grid is admitted by this section; without that separate registration and confirmation window, EXP-025 remains Deferred.

### 30.8 Metrics, exact development gates, and deterministic selection

#### 30.8.1 Measurements for every arm

Compute six AP values at each origin from continuous raw HGB probabilities; origin mean AP is their arithmetic mean, and `AP_all4` is the arithmetic mean of the four origin means, equally weighting origins and targets. Compute `AP_mature3` analogously. Do not substitute AP pooled across users/origins. If any report target has only one class, record undefined AP and block the aggregate screen; do not silently drop a target or change its weight. The existing pinned report set has both classes; assert rather than assume.

For each layer/origin/target report `k`, TP, FP, FN, TN, precision at k (`TP/k`), recall at k (`TP/actual positives`), F1, raw AP, predicted-positive fraction and support. Mark undefined precision/recall as null with the denominator; retain the instructor's F1 convention independently. Because k and actual support are shared across feature arms, `F1=2*TP/(k+actual positives)` exposes changes at the actual operating budget. AP elsewhere on the ranking is not enough to select a deployable winner.

Report macro-F1, micro-F1 and combined F1 for each origin and for pooled all4/mature3 confusion counts. Mean weekly combined F1 is a separately named diagnostic and never replaces pooled score. Compare feature arms with H2 and persistence; also report comparisons with REF/P021 for deployment relevance. For unions report both parent comparisons. Show 09-02 separately; it never changes a screen verdict.

#### 30.8.2 Feature-arm screen eligibility

All clauses below are required. This is a **new development-screen heuristic**, not a significance test or a retroactive EXP-012/013 gate. Comparisons use the same rows/labels and the H2 control unless specified. Tolerance for exact zero/positive comparisons is `1e-12`; threshold boundaries use that tolerance.

1. **Validity:** complete scheduled origin/target outputs, feature/source/maturity tests, F0/key parity, protected-input hashes and official-score parity pass.
2. **Ranking magnitude:** `AP_all4(candidate)-AP_all4(H2) >= 0.002`.
3. **Ranking stability:** origin mean-AP delta is strictly positive on at least **3/4** all4 origins; on mature3, mean-AP delta is strictly positive and at least **2/3** origin deltas are strictly positive.
4. **Operating-policy increment:** pooled P021 combined-F1 delta versus H2/P021 is strictly positive on all4 and nonnegative on mature3. P016 is reported, not substituted when P021 fails. No new threshold is fitted to rescue an arm.
5. **Absolute policy screen:** P021 versus P_persist7 satisfies the existing §29.2 screen: positive pooled all4 delta, at least **3/4** strictly positive origin deltas, and minimum pooled leave-one-origin-out delta at least **-0.005**. Leave-one-out recomputes pooled confusion counts; it is not the mean of remaining score deltas.
6. **Target guard:** for no target is P021 F1 more than **0.02 below H2/P021** on two or more all4 origins. Report open_issue separately because H2 already has an AP decline there versus REF. Passing AP cannot conceal this guard.
7. **Operational feasibility:** complete within the registered arm budget and feature memory bound, with final-inference-compatible construction and no candidate-specific missing-row exclusions.

An arm that fails one clause is `screen_rejected` with every failed clause listed. An incomplete/invalid run is `inconclusive` or `blocked`, not rejected for predictive performance. AP improvement with no P021 increment is `ranking_gain_only` in interpretation and is **not eligible** for deployment selection in this campaign. Record every outcome; no silent replacement by a different group definition.

#### 30.8.3 Parent and single-group ordering

Among eligible candidates in the relevant allowed set, sort by descending `AP_all4`; AP differences at most **0.001** are practical ties. Resolve an AP tie by fewer added model columns; if equal, use the fixed order **S, M, R, W, SM, RW** for T*, and **A, B, AB** for R*. Implement ties by first finding the maximum AP, retaining arms within 0.001 of that maximum, then applying column count and ID order; do not use a nontransitive pairwise comparator. No outer October result or continuity-origin value breaks a tie.

Lock T* after S/M/R/W/SM/RW either finish or are formally `not_run` for the time cap. Lock R* after A and triggered B/AB finish. If a scheduled parent study is incomplete, do not infer its missing result; record the shortened candidate set and reason before any downstream fit. This cap-based omission does not permit choosing an omission after seeing that arm's partially favorable origin.

#### 30.8.4 Final development nominee and interaction increment

Eligible SM, RW, TR or TX may enter the final nominee pool only if its P021 all4 pooled score exceeds **both exact parents** by at least **0.002**, its AP_all4 is not below either parent, and its mature3 P021 score is not below the stronger parent. These are added complexity screens; a useful but redundant union remains a reported result. A preregistered synergy union may qualify even when its individual parents fail the H2 screen, provided the union itself passes every §30.8.2 clause.

Select **one** feature nominee from all eligible single groups and qualifying unions: maximize P021 all4 pooled combined F1; treat differences within **0.002** of the maximum as practical ties; among that tie set choose fewer added columns, then higher AP_all4, then lexicographically ascending full arm ID. Store all sorting values. This rule chooses a **complete six-target shared-feature policy**, not separate winners per target. The nominee is a development choice only.

#### 30.8.5 Uncertainty and diagnostic controls

Report all four origin deltas, mature3 sensitivity, per-target changes and maximum weekly loss. A fixed UserID-cluster bootstrap with **1,000 replicates**, seed **20261020**, resamples stable users with replacement, preserving each sampled user's repeated origins/targets; use those multiplicities consistently in AP and confusion counts. Report percentile 2.5/97.5 intervals for candidate-versus-H2 deltas. Label these user-sampling intervals, not confidence about future-week transfer or correction for adaptive selection.

For the final feature nominee only, report **1,000 within-origin/within-target label permutations** and **500 size-matched random edits of persistence**, seed **20261020**, using fixed predictions. These are diagnostic nulls and never rescue a failed gate. All new learned policies are size-neutral relative to persistence by construction; report that fact. The old screen gate's approximately 27% size-neutral null pass rate is why a screen pass alone is insufficient. No scientific claim of independence arises from a small permutation p-value on reused September data.

Use `numpy.random.default_rng(20261020)` separately for each diagnostic family. Iterate all4 origins in ascending UTC order and targets in the fixed order; sort users by stable ID before each draw. For the UserID bootstrap, sample with replacement from the sorted union of all4 user IDs, drawing exactly that union's size; each user's multiplicity weights all its available origin/target rows. Recompute each origin's six weighted AP values and then the equally weighted origin mean; pooled confusion counts use the same multiplicities. Compute candidate-minus-H2 differences on the same draw. If a replicate makes AP undefined for a target, omit that replicate from the AP interval only and report its count; fewer than 950 valid AP replicates means no AP interval is reported. Score intervals can still use all replicates. This does not change the fixed screen verdict.

For permutation nulls, independently shuffle the observed binary labels among users within each origin/target, preserving that cell's prevalence; apply the same shuffled labels to the fixed nominee/P021 and P_persist7 decisions. The statistic is pooled all4 combined-F1 delta versus P_persist7. For size-matched edits, in each origin/target let d be the nominee's number of removals from the persistence-positive set; size neutrality implies exactly d additions. Sample d persistence-positive users and d persistence-negative users uniformly without replacement, flip those memberships, and pool the score on the real labels. This preserves both k and the nominee's edit count. No-edit cells remain unchanged. Report the real delta, null delta distributions, and one-sided Monte Carlo tail fraction `(1 + number(null_delta >= real_delta))/(1 + replicates)`. Nulls are descriptive on reused development weeks, not multiplicity-adjusted significance tests.

### 30.9 Confirmation, candidate freezes, and champion selection

#### 30.9.1 Preserve confirm-plan v0 exactly

The existing manifest `reports/unattended_sessions/s29-unattended-20261018/CANDIDATE_MANIFEST_confirm_v0.json` stays byte-identical. Its primary remains **REF/P021_blend50**; secondaries remain P016_a100 and EXP-014b C1; rule fallback remains P020_multi; P_persist7 remains the default. Continue its own registered primary/secondary rules, Stage 2 obligations and pinned-file checks. D3 has already been accepted: use and record `--holdout-overlap-ack` where required, without asking for the same decision again. The manifest's older text calling D3 open is historical, not a fresh blocker.

Do not replace its REF with H2, a feature nominee, or a "best per-target" mixture, and do not widen its grids. Its primary remains decision-bearing within v0; diagnostic secondaries cannot replace it opportunistically. A future new-candidate manifest is separate and references v0 as comparator.

#### 30.9.2 Exact maturity and exposure ledger

| Origin ID (UTC) | Taiwan target start | Label window ends (Taiwan) | Minimum source latest-observed timestamp | Arithmetic eligibility (Taiwan) |
|---|---|---|---|---|
| 2026-10-07T16:00:00Z | Oct 8 00:00 | Oct 15 00:00 | 2026-10-20T13:00:00Z | Oct 20 21:00 |
| 2026-10-14T16:00:00Z | Oct 15 00:00 | Oct 22 00:00 | 2026-10-27T13:00:00Z | Oct 27 21:00 |
| 2026-10-21T16:00:00Z | Oct 22 00:00 | Oct 29 00:00 | 2026-11-03T13:00:00Z | Nov 3 21:00 |

The threshold is `S + 7 days + 5 days 21 hours`. The approximate Oct 21/28/Nov 4 checkpoints allow for actual releases; the calendar and a repository commit date do not prove coverage. A source 31 seconds short is still short. A gate that passes this threshold is an operational maturity convention, not proof that every late event has arrived.

Maintain `exposure_ledger.csv`: candidate ID, freeze UTC timestamp/hash, origin ID, earliest inspection time of that origin's labels/scores or equivalent outcome-bearing analysis, source SHA, maturity result, eligibility, and reason. Read/access events from earlier chats and artifacts count as exposure; absence of a newly opened CSV does not reset the history. September is development for all candidates. An October origin whose outcomes influenced a candidate's features, parameters, selection rule or parent choice cannot confirm that candidate.

**Freeze before outcome inspection, not merely before running a scorer.** New candidate manifests must be recorded before reading the corresponding October outcomes. Fitting a frozen rule on the training information allowed at an origin is permitted; choosing its feature/weight configuration from that origin's outcomes is not. Historical origins can predate the calendar freeze while their labels remain uninspected; describe such a result as **label-blinded retrospective confirmation**, not a model that was deployed before the target week. Reserve "fully prospective" for a decision rule fixed before the target-week start. Disclose any raw outcome-bearing analysis that compromises blinding. Never claim that writing a preregistration after looking at an outcome restores independence.

The first v0 confirmation may be inspected while new feature development continues, but it then becomes development information for any later-informed new candidate. New feature/H2 candidates do **not** use the 10-07 result as their confirming evidence in this plan. Their intended confirm set is **10-14 and 10-21**, provided those outcomes have not been inspected before the candidate freeze and remain otherwise eligible under the ledger.

#### 30.9.3 Shortlist cap, freeze dates, and new manifest

At freeze #1, retain at most **three selectable learned complete policies**: (1) v0 REF/P021 incumbent; (2) H2/F0/P021; (3) the single §30.8.4 feature nominee/P021, if one qualifies. P_persist7 and P020_multi are separately retained comparators/fallbacks. Existing v0 diagnostic/Stage 2 rows are outside this shortlist and remain diagnostic; the cap is not permission to choose among all of them later.

**October 25** is the target shortlist checkpoint, not an assignment deadline. The latest planned slip is **October 26, 18:00 Taiwan**, with a dated reason and before any 10-14 outcome exposure. Do not postpone inspecting or producing the unchanged v0 result merely to manipulate a new candidate's status; isolate new-candidate research from outcome exposure where practicable, otherwise record the lost confirm opportunity. A candidate not validly frozen before 10-14 exposure has only one scheduled fresh origin left and cannot replace the incumbent under §30.9.4. Its September result can still be reported as exploratory.

Create a separate planned `CANDIDATE_MANIFEST_feature_v1.json` with exact ordered features, H2 policy, resolved estimator/defaults, decision rules, source adaptation, parent/nominee choices, all trial IDs, code/config hashes, shortlist, confirm origins, exposure state, baseline rules, tie-breaks and this section's replacement gate. Freeze it **before** the first new-candidate confirm outcome is inspected. Hash the plan/executable specs; retain v0's original bytes. No per-target feature choices or confirm-driven switch from P021 to P016.

Freeze #2, targeted **November 2**, may remove failed candidates and finalize packaging, environment and commands. It does **not** admit a new feature/architecture/parameter configuration into final selection if that leaves fewer than two eligible confirm origins. A substantive revision gets a new ID/freeze and resets its eligible evidence; it cannot inherit its parent's confirm wins.

#### 30.9.4 Fixed champion rule for the new shortlist

This is a forward amendment for the separate feature shortlist, registered before its confirm outcomes. It does not alter the existing v0 within-manifest rule.

1. Score every eligible frozen shortlisted policy and the fixed comparators on identical origin/user/label keys. Publish all eligible confirm results. For each policy, the absolute confirmation gate remains positive pooled combined-F1 delta versus P_persist7 and **no origin delta below -0.010**. Do not pool different eligible origin sets to rank challengers.
2. A new H2 or feature challenger may replace the v0 incumbent only with **at least two common eligible confirm origins**, positive combined-F1 delta versus the incumbent at each of those origins, pooled delta versus incumbent **at least +0.002**, the absolute persistence gate above, and no target F1 decline versus incumbent greater than 0.02 on two or more common origins.
3. Intended common set is 10-14/10-21. Evaluate the incumbent on exactly that subset for replacement comparisons, even though it also has a v0 10-07 result. The 10-07 result remains part of its own v0 history; it cannot be selectively dropped from the published record.
4. Among challengers passing replacement, choose the largest pooled score on the same common set. Differences within **0.002** of the maximum are practical ties; choose fewer added columns, then fewer complete learned components, then lexicographic policy ID. If no challenger passes, keep the incumbent if it passes its v0 absolute confirmation gate.
5. If the incumbent fails, choose a new challenger only if it satisfies clauses 1–2, including sufficient common origins. Otherwise use P020_multi only if it has at least two mature confirm origins and passes the same absolute gate; if not, use **P_persist7**. No gate bypass because a September score is high.
6. Missing maturity, access failure, invalid keys, or fewer than two common eligible origins is **inconclusive**, not a confirmation win. No architecture/feature changes after inspecting confirmation may be evaluated again as if the same evidence were fresh. If the final choice uses additional information beyond this rule, record a deviation and remove any independence claim for that choice.

These are project decision tolerances, not statistical guarantees. Two October weeks still provide limited evidence; publish individual deltas and uncertainty honestly. The one-week absolute catastrophe guard is not replaced by a favorable pooled score.

#### 30.9.5 Official roster and the downgraded holdout

EXP-026, once the actual roster is supplied, records exact stable-ID membership, overlap with the historical proxy, source visibility, empty-history users and feature coverage without future labels. Primary September/October proxy comparisons remain on their frozen populations. A second table may rescore frozen policies on the official roster/intersection when historical labels/features are valid, but it is a separately named population sensitivity, not a silent replacement of the primary confirmation gate. Do not remove roster users because a feature is unsupported; use each group's registered zero/support behavior. Do not select features based on which roster subgroup retrospectively scores best.

The final inference policy is applied to **every official test user**, with ranks and k computed over that exact roster under the frozen rule. The roster adaptation rule must be in the candidate manifest before using roster-aligned scores for any deployment decision. If membership requires a substantive unregistered model/policy change, record it as such rather than reusing a confirmation label.

The Oct 12 holdout stays a **non-independent, optional, one-shot sanity check** under D3. Its scores cannot select a feature, override the champion rule, break a tie, or be described as untouched independent evidence. Preserve its original Monday-midnight design, cohort and one-shot hygiene if executed. No need to run it merely to produce another score.

### 30.10 Required verification and release-drift work

#### 30.10.1 Feature and temporal checks before fitting

- Independent hand/oracle calculations for each new formula, including empty history, one event, tied timestamps, window-left boundary, exact-cutoff exclusion, and events with the same ID in different source tables.
- For each group, compare full-source features with features built after deleting every source row with official visibility time `>= C`; require unchanged keyed values. Inject future rows and later-known field changes; earlier features must remain unchanged. Retain original first-seen/event-time distinctions.
- S18 exact complete-bin handling; M12 EW equality with the frozen rule helper; recent/prior interval disjointness and count conservation; X6 exact pair/block counts and entropy guards; R24/W24 parity with their accepted builders.
- Relational many-to-one joins, actor/repository visibility, disjoint unresolved-reason counts, preservation of unresolved events in F0, focal-self exclusion, deduplication across neighbor repositories, and unchanged neighbor metrics when only focal volume is inflated for fixed exposure.
- At every built origin: row/column schema, exact F0 values/order, stable-ID equality, no duplicate keys, feature count/dtypes/finiteness, missing/zero coverage, min/median/p95/max and compute cost. Save five deterministic user/origin traces per group, selected by ascending stable ID within zero/low/high support strata; no handpicked favorable predictions.
- Cache keys include source file hashes, builder code/group-spec hashes, origin/cutoff, population definition and ordered feature groups. Label cache keys additionally include target semantics and label_as_of. A matching filename alone is insufficient; never copy feature caches across release SHAs without authenticated keys.

#### 30.10.2 Output completeness, scoring, and isolation

Before executing fits, derive the exact expected `(arm, report_origin, target, UserID)` set for mandatory and conditional arms from the freeze. Save it independently of observed output. Detect omitted whole fits, origins, targets and policies; iterating only over rows that happen to exist is insufficient. Conditional decisions and `not_run` statuses must reconcile with that schedule.

Independently recompute AP, both ranking layers, all confusion counts, weekly/pooled official scores, leave-one-origin-out screens, target guards, parent choices and final nominee sorting from saved predictions. Scorer parity tolerance is `1e-12`; discrete keys/decisions must match exactly. Test all-negative targets with the instructor's `F1=1` convention separately from undefined precision/AP diagnostics.

Replace current report labels with random values and require unchanged features, fitted parameters, raw probabilities and binary decisions for that origin. In this development campaign, selection tables may change because the screen legitimately reads development labels; do not demand an impossible invariant there. In confirm mode, all frozen feature/parent/parameter choices must remain unchanged and only metric/gate reports may change. Preserve later-fold access to earlier labels only according to their recorded as-of clock.

#### 30.10.3 Appropriate audit depth

For each feature screen: targeted formula/temporal tests, exact baseline parity, official metrics, complete schedule, and an independent saved-prediction selection check. Do not repeat a full archive-wide model audit after every small group. Before promoting the single nominee: independently trace its raw-source features/labels, fitted bundles/weights, complete schedule, decision outputs and selection. EXP-030 then audits the final chosen complete deployment policy and fallback. The original EXP-013 retains its own mandatory independent auditor regardless of this lighter campaign-screen workflow.

Run tests appropriate to changed modules plus preserved-core regression once per material implementation change. Record exact commands, interpreter path, versions, pass/fail/skip counts and basetemp. Historical test counts in §§2/29 are not evidence for new code. Do not install or replace shared runtime packages while a controlled paired run is in progress. A numerically incompatible runtime blocks reproduction rather than justifying relaxed parity.

#### 30.10.4 EXP-028 exact distinct-release comparison

Preserve the pinned source in a separate snapshot path before acquiring a newer instructor release. Record actual SHAs and eight-file hashes; never destroy the only reproducible baseline or use network data outside the instructor source. Match entities across releases only by stable GitHub IDs, never internal PK/FK row numbers.

For the same five September origins and each currently eligible October confirm origin, create two explicitly named label comparisons. **fixed_asof** applies one common `label_as_of=min(latest_old,latest_new)` to both releases to isolate record/field revisions at a common observation cutoff. **release_latest** applies each release's own latest timestamp under the same target/window semantics to measure additional delayed observations as coverage advances. Do not conflate these panels. For matched users report 0→1, 1→0, unchanged positives/negatives, missing stable-ID mappings and class prevalence per target, with both exact as-of timestamps. Compare F0 and shortlisted feature vectors at fixed historical cutoffs on matched users. Export coverage maxima per source and observed collection-delay summaries; a global latest timestamp alone does not establish equal table completeness.

October label-drift tables are outcome-bearing exposure. Before reading them, either freeze all candidates intended to use that origin for confirmation, or restrict the pre-freeze drift study to September and input-only schema/coverage checks. Log the October table's inspection in the exposure ledger even if it reports only transition counts rather than per-user labels. This rule also applies to roster-aligned October score diagnostics.

Do not choose whichever release gives the best score, average labels across snapshots, or silently shorten/extend the training buffer. A demonstrated source/label defect requires a versioned pipeline amendment and a reset of affected comparison/confirmation claims. The final refit may use the final permitted snapshot under the frozen recipe, with actual training labels and source provenance recorded.

### 30.11 Planned implementation boundaries, artifacts, and handoff contract

Use shared builders/evaluation helpers for reusable logic and experiment folders for schedules/manifests/results. Do not duplicate F0, target semantics, scoring or ranking code in each experiment. Preserve pinned historical modules; new extensions should be isolated, with compatibility tests if an adapter is necessary. In particular, importing old helpers must not give a new runner permission to overwrite old caches/runs or to load confirm labels during a feature build.

Planned paths, to be created during implementation (not claimed present by this plan edit):

| Artifact / location | Required contents |
|---|---|
| `docs/exp/FEATURE_CAMPAIGN_20261020_PREREGISTRATION.md` | Executable-spec expansion of §30, dated amendments, actual freeze and scientific-change history |
| `configs/feature_campaign_20261020.json` | Exact expanded origins, ordered F0/group columns, HGB/defaults/H2, source/code locks, arm triggers, metrics/gates, resource and output schedule |
| `src/feature_campaign.py` | S18/M12/X6 feature construction and explicit group assembly; adapters to accepted temporal/relational builders; no model/label selection side effects |
| `experiments/exp024_short_horizons/` | S implementation entry and immutable run artifacts |
| `experiments/exp031_smooth_momentum/` | M entry and run artifacts |
| `experiments/exp032_temporal_reevaluation/` | R/W entries and run artifacts |
| `experiments/exp033_relational_ranking/` | A/B/AB entries and run artifacts |
| `experiments/exp034_cross_target/` | X entry and run artifacts |
| `experiments/exp035_feature_interactions/` | SM/RW/TR/TX entries, parent locks and run artifacts |
| `reports/feature_campaign_20261020/` | Baseline/source audits, error analysis, trial/exposure ledgers, common comparisons, shortlist and frozen new-candidate manifest |
| `tests/test_feature_campaign.py` and experiment-specific tests where needed | Boundary/oracle/group-schedule/isolation tests; no tests that merely assert a copied implementation formula |
| `reports/verify_feature_campaign.py` | Independent saved-output, schedule, metric, gate and selection reconstruction; do not import the selection implementation being checked |

A shared runner is acceptable; thin experiment entries may call it with explicit IDs/arms. They must still export immutable per-ID manifests and one campaign-wide trial ledger. Do not create six near-identical full pipelines just to match folder names. If the final module layout differs, record a score-free mapping before fitting; scientific inputs/outputs above do not change.

Every executed experiment records at least:

1. `manifest.json`: source SHA/file hashes, Git HEAD plus dirty-file hashes, interpreter/packages, code/config/plan freeze, ordered columns, timing/population, expected cells, seeds, and resource budgets.
2. `feature_manifest.csv`: one row per instantiated new column, catalogue ID/variant, formula, source/field/time predicate, dtype, support, overlap classification, cost and experiment ID.
3. `folds.csv` and `training_membership.parquet`: actual origins, C/A/label_as_of, eligible user keys, row weights and training-row digests. Do not infer these from a final score table.
4. `predictions.parquet`: stable keys, y, raw probabilities, persistence flags, fixed-layer decisions, per-target EW score/k, source/arm IDs; labels remain unavailable to feature selection in confirm mode.
5. `ap_by_origin_target.csv`, `metrics_by_origin.csv`, `per_target_metrics.csv`, `pooled_metrics.csv`, `screen_gate.csv`, `parity.csv`, `cost.csv`, and complete error/coverage tables.
6. `selection.json`, conditional `parent_selection.json`, campaign `trial_ledger.csv`, and `exposure_ledger.csv`, including every failed, skipped, repaired or incomplete attempt. Never overwrite a failed run to hide it.
7. `findings.md`: measured outcome, failed/passed clauses, primary and parent contrasts, limitations, acceptance status, and reproducible commands with real paths. Append the completed evidence to `EXPERIMENT_LOG.md` before changing this plan's status.

Keep compact metrics/configs/manifests/source versioned; preserve large models/per-user data locally according to repository ignore policy. The working tree currently contains substantial uncommitted work: capture content hashes, and prepare a clean reproducible source commit before final delivery rather than treating old Git HEAD as the full implementation.

### 30.12 Revised calendar, priorities, and deliverable protection

All dates/times here are **Asia/Taipei**, and are targets rather than claims of execution. Source coverage and outcome-exposure rules take precedence over a nominal date. Freeze #1 is a research checkpoint; **November 11 23:55** is the recorded course submission deadline.

| Target window | Work to execute | Exit evidence | If the work does not fit |
|---|---|---|---|
| **Oct 20–21** | Source preservation, runtime/REF/H2 preflight, error/exposure ledger; EXP-028 second release; first unchanged v0 confirmation once maturity passes | Baseline parity, source hashes/coverage, first v0 row or explicit data blocker | Do not invent October labels or score immature data; continue September feature QA |
| **Oct 20–23** | Implement/freeze S18/M12; run S, M, SM; reconcile original EXP-013 auditor and preflight | Core feature block and exact EXP-013 readiness report | Drop optional capacity/bagging and X6 first; retain core/Q2 budget |
| **Oct 22–25** | Execute/accept original EXP-013; run EXP-033 A and triggered B/AB; bounded R/W/RW; eligible TR/TX/X as time permits | Log all executed/skipped arms; lock T*/R*/nominee; complete shortlist | Mark omitted arms not_run with budget reason; do not change gates or hide partial results |
| **Oct 25, target freeze #1** | Freeze up to three complete learned policies and two fixed rule comparators | New manifest with hashes and label-exposure eligibility | One justified slip to **Oct 26 18:00**, before 10-14 exposure; later new policies remain exploratory |
| **Oct 27 21:00 arithmetic maturity; checkpoint Oct 28** | Next permitted release; unchanged v0 and eligible frozen shortlist confirmation at 10-14 | Paired results with identical common keys, maturity/exposure records | No refitting the feature definition after seeing scores; wait if actual release is short |
| **Oct 28–30** | EXP-026 official roster audit; final feature/inference coverage; prepare reproducibility README and Q1 graphs | Roster sensitivity separate from primary confirmation; complete graph/pipeline drafts | Submission compatibility outranks optional research; no new roster-driven feature search |
| **Oct 29–Nov 2** | Packaging, nominated-policy audit, source cleanup, fallback rehearsal; optional 023/025 only if their earlier timing gates were met | Freeze #2 by Nov 2, preserving prediction recipe and shortlist | No new candidate can inherit a parent's evidence; fewer than two fresh origins means no replacement |
| **Nov 3 21:00 arithmetic maturity; checkpoint Nov 4** | 10-21 confirmation for eligible frozen candidates; fixed champion/fallback selection | Log common-origin scores and exact §30.9.4 decision | If inconclusive/failed, retain confirmed incumbent or registered rule fallback |
| **Nov 5–7** | EXP-029 timed full rehearsal; EXP-030 audit/freeze of final recipe and fallback; optional downgraded holdout | Real commands, cost, valid CSV and clean-checkout reproduction | Eliminate fragile optional components; protect final execution window |
| **Nov 5–10** | Finish English slides, required Q1 schema/entity graphs, Q2/Q3 measured ablations, Q4 pipeline; student records video | At most 10 slide pages; 8–12-minute student-recorded video; figures traced to log | Use valid negative findings; do not wait for an unrun catalogue branch |
| **Nov 11 before 21:00** | Check source commit/environment/roster/upload names; freeze commands and rollback choice | Final checklist and tested fallback; slides/video already ready | No new dependencies, architecture, feature or threshold changes |
| **Nov 11 21:00–23:55** | Obtain final allowed instructor snapshot; verify actual coverage; refit frozen policy on permitted available labels; infer/validate exact roster CSV; submit | Full SHA/hash provenance, official format validation, submission artifacts | If champion path fails the rehearsed validity/runtime checks, use the rehearsed fallback immediately |

Q1 requires the full eight-table schema plus entity graphs for GitHub users **22996444** and **28983449** over **2026-09-15 00:00–2026-09-17 00:00 UTC**. Q2 must show actual cross-table mechanisms and measured relational results or limitations; Q3 temporal mechanisms and measured ablations; Q4 the actual implemented end-to-end pipeline. Prepare these in parallel with research as ordinary work scheduling, without implying that agents have been spawned. The final snapshot's stated observation endpoint is approximately Nov 11 20:00 Taiwan; target start remains Nov 12 00:00. Do not manufacture observations for the four-hour gap.

November 5–7 freezes the **recipe**, tested environment and commands, not a claim that the final November 11 refit has already happened. If a real bug is found after the code freeze, preserve the old artifact, document the change and its prediction impact, rerun the relevant checks, and reassess which confirmation claims survive. Never waive a correctness bug merely to obey a date, or call a changed predictor unchanged because it matches one dry-run origin.

### 30.13 Catalogue coverage and explicit deferred scope

| Catalogue group | Current evidence | What this campaign adds | Explicitly outside this campaign |
|---|---|---|---|
| G01 baseline/multiscale | F0 already contains a subset | S18 short horizons and complete-bin count | All remaining windows/transforms Cartesian product |
| G02 rhythm | Exact R24 tested in old protocol | Exact R24 follow-up and one RW union | Gap CV/p90, alternative K, burst thresholds, seasonal sweeps |
| G03 weekly state | Exact W24 tested in old protocol | Exact W24 follow-up and RW | Smoothed supervised transitions, survival/hazard models, broad lag grid |
| G04 momentum/decay | F0 count delta; EW score used in decision blend | M12 and SM | All decay constants, ratio variants and calendar hypotheses |
| G05 cross-target | F0 includes all target histories and activity-type count | Conditional fixed X6 and TX | Exhaustive pair/lag/workflow search |
| G06 repository-self | F0 simple breadth; original A9 built | Original EXP-013 closure and A9 ranking follow-up | Extra organization/metadata/continuity definitions |
| G07 repo-neighbor | B7 built, no fitted result at review | Conditional original B and current B7/AB follow-up | Unrestricted neighborhood depth/degree tuning |
| G08 PR workflow | No dedicated group ablation | None | Branch/draft/lifecycle features; revisit only after separate field-availability audit and registration |
| G09 issue workload | No dedicated group ablation | None | State/body/assignee workload features requiring new availability assumptions |
| G10 review collaboration | No dedicated group ablation | None | Reviewer-author and thread-pair feature search |
| G11 handcrafted two-hop | No dedicated ablation | None | Two-hop/org/graph-statistic expansion |
| G12 cohort regime | Twelve columns tested in EXP-014a's old policy setup | Retain measured result and caveat | New regime models or tuning |
| G13 metadata/text/auto | No dedicated ablation | None | Titles/body embeddings, text vocabulary, DFS, automated synthesis |

The EXP-015 GNN pilot is a completed separate relational-learning recipe, not completion of G08–G11 handcrafted groups. The absence of these tests remains a limitation, not a zero-gain result. These branches can be reconsidered in a later separately registered study only with a specific residual-error hypothesis, legal field timing, available budget and a new valid confirmation opportunity. They do not block homework completion.

### 30.14 Stop rules, deviations, and acceptance checklist

**Stop/skip rules:**

1. Stop fitting for unexplained source/key/F0/runtime parity changes, future information, illegal joins, incomplete schedules or a report-label isolation failure. Preserve the failure; repair and re-freeze before new candidate scores.
2. Stop expanding groups once the 12-new-arm/resource budget or valid confirmation window is exhausted. Record all unrun work. No tuning another seed/window because a result is close to a gate.
3. Keep original EXP-013 and the core S/M/SM block ahead of optional X6, capacity and bagging. If only S/M and the original relational study fit, freeze that evidence honestly rather than delaying submission work.
4. Do not extend a rejected single group into new unregistered combinations. Only the explicit SM/RW exceptions and conditional TR/TX slots are allowed.
5. No substantive new selectable predictor after losing the opportunity for two common eligible confirmation origins. November 2 is a final packaging/shortlist deadline, not an invitation to introduce an unconfirmed predictor.
6. A meaningful source/algorithm bug fix may require a new candidate and reset confirmation eligibility. A purely textual/reporting fix may preserve predictions, but record and verify that distinction.
7. No result in the optional Oct 12 sanity check or late October 28 outcome week may retune the final policy. Do not inspect November target-week outcomes to make the November 11 forecast.

**Acceptance checklist for the next executor:**

- [ ] Acknowledge §30 as the controlling forward plan; preserve old protocol/result artifacts and v0 bytes.
- [ ] Source SHA/eight-table hashes, runtime, exposure ledger and REF/H2 parity recorded before new fits.
- [ ] Exact column manifests, schedule, weights, gates, resources and expected-output keys frozen before scores.
- [ ] S/M/SM and original EXP-013 receive the protected work budget; conditional arms are executed or explicitly skipped by their stated trigger.
- [ ] New group correctness, temporal invariance, join/self-exclusion, cache provenance and complete schedule checks pass.
- [ ] Every attempted arm, repair, failed guard and not_run reason appears in the ledger; scores are measured rather than copied from illustrative tables.
- [ ] Independent metrics/decisions/selection check agrees; nominee/parent locks follow the stated tie-breaks.
- [ ] Shortlist contains no more than three selectable learned policies and is frozen before its eligible outcome exposure.
- [ ] Confirmation is run only on actual mature source coverage, with D3 acknowledgment and common-origin replacement comparisons.
- [ ] Original EXP-013 and the final nominee/final deployment receive their appropriate independent audits.
- [ ] Public reproducible source, exact-roster CSV, Q1 graphs, Q2/Q3 evidence, Q4 pipeline, English slides and student recording are completed before the deadline.
- [ ] Append actual findings to `EXPERIMENT_LOG.md` first, then update this plan's status. A plan edit alone never checks an execution box.

### 30.15 Immediate handoff: what to do first

The next implementation session should start with the source/exposure/baseline preflight, exact S18/M12 builder and tests, and original EXP-013 audit reconciliation. Its first deliverable is a score-free machine-readable campaign freeze and an EXP-013 readiness report, not a new claimed improvement. Run unchanged v0 confirmation as soon as actual permitted coverage qualifies, with the prior D3 decision recorded. Then execute S, M and SM, followed by the gated and budgeted relational/temporal work in the register. Use the exact artifact/gate definitions above; ask for clarification only if a real source/protocol conflict cannot be resolved within the registered amendment process.

**Status at this amendment:** new EXP-024 specification and EXP-031–035 are planned only. No candidate from this campaign has been fitted, selected, confirmed or promoted. The submission default remains **P_persist7**.

## 31. Controlling research program: stronger neural and relational models — October 22, 2026

### 31.1 Decision, authority and immutable history

The user explicitly requested stronger deep learning and broader research with approximately three weeks remaining, followed by this detailed plan update and a course-data refresh. The objective for the next research phase is to discover better representations, learners and complementary predictions. October 25 is a progress checkpoint, **not a research stop or final architecture freeze**. Development continues into early November. Administrative deliverables are maintained alongside research; they do not take its primary compute budget now.

This section controls future work when older sections conflict. It opens new EXP-036–041 studies; it does not reopen EXP-015 under easier success criteria, reinterpret rejected §30 features as successful, or modify original EXP-013/confirm-plan v0. A model can be scientifically informative without being eligible for final selection. Planned settings below are not claims that dependencies, memory, runtime or predictive gains have already been verified.

The complete plan immediately before this amendment is preserved at [`EXPERIMENT_PLAN_before_research_amendment.md`](../reports/research_refresh_20261022/EXPERIMENT_PLAN_before_research_amendment.md), SHA256 `1b69a69d549fd3e2d474b6c97b3d3e272e8674e57fa316072d469b3d2c8cc046`. Frozen campaigns must authenticate that copy rather than require the living plan to remain unchanged forever. Preserve their existing manifests, source/config hashes and exposure records. An old launcher pointing at the living plan needs a dated provenance-path amendment before reuse; do not silently edit a frozen hash.

### 31.2 Evidence and the questions it actually supports

On the fixed September all4 development pool, H2/F0/P021 has pooled combined F1 **0.5495393360051135**. The seven accepted feature arms S/M/SM/A/R/W/RW all reduce this operating score; small AP gains from some arms do not meet their registered ranking/stability/operating gates. B/AB were skipped by the original condition, not evaluated as negative. The catalogue in `FEATURE_ENGINEERING.md` remains only partly explored; exhaustive combinations are neither completed nor required before trying a different learner.

EXP-015's bounded pilot gives mean AP MLP0 **0.4666**, GNN1 **0.5063**, GNN1T **0.5126**, REF **0.5312**, H2 **0.5375**. It tested small networks, two seeds, a single training recipe and a graph that collapses repeated typed edges into count/recency summaries. That is evidence against those particular configurations. It does not establish a ceiling for tuned neural tabular models, explicit activity sequences, composite relational reasoning, graph transformers or ensembles. Its graph–MLP gain also cannot be attributed solely to topology when inputs differ.

New hypotheses are: H36, a stronger neural tabular learner improves on the pilot when inputs/training opportunity are controlled; H37, event timing within the aggregate windows contains useful signal; H38/39, better relational computation adds value beyond user histories and a strong node-only network; H40, input representation rather than only architecture explains part of the graph gap; H41, neural predictions improve an HGB blend even when the neural model alone is weaker. Each gets a separate result, including negative results and inconclusive resource failures.

### 31.3 Register, dependencies and first five runnable experiments

All new experiments below are **Pending** at registration. Completion requires actual outputs, the appropriate verification, and a measured log entry first. A source audit, dependency installation or smoke test alone does not complete a model study.

| Order / ID | Work package | Required predecessor | Deliverable / scientific contrast |
|---|---|---|---|
| Preparation | Source archive, development replay and temporal/exposure checks | None | Verified source manifest, exact F0/H2 control parity, common input/label keys; no reserved outcome scoring |
| 1, EXP-036-R | Strong residual MLP on F0 | Preparation | Six-config search and full-window finalist replay; upgrade the weak node-only reference |
| 2, EXP-036-T | TabM on F0 | Same preparation; independent of MLP success | Six-config search and full-window finalist replay; same inputs/objective as EXP-036-R |
| 3, EXP-037-G | Daily-sequence GRU plus F0 | Tensor correctness + EXP-036 reference | Four-config search, order controls and same-input flattened baseline |
| 4, EXP-037-C | Daily-sequence TCN plus F0 | Same sequence tensors; independent of GRU success | Four-config search; compare temporal inductive biases |
| 5, EXP-038 | RelGNN-style composite relational model | Graph/code preflight + strong node-only control | Four-config search, self-only/no-neighbor contrasts, paired topology claim |
| EXP-039 | RelGT-style relational transformer | Graph/token preflight; reuse correct graph inputs, not a requirement that EXP-038 wins | Four-config search; attention versus message passing under declared resource limits |
| EXP-040 | Event-preserving graph representation ablation | Select one fixed implemented relational architecture by the development rule below | Collapsed-edge versus event-preserving input, with unchanged model/training settings |
| EXP-041 | Neural–HGB ensembles | Any complete accepted full-window neural replay | Fixed blend grid and chronological weight-selection replay; do not wait for every architecture |
| EXP-037-X, conditional | Compact sequence transformer | Explicit trigger in §31.6 | Four-config follow-up; never run merely because the name is newer |

The first five are five scientific work packages, not five single optimizer launches. EXP-036-R/T share one experiment because their common question is neural tabular capacity; EXP-037-G/C share one sequence study. Run the stronger tabular work first to establish a credible reference, then sequence models, then relational models. Prepare graph/token caches during CPU availability where useful. EXP-041 can consume accepted predictions between GPU jobs. Do not serialize the whole program behind original EXP-013 or behind a requirement that hand-engineered neighbor features win.

### 31.4 Shared source, clocks, labels and comparisons

**Two explicitly named source lanes.** Primary architecture comparisons use preserved instructor SHA `506d860693df60a698b68aa63a8edb4a984c8e1b`. After the successful refresh, its full Git checkout is at `assignment2_backup/506d860693df60a698b68aa63a8edb4a984c8e1b_20261022/assignment2/`; explicitly pass its `data/db/` as the research data directory and assert the loader's source SHA. The eight-table mirror at `reports/feature_campaign_20261020/snapshots/506d860693df60a698b68aa63a8edb4a984c8e1b/data/db/` remains available for hash verification, but is not itself a full instructor Git checkout. Verify the original startup-audit table hashes. This retains direct compatibility with accepted September controls. Active `assignment2/` now contains release `2053302ba75751be975b9e779f567557b66f4071` for current coverage, eventual roster and deployment work. Refreshing that directory does not migrate the research silently; a runner's default active-data path is not appropriate for the old-source primary comparison.

New-release sensitivity is a separate matched replay: rebuild **both** H2 and the nominated neural system on the same refreshed snapshot, with identical origin/user/label keys within that release. Save changed/missing historical events, cohort membership and label revisions. Never subtract a new-source neural score from an old-source H2 score and call it an architecture effect. No cross-release join uses integer row PKs; join users through `github_user_id` and repositories through `github_repo_id`. Hash all eight tables and source evaluator/docs at every freeze.

**Common outer design.** Retain six-target order, eligibility populations, 56 training origins, C_s, A_S, feature-visibility and label semantics from §30.4. Report all4 (`09-09/16/23/30`, all at 16:00 UTC), mature3 (`09-09/16/23`) and continuity-only `09-02`. September 30 retains its provisional designation on the old source. All4 and mature3 are development, already repeatedly examined; they are not new independent test sets. Reproduce REF, H2, persistence and H2/P016/P021 on the original source before model fitting. Exact row/target keys and probability values in the accepted runtime are required for control reproduction; metric agreement alone is insufficient.

**Neural training opportunity.** Final reported neural predictions use all 56 daily origins `S-63d ... S-8d`, like H2, with labels reconstructed as of `A_S=S-3h`. Models use six sigmoid logits, weighted binary cross entropy averaged equally across six targets, no class weights, no oversampling and the same H2 row-weight formula/mean normalization as §30.4.3. This prevents an apparent neural deficit caused only by withholding its latest training origins. Missing targets are a build error, not dropped loss terms. Do not add raw user IDs, repository IDs, report date, eligibility masks or target-derived predictors to F0.

**Purged inner epoch/config selection.** For each report S, the seven latest inner validation origins are `S-14d ... S-8d`. Let V be the first of those origins minus three hours. Inner fitting origins are the members of the 56-origin schedule with `s+7d <= V`: exactly **42** origins `S-63d ... S-22d`; seven origins `S-21d ... S-15d` are purged. Build inner-fit labels as of V and validation labels as of A_S; fit preprocessing only on inner-fit rows. This stricter inner split is a new-study choice and differs from EXP-015's 43/6/7 split. Historical inner-validation labels can be late-complete as of the actual outer decision time; disclose that this inner selection is an offline selection procedure, while the outer prediction remains as-of A_S.

Each configuration is searched with seed 42 on the two fixed **earlier tuning origins** `2026-08-19T16:00:00Z` and `2026-08-26T16:00:00Z`, using the same origin-relative inner split and population definitions. They are not members of all4 or the continuity report. Rank configurations by the arithmetic mean of their two inner-validation mean-six-target AP values; ties within 0.0005 use fewer trainable parameters, then lexicographic config ID. This prevents a shared configuration for the first September report from being selected using a later September week's training/validation labels. Outer September scores are not used to change a config grid. Lock the selected config before generating its four outer report scores. For each outer report and each finalist seed, choose the earliest best epoch on that report's purged inner validation, then refit from initialization on all 56 origins for exactly that epoch count. Refit preprocessing on those 56 permitted training origins. Inference uses the refitted model; save both inner and refit manifests. Do not train the final model on only the inner 42 origins and compare it with full-data H2 without labeling the mismatch.

**Shared optimization defaults.** AdamW; batch size 2,048 tabular/sequence examples, or 256 seed users for sampled graphs; weight decay 0.0003; max 128 epochs; patience 12; minimum AP improvement 0.0001; gradient norm clip 1.0; no learning-rate scheduler; seeds 42,43,44 for full finalists; arithmetic mean of their probabilities. Search uses seed 42 only. Keep a deterministic origin/row schedule and normalize H2 weights over the applicable fitting rows, not independently per origin. Log achieved epochs, validation trajectories, loss, norm, wall time, VRAM/RSS and parameter counts. Graph batches may be reduced under the explicit OOM rule in §31.11; effective gradient batch must be recorded. Finite/saturated outputs, train–validation gaps and classwise behavior must be checked before interpreting a loss curve.

### 31.5 EXP-036 — stronger tabular neural learners

Use exactly the frozen 59 F0 columns and common populations/labels. For both learners, leave binary columns unchanged; apply `sign(x)*log1p(abs(x))` to each other continuous column, then subtract its fitting mean and divide by its fitting population standard deviation (zero standard deviation becomes 1); clip standardized values to [-8,8]. Freeze the binary/continuous column lists in the score-free config. Reject unexpected nonfinite source values rather than invent imputation. This common deterministic preprocessing is part of the new neural recipe. HGB retains its accepted raw F0 inputs; any advantage is a whole neural recipe advantage until the preprocessing ablation below.

**Residual MLP grid, six configs.** Input projection to width w; three pre-normalized residual blocks, each LayerNorm → Linear(w,2w) → GELU → Dropout(0.1) → Linear(2w,w) → Dropout(0.1), added to its input; final LayerNorm and Linear(w,6). Cartesian grid `w in {128,256,512}` and `lr in {0.0003,0.001}`. All other settings are shared. This is the strong node-only reference. Do not keep the pilot's weak 64-wide MLP as the only tabular comparator.

**TabM grid, six configs.** Use the authors' official implementation from [yandex-research/tabm](https://github.com/yandex-research/tabm), initialized from scratch. Pin its upstream commit/license/API before implementation; save the complete resolved constructor and code patch. Grid `width in {128,256,512}` and `lr in {0.0003,0.001}`, with three backbone blocks, ensemble size 16, dropout 0.1 and the same transformed F0. Use the implementation's documented numerical-input TabM variant without learned numerical embeddings for this primary grid. If an option is unavailable in the pinned API, record the exact proposed mapping and freeze it before fitting; do not silently relabel a custom MLP ensemble as native TabM. Train memberwise BCE and average member probabilities at inference, as verified against that implementation; do not mistake ensemble members for the three independent seeds.

After selecting each learner, run its full four-origin, three-seed refit replay, plus continuity diagnostic when within budget. One preprocessing contrast uses the selected residual MLP with standardized raw continuous F0 (no log transform), fixed config/seeds and no extra search. This distinguishes an architecture claim from a preprocessing claim. Report TabM versus residual MLP, both versus H2 and versus the archived pilot (the latter is contextual because its training recipe differs). Preserve negative results and allow either complete finalist into EXP-041 regardless of standalone superiority.

### 31.6 EXP-037 — daily histories and explicit sequence learning

Build a tensor per `(origin s, stable user)` covering exactly the **56** consecutive elapsed 24-hour bins ending at `C_s=s-4h`, oldest first. Bin j covers `[C_s-56d+j*d, C_s-55d+j*d)`, j=0..55. Six channels count visible source observations using `OfficialData.events` and the same six-stream convention as F0; a row must be visible before C_s and lie in the bin by its observation time. Apply log1p to counts. Add one availability channel: 1 only if the user's `first_seen_at` is no later than that bin's left edge, otherwise 0. Zero observed events means zero observed events, not certified inactivity. Standardize each count channel with inner-fit or refit training statistics only; retain the mask as binary. No interpolation from later days; no centered smoothing.

Use a common 128-wide two-layer GELU F0 branch and concatenate it with the sequence embedding, followed by Linear→GELU→Dropout(0.1)→six logits. Include two same-input controls before claiming an ordering benefit: (i) a flattened 56×7 tensor plus F0 passed to the selected residual MLP recipe, with width fixed from EXP-036-R, and (ii) the selected sequence architecture trained with a deterministic per-example permutation of its 56 bins (all seven channels move together, seed recorded), using the same procedure for train/validation/report. The latter destroys order while retaining the daily-value multiset; it is a falsification contrast, not a candidate for later cherry-picking. Also report F0-only versus sequence+F0, and a fixed sequence-only variant of the selected temporal architecture.

**GRU four-config grid:** `(hidden width, layer count) in {(64,1),(128,2)}` crossed with `lr in {0.0003,0.001}`; dropout 0.1 between recurrent layers when there is more than one; use the final hidden state as the embedding. Zero is a meaningful observation, so do not mask it out of recurrence.

**TCN four-config grid:** six causal residual convolution blocks, kernel 3, dilations 1,2,4,8,16,32, left padding only; width in {32,64} crossed with dropout in {0.1,0.3}, learning rate 0.001. Each block uses two causal convolutions with GELU/dropout and a residual projection if widths differ. Use the final time-step embedding. An end-of-window representation may attend to all earlier bins; verify causality rather than relying on a library default.

**Compact transformer trigger and grid:** execute only if the selected GRU or TCN improves mean all4 AP over the strong F0 MLP by at least 0.002 and wins at least 3/4 origins, or its fixed EXP-041 blend passes that study's development gate. Freeze the trigger calculation before launching. Four configs: two encoder layers, width {64,128} crossed with learning rate {0.0003,0.001}, four attention heads, feedforward width 2w, dropout 0.1, learned 56-position embeddings and mean pooling over observed-history positions; all-zero-mask users receive a zero sequence embedding. All bins precede C_s, so bidirectional attention inside the observed history is permitted; future-bin inputs are not. Record trigger-failed as conditional not_run. A temporal-model failure does not block RelGNN/RelGT.

### 31.7 EXP-038 and EXP-039 — stronger relational learners

First audit [RelGNN](https://github.com/snap-stanford/RelGNN) and its [paper](https://arxiv.org/abs/2502.06784), then [RelGT's paper and linked implementation](https://arxiv.org/abs/2505.10960). These are implementation sources, not evidence that either will improve this task. Use only instructor data; train all weights from scratch. Do not import a benchmark's labels, a pretrained table embedding, unrestricted ID embeddings or an external language encoder. Audit licenses and pinned dependencies in an isolated environment; do not change the accepted HGB runtime to install graph dependencies.

**Common initial graph:** reuse EXP-015's audited four node types, eight forward typed relations and their reverses, 90-day observation window, edge collapse/count/recency definition and safe node attributes. Build a separate graph for every training/validation/report origin from rows and endpoints visible before its own C_s. Exclude later-known lifecycle fields entirely. User input is F0; repository/PR/issue inputs are only the audited causal attributes or structural counts whose definitions are frozen. Save the exact input inventory. A row existing in the newest snapshot does not authorize its node/edge at a historical cutoff. Neighbor labels and target-week activity are never node features. No learnable per-entity ID table is allowed. Reverse relations do not grant future visibility.

**RelGNN primary implementation:** implement the authors' composite relational message-passing/atomic-route mechanism on the available schema, rather than renaming GraphSAGE. Register all schema-valid route sequences of one or two typed hops, their direction and aggregation; materialize the finite route list before any score. Four-config grid: hidden width {64,128} × learning rate {0.0003,0.001}; two message-passing layers, dropout 0.1. Use CPU-side temporal sampling, max fanout [15,10] for the two layers, followed by GPU computation. Freeze whether a route consumes one sampling hop or a composite operation and the resulting expanded cap. Paper-specific semantic options must be written into the score-free implementation manifest before fitting. Unsupported routes cannot be fabricated to make an arbitrary heterogeneous GNN look like RelGNN.

**RelGT primary implementation:** use its documented local relational-subgraph attention and global mechanism where supported; preserve token type, hop, time and local-structure information. Four-config grid: width {64,128} × learning rate {0.0003,0.001}; two transformer layers, four heads, feedforward 2w, dropout 0.1; at most 128 sampled local tokens per seed user, two relational hops, 16 learned global summary tokens/centroids if the pinned implementation exposes that mechanism. Freeze exact sampling, time encoding, pooling and centroid update semantics after code audit and before fitting. A custom model that omits a defining mechanism is explicitly named **RelGT-inspired**, with the deviations listed; it is not reported as a faithful replication. Any incompatible proposed constructor setting gets a dated score-free amendment, not a hidden runtime fallback.

For both families, produce a strong **no-neighbor** ablation with the same user input/training/head and no messages from other nodes, plus the EXP-036 node-only control. Verify a sampled subset against deterministic full-neighbor aggregation when feasible. Save edge/type counts, sampled fanout distributions, isolated-user predictions, missing endpoint handling, and same-user cross-origin mapping. A same-recipe GNN gain over no-neighbor supports a relational claim; a gain over the old weak MLP alone does not.

EXP-039 may be implemented while EXP-038 is scientifically negative. Stop a family for demonstrated incompatibility or resources, not because an earlier graph architecture lost. A full native-paper port that cannot fit this setting is a documented feasibility outcome; a clearly named adaptation may still answer the assignment's prediction question.

### 31.8 EXP-040 — representation before claiming architecture limits

Select the accepted RelGNN/RelGT configuration with the largest mean all4 AP (ties within 0.0005: fewer parameters, then lexicographic ID). Freeze it before representation changes. Compare its collapsed-edge graph with an **event-preserving** graph: one node per eligible direct event row from Pushes, IssueComments or PRReviewComments; join actor→event→repository, event→visible parent issue/PR where applicable, and reverse relations. Include relation type, elapsed age `log1p((C_s-event_observation_time)/day)` and cutoff-safe support masks. Retain each distinct source PK, including simultaneous events. No text or final lifecycle state enters the new nodes. Entity node attributes and all shared relations remain the original causal definitions.

Use at most the **64 most recent eligible events per user per direct-event type** within 90 days; sort descending observation time, then source PK for deterministic ties. Apply this cap independently at each origin, before sampling; report retained/dropped counts and coverage by type. Time values are computed from that origin's C_s. Event identity is for joins/ties only, not a learned feature. Reuse the selected architecture, width, optimizer, seeds and epoch-selection protocol; do not retune the event version. Input adapters required by the expanded schema are disclosed, including parameter-count differences. These two representations are a controlled system contrast, not a mathematically pure architecture contrast.

Run the selected event model once more with event ages zeroed, unchanged order/cap/settings. This identifies whether any gain survives removal of the explicit time signal. Report event-preserving versus collapsed, event ages versus zero ages, and H2. If even the capped graph exceeds resources, record incomplete feasibility; do not claim the collapsed representation was vindicated by a model that never ran.

### 31.9 EXP-041 — complementary neural–tree predictions

This is a new ensemble protocol. EXP-015's original requirement that a GNN beat REF before blending remains part of that completed pilot; it is not imposed on this new study. Every accepted full-window neural finalist with valid six-target, four-origin predictions is eligible, including a standalone neural loser. Pool at most one locked finalist from each of residual MLP, TabM, GRU, TCN, sequence transformer, RelGNN, RelGT and the event-preserving follow-up. Do not fit a second neural member merely to fill the pool.

For each member N and alpha in **{0.10,0.25,0.50}**, form a per-target rank score `q=(1-alpha)*rank01(p_H2)+alpha*rank01(p_N)` on the same origin population. Rank01 and stable-ID tie behavior are exactly the accepted P021 implementation. Evaluate (a) top-k by q and (b) top-k by `0.5*q+0.5*rank01(EW7)`, where k is the same persistence count for that target/origin. Alpha=0 is the matched H2 control for each layer. This is at most **48** fixed neural-blend policies plus controls, not unrestricted per-target mixing. No stacking, confirm-tuned weights, or two-neural mixtures in this registration.

Report every fixed grid cell. For the only selectable adaptive ensemble, select one member/alpha/layer using the mean of weekly combined F1 on the **two earliest** all4 origins (09-09,09-16). Only the six unconditional families whose configuration was selected on the earlier August inner folds may enter this adaptive selection pool: residual MLP, TabM, GRU, TCN, RelGNN and RelGT. The conditional transformer and EXP-040 representation parent depend on all4 information; their blend cells are exploratory and cannot enter this earlier-to-later selection replay. Ties within 0.001 prefer smaller alpha, then fewer neural parameters, then lexicographic full policy ID. Lock that complete choice and evaluate transfer on 09-23/09-30 with no revisions. Then freeze that same rule/choice for confirmation; do not rechoose it from the last two scores. The first two weeks are weight-selection development; the last two are development transfer checks, already exposed in the wider project and never called independent confirmation. This is an earlier-to-later weight-selection replay, not an out-of-fold stacking claim. A later standalone nominee can still be a conditional model if it passes §31.10 and is frozen before its eligible October outcome exposure.

Measure rank correlation with H2, per-target disagreement, incremental AP and top-k true positives lost/gained, and user-group stability. Diversity alone is not a win. Ensemble development eligibility requires: positive mean weekly score delta on the two later transfer origins, neither weekly delta below -0.005, and pooled all4 delta at least +0.001 versus matched H2 in the selected layer. Report the selected policy's delta versus H2/P021 as well even when the selected layer is raw top-k. No individual hindsight grid winner bypasses this procedure. A failed adaptive blend is an accepted negative; a weaker standalone neural model may still support an ensemble gain if these measured conditions hold.

### 31.10 Development nomination and fresh confirmation

**Metrics for every complete system:** mean of per-origin six-target AP; each target's AP/F1 and precision/recall; weekly macro/micro/combined F1 and pooled combined F1 under P016 and P021; all4/mature3 results; worst week; seed dispersion; parameter count, inference time and peak memory. Use the official evaluator and its all-negative convention. Save probabilities before operating policies. Do not choose per-target architectures or thresholds from these development results.

A standalone neural candidate is development-eligible if it beats matched H2 by at least **0.002 mean AP** over all4, AP improves on at least **3/4** origins, pooled P021 combined F1 improves by at least **0.001**, mature3 pooled P021 delta is nonnegative, and no weekly combined-F1 delta is below -0.010. Failure does not preclude EXP-041 or conditional representation diagnosis. Nominate at most one standalone model: highest mean AP among eligible systems; ties within 0.0005 prefer fewer parameters, then lower measured inference latency, then lexicographic ID. Report these thresholds as practical project tolerances, not significance tests.

For final research selection, freeze at most **two new policies**: the single standalone nominee and the single EXP-041 adaptive ensemble, if eligible. Add unchanged REF/P021 and H2/P021 as learned comparators, plus persistence and P020_multi as fixed rules. This replaces §30's three-learned-policy cap for this distinct new research manifest only. The closed §30 feature shortlist stays REF and H2 with no feature nominee. The new manifest must list complete code/config/source hashes, all attempted configs, preprocessing/epochs/refit rules, membership/roster adaptation, policies, permitted confirmation origins, exposure ledger and selection gates. No partial target policy is frozen.

**Origin reservation and source maturity:** use the existing `S+12d21h` operational maturity rule and measured source coverage, not commit date. Thresholds in UTC are 10-07→10-20 13:00; 10-14→10-27 13:00; 10-21→11-03 13:00; **10-28→11-10 13:00**. Source maximum is an operational proxy, not proof of perfect completeness. The new default research confirmation pair is **10-21/10-28** at 16:00 UTC; those are target-week starts Oct 22/29 Taiwan. A manifest frozen after a target start but before any outcome exposure gets label-blinded retrospective evidence, not a fully prospective claim. October 14 is available only to a candidate frozen before any direct or indirect outcome exposure; v0 and the feature shortlist keep their existing sets/rules. Do not postpone or suppress their scheduled results to create a favorable neural confirmation story.

Check the exposure ledger before every nomination and outcome read. Running old-model October confirmation, inspecting target summaries or raw activity from the target week also counts as exposure to a later-informed new model. Data-refresh integrity checks may compute schemas/PKs/FKs/coverage maxima, but do not build user-target-week labels or performance. If October 21 outcomes influence later architecture selection, only October 28 remains fresh for those candidates: retain their exploratory value but record that they cannot pass this two-origin replacement gate. More configurations cannot create additional independent weeks.

**Replacement rule for the new manifest:** on at least two common eligible mature origins, require positive pooled combined-F1 delta versus persistence; no weekly persistence delta below -0.010; positive weekly delta versus each of REF/P021 and H2/P021 on both origins; pooled delta of at least **+0.002 versus each learned comparator**; and no target F1 loss greater than 0.02 versus the stronger pooled learned comparator on two or more origins. Score all comparators on exactly the same population/labels and common origin subset. These controls prevent replacing H2 with a model that merely beats weaker REF. Among passing new policies choose highest pooled combined F1; differences within 0.002 prefer fewer neural components, then fewer parameters, then lexicographic ID. If none pass, retain the valid prior policy decision under its own frozen protocol; absent valid learned/rule confirmation, persistence remains default. Missing maturity/eligibility means inconclusive, not a neural rejection or a win.

Publish individual weeks and paired user-cluster bootstrap intervals (1,000 resamples, seed 20261022, resample stable users with all their repeated origin rows together). Intervals describe observed-user uncertainty, not temporal generalization with only two weeks. Do not alter gates after looking. A late finding may inform the report without entering final selection; any discretionary deviation must be explicit and relinquish the affected independence claim.

### 31.11 Resource budgets, failed trials and implementation boundaries

The observed pilot ran on an 8-GB RTX 3070 Ti Laptop GPU; this is the planning hardware, not a promise that the new models fit. One GPU training job at a time; at most **one subagent at a time**, no nested subagents, if the later executor uses the user's requested Sol-orchestrator/Luna-implementer workflow. These preferences do not authorize this plan-edit session to launch training. CPU provenance/scoring jobs may overlap only when they do not evict the GPU trainer or consume uncontrolled memory.

Initial search allocation: 12 tabular configs, 8 sequence configs, 4 RelGNN and 4 RelGT configs; optional transformer adds 4. Each config gets the two tuning-origin inner fits described above. Each family selects one config; each accepted finalist outer replay has 4 origins × 3 seeds × (inner fit + 56-origin refit) = **24 fits**, plus separately logged continuity diagnostics. Ablation fits are named separately and counted. The sequence controls, node-only controls, preprocessing contrast and graph representation trials are experiments too, not free hidden tuning. No five-origin × all-config × all-seed Cartesian sweep.

Allocate up to **18 GPU hours for EXP-036**, **24 for EXP-037 including its controls**, **24 for EXP-038**, **24 for EXP-039**, **16 for EXP-040**, and **8 incremental GPU hours for all extra EXP-041 prediction completion**: initial total **114 GPU hours** through the November 3 review. CPU input/preflight work is separately timed; this is not the old feature campaign's expired 8/12-hour authorization. A search inner fit has a 45-minute wall cap; a finalist inner/refit has a 90-minute cap. Measure the first complete job before scheduling the rest. A cap hit is incomplete training; do not rank a truncated run against completed ones as if optimization were comparable. Log a score-free budget amendment before extending a family, or defer its remaining variants with a reason. An unsuccessful early family does not consume all later allocations automatically.

On OOM, preserve the failure and retry once with batch size halved and gradient accumulation doubled to retain effective batch, under a new attempt ID. Graph sampler semantics/caps stay fixed. A second OOM stops that config as infeasible; changing fanout, token cap or graph semantics requires a new named resource variant frozen before its scores. Record failed, interrupted and repaired trials, their source/config hashes, elapsed work and fitted checkpoints. Do not erase the failed attempt or reuse its directory. At least once per day review completed evidence and remaining costs; do not allocate more compute solely because a model is fashionable.

Implementation should reuse the audited `OfficialData`, F0/labels, scoring and policy modules behind small adapters. Planned paths are `experiments/exp036_neural_tabular/`, `exp037_temporal_models/`, `exp038_relgnn/`, `exp039_relgt/`, `exp040_graph_representation/`, `exp041_neural_tree_ensemble/`, matching configs under `configs/` and preregistrations under `docs/exp/`. These are proposed, not existing deliverables. Keep source/graph/sequence construction separate from model training and from outcome scoring. An alternative of embedding custom time/label logic in each paper's runner was rejected because it would make causal parity difficult to audit. Adapters must return a common `(origin,UserID,target,seed,p)` contract and reject duplicate/missing keys.

Every run saves a complete resolved JSON config, upstream/package versions and hashes, table hashes, input/label key digests, feature/token manifests, seeds, optimizer/epoch histories, checkpoints, per-seed/ensemble predictions, metrics, resource logs and final disposition. Cache keys include source hashes, origin/cutoff/as-of times, populations, input definition and code hash. Do not reuse an old-source cache merely because its path exists. Evaluation must reject missing users/targets, wrong target order, nonfinite/out-of-range probabilities, accidental October rows and source/hash mismatches.

### 31.12 Required verification and bounded legacy closure

Before any real model fit, verify: future-row injection cannot alter earlier F0/sequence/graph inputs; feature timestamps at C are excluded and left boundaries included; per-origin stable-ID populations match controls; training label windows/as-of times satisfy the inner/outer clocks; preprocessing never sees validation/report rows; sequence dimensions, bin order and masks are exact; graph nodes/edges/routes cannot reference unavailable endpoints; no later-known fields or raw ID embeddings enter tensors. Use synthetic adversarial cases plus sampled real-row reconstruction. A small-data training check should show that the model can reduce loss, with a shuffled-label control that does not produce systematic outer gains. Do not mistake a tiny overfit check for real-data validation.

For each family, independently recompute predictions-to-metrics/policies from saved probabilities and check key/source/config completeness. For the nominated neural model and selected ensemble, additionally inspect representative input provenance and checkpoint inference on sampled keys. Validate official-scorer parity. Reserve expensive whole-source/fit replay for ambiguous correctness or final nominees; do not repeat hours of unchanged auditing between every config.

Original EXP-013 receives one bounded **two-hour closure window** in the accepted sklearn 1.9.1/NumPy 2.4.6 runtime, only after checking its residual attempt budget and authenticating its immutable inputs. The existing ledger conservatively counts 26 bundle starts toward cap 74; verify the actual ledger before authorizing any remaining work (at most 48 under that count). First reproduce all 12 control bundles exactly; only then enter its original candidate schedule. Do not change its scientific settings, max-error tolerance or gate to get past runtime drift. If that closure cannot complete under its remaining budget, preserve **Ongoing/incomplete** and record the missing candidate evidence. This legacy task does not hold the new neural program hostage and is not a reason to rerun the rejected §30 arms.

### 31.13 Calendar, feature follow-ups and course-data refresh procedure

| Taiwan dates | Main research work | Evidence/decision expected |
|---|---|---|
| **Oct 22–24** | Refresh/archive source; reproduce controls; implement residual MLP/TabM and sequence tensor checks | Actual source audit; executable score-free configs; first completed neural training diagnostics |
| **Oct 25 checkpoint** | Review tabular search and measured costs; continue GRU/TCN and RelGNN preparation | Completed/failed/pending ledger, not a forced final model freeze |
| **Oct 26–29** | Complete sequence and RelGNN replays; run EXP-041 as predictions arrive; RelGT implementation/preflight | Strong node/temporal references and initial relational results; test official roster when released Oct 28 |
| **Oct 30–Nov 3** | RelGT, representation ablations, conditional sequence transformer; final development comparisons | Complete negative/positive results, resource review, research-candidate manifest before eligible outcome inspection |
| **Nov 4–7** | Eligible confirmation, bounded fault diagnosis and robust inference; finish scientific comparisons | New architectures remain possible as exploratory studies; no reuse of exposed October outcomes as independent evidence |
| **Nov 8–10** | Final eligible confirmation when measured coverage permits, particularly Oct 28 after Nov 10 21:00; report/recording integration | Frozen model/policy and tested fallback already prepared before late data arrive |
| **Nov 11** | Final instructor refresh, exact-roster inference, end-to-end output validation and submission before 23:55 | Reproducible deliverables; no new architecture requiring last-night debugging |

The November 3 review is a resource/evidence checkpoint, not permission to mislabel a later useful model as forbidden. Later architecture research may continue while inference/report work proceeds, with its actual evidence limitations stated. Research time is substantial; fresh independent evaluation time remains finite.

Feature engineering continues **hypothesis-first** where a strong learner indicates a specific gap. Do not run all catalogue combinations. Existing rejected S/M/R/W/A features remain rejected for the H2 feature study; a neural interaction study would be a new preregistered contrast. EXP-034 X6 and conditional unions remain unrun unless explicitly admitted by a score-free amendment naming the fixed parent, added columns, contrast and extra budget. EXP-023 deeper HGB/EXP-025 population variants remain optional separate studies, not hidden changes to the reference. No newly tried combination inherits an old experiment's completed status.

**Data refresh steps, in order:** (1) record current checkout SHA/status and per-table hashes; preserve the entire checkout, including local untracked files; (2) fresh-clone the configured instructor remote into a new staging folder because the instructor force-replaces history and explicitly forbids `git pull`; (3) validate cloned HEAD, eight expected tables, schemas, non-null unique PKs, within-release FK coverage, stable cohort IDs, row/visibility maxima, and evaluator/schema changes; (4) on validation success archive the old checkout under a dated, SHA-labelled directory and move the verified clone to active `assignment2/`; (5) verify old/new archive hashes and active HEAD after the move; (6) write a machine-readable source/provenance/maturity report and a measured log entry. If validation fails, retain the original active checkout and preserve the failed staging clone for diagnosis. An integrity check may read FK columns as identifiers, but it does not make their late-known values permitted model features.

No training-label, confirmation-label or target-week user analysis is part of this refresh. Record the loader's actual `latest_observed()` coverage proxy separately from per-table maxima. An apparently newer commit can still fail a confirmation maturity gate. Keep old snapshots byte-identical; the current-data path update alone neither promotes H2 nor resets exposure history.

**Executed refresh receipt (October 22, 00:16 Taiwan):** active release is `2053302ba75751be975b9e779f567557b66f4071`; 2,568,311 rows (+15,523), same 9,000 cohort users, unchanged Arrow schemas and 20/20 FK checks without orphans. Old-source table hashes match the feature-campaign freeze. The only changed support document is SCHEMA.md's table row counts; evaluator bytes are unchanged. Loader coverage proxy is `2026-10-21T11:59:00Z` (Oct 21 19:59 Taiwan): 10-07 maturity proxy passes, 10-14/21/28 do not. No confirmation result was produced. See log §30.2, [input audit](../reports/research_refresh_20261022/source_audit_before_install.json), and [installation receipt](../reports/research_refresh_20261022/installation_receipt.json). These are executed source-update facts, not neural experiment results.

### 31.14 Executor completion checklist

- [x] Preserved pre-amendment plan and old instructor source hashes verified; refreshed source receipt records actual success.
- [ ] EXP-036–041 implementation manifests list exact inputs, native-paper deviations, grid, seeds, clocks, epoch/refit procedure and resources before fits.
- [ ] Original H2/REF probability parity and causal-input checks pass; full 56-origin neural refits use the same permitted training opportunity.
- [ ] All attempted configs/ablations and conditional skips are logged; no new model is marked Completed from this plan text alone.
- [ ] Strong MLP/TabM, GRU/TCN, RelGNN and RelGT receive their allocated research opportunity or a measured feasibility reason; EXP-040 separates representation effects.
- [ ] EXP-041 evaluates complementary predictions under its fixed pool/grid and earlier-to-later selection rule, including eligible standalone losers.
- [ ] Saved probabilities independently reproduce scores/policies; final nominee inference/provenance passes the deeper check.
- [ ] Research shortlist/exposure ledger is frozen before eligible outcome reads; revised candidates cannot inherit exposed confirmation wins.
- [ ] Results distinguish September development, retrospective label-blinded confirmation, fully prospective evidence and exploratory late studies.
- [ ] Measured findings enter `EXPERIMENT_LOG.md` before plan statuses change; exact-roster inference and final deliverables finish before the actual November 11 deadline.

**Registration state:** EXP-036–041 Pending. No new architecture has been implemented, trained, benchmarked, confirmed or promoted by this document edit. The old feature campaign remains a verified negative; stronger neural research is now the primary forward priority.

## 32. Luna Max source migration amendment — October 22, 2026

### 32.1 Authority and precedence

The user's active `/goal` explicitly selects refreshed instructor release `2053302ba75751be975b9e779f567557b66f4071` for this execution and requires source migration before any new-source control or neural fit. This dated amendment supersedes the old-source-first direction in §31.4 **for this goal's new work only**. The previous release `506d860693df60a698b68aa63a8edb4a984c8e1b` and all old manifests, caches, predictions, scores, exposure records and findings remain historical. Nothing is relabeled as if it used the refreshed source.

The refreshed checkout at `assignment2/` is the sole source for every new research dataset, cache, persistence prediction, REF/H2 fit, neural fit and blend in this campaign. Build all derived artifacts into new campaign-specific paths from empty caches. Authenticate across releases only by stable GitHub IDs; never use row-number IDs across source releases. Historical old-source features and predictions are not controls for any new-source model.

### 32.2 Verified source identity

The remote-HEAD verification and installation receipt name the active release as `2053302ba75751be975b9e779f567557b66f4071`; `git -C assignment2 rev-parse HEAD` returned the same SHA. The preserved checkout remains `assignment2_backup/506d860693df60a698b68aa63a8edb4a984c8e1b_20261022/assignment2/`. The source-refresh audit and receipt pass; current SHA256 values independently checked from `assignment2/data/db/` are:

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

Refresh evidence is retained in `reports/research_refresh_20261022/`. The old/new cohort and maturity facts there describe only input coverage, not predictive confirmation. No October outcome labels or target-week user results are authorized by this amendment.

### 32.3 Unchanged science and execution order

Except for the source choice and the explicitly requested work order below, use §31's registered source schema, ordered 59-column F0, target order, Thursday-origin schedule, population/eligibility, temporal cutoffs, label as-of semantics, HGB recipe, neural grids, objective, seeds, selection/refit procedure, preprocessing contrast, blend policy, temporal controls, verification and resource ceilings. Retain the same September report pools, and recalculate their source-specific maturity without retuning or changing the pool. The required order is: Step 0 source-aware data/cache rebuild and fresh matched P_persist7/REF/H2 controls; EXP-036-R; EXP-036-T; EXP-041-tabular immediately after both tabular finalists; EXP-037-G; then EXP-037-C and its temporal diagnostic controls. A verified negative result is complete evidence; a partial fit or resource failure is not.

Use the dedicated campaign `reports/neural_refresh_20261022T0033Z/` for state, protocol, manifests, progress and final reporting; use source-specific empty caches and outputs under `outputs/neural_refresh_20261022T0033Z/`. Do not overwrite or seed from historical runs. Preserve v0's 19 pinned source/config files. Keep RelGNN, RelGT, EXP-040, October confirmation, and all other unscoped work at their prior statuses.

### 32.4 Registration status

This is a score-free amendment. It records source authority and scope, not fitted results. Step 0 is now the active work package; model statuses remain Pending until implementation/execution begins, and may be marked Completed only after their required real-data evidence and verification are logged. The exact pre-amendment living documents are preserved as `EXPERIMENT_PLAN_before_source_migration.md` and `EXPERIMENT_LOG_before_source_migration.md` in the campaign directory.

## 33. Luna Max current execution ledger — October 22, 2026

This dated status supersedes §32.4's registration-time snapshot for the active campaign. It does not alter the frozen scientific protocol or historical study statuses.

### 33.1 Step 0 — Completed

The Step 0 rebuild and fresh matched REF/H2/P_persist7 controls use refreshed instructor SHA `2053302ba75751be975b9e779f567557b66f4071` from `assignment2/data/db/`. The source manifest, exact table hashes, cache identities, ordered 59-column F0, registered temporal splits, new-source maturity, source deltas, trainer runtime and full HGB defaults are stored under `reports/neural_refresh_20261022T0033Z/` and `outputs/neural_refresh_20261022T0033Z/`.

The build produced 100 feature origins and 403 label cache keys. All 100/403 caches passed source, table, builder-code, config, origin/as-of, schema, row/population and content-digest checks. All four fixed report populations pass the registered source-specific `S+12d21h` maturity proxy. Cross-release comparison joined stable GitHub IDs only: the four populations are unchanged, F0 changed on 0/1/1/1 users, and the common-cutoff label audit found 0 changed cells in 24 origin-target pairs. No old-source cache or prediction was used as a control, and no October-origin outcome label was built.

All 48 refreshed-source target fits (four origins × six targets × REF/H2) completed. The saved predictions contain report labels, populations, probabilities, P_persist7 flags, EW7 and P016/P021 decisions. `verify_controls.py` independently reconstructed training schedules/keys/labels/H2 weights, policies, AP/F1 and pooled metrics, checked all-negative official-scorer parity, and authenticated 48 checkpoint sample inferences. The verifier passed. Full results and score table are recorded in EXPERIMENT_LOG.md §32 and `outputs/neural_refresh_20261022T0033Z/controls/verification.json`.

### 33.2 Current scoped work status and order

| Order | Work package | Status on October 22 | Next required evidence |
|---:|---|---|---|
| 0 | Refreshed-source dataset/cache and matched controls | **Completed** | Already accepted under §33.1 |
| 1 | EXP-036-R residual MLP and fixed preprocessing contrast | **Completed — verified negative** | Both full finalist arms, scorer/policy/checkpoint parity, bootstrap and development gate recorded in log §36 |
| 2 | EXP-036-T official TabM | **Completed — verified negative** | Search, three-seed four-origin refits, all metric/gate checks and scorer/checkpoint verification recorded in log §37 |
| 3 | EXP-041-tabular child study | **Completed — verified negative** | Twelve cells and locked Sep 23/30 transfer independently verified; selected MLP/P021 gate failed, recorded in EXPERIMENT_LOG.md §38 |
| 4 | EXP-037-G GRU+F0 | **Completed — verified negative** | Search, three-seed finalist replay, complete verification and gate in log §39 |
| 5 | EXP-037-C TCN+F0 | **Completed — verified negative** | Full amended search/refits and independent verification passed; frozen gate failed, evidence in log §42 |
| 6 | EXP-037 temporal diagnostic controls | **Completed — independently verified** | All three registered controls completed 12 origin/seed pairs each; scorer/checkpoint parity and summaries in log §44 |

## 39. Luna Max live status update — EXP-041 tabular child complete

This supersedes the EXP-041 row in §33.2 and the registration-time status in §38. The frozen two-member blend child completed its full 12-cell scoring, selection transfer replay, official-scorer checks, per-target reporting, stable-user bootstrap, hash checks, and independent verification. The selected MLP alpha 0.10/P021 policy failed the registered transfer development gate; no candidate was promoted. Full results, corrections, verifier digests, and exact replay commands are in `docs/EXPERIMENT_LOG.md` §38 and `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/`.

EXP-037-G and EXP-037-C remain Pending. Their work remains sequential under the one-trainer rule; the temporal sequence cache must bind the refreshed source, exact origin cutoffs, feature populations, builder code, and the registered 56-bin tensor definition before fitting.

The global multi-family EXP-041 remains Ongoing under its existing scope; completing the tabular child will not close it. EXP-037-X, RelGNN, RelGT, EXP-040, October confirmation, and all other work outside this user goal retain their previous statuses. A negative model result may be accepted only after the complete registered real-data schedule and verification.

## 43. Luna Max live status update — TCN complete; temporal diagnostics next

This supersedes the TCN status in §§39, 41 and 42. **EXP-037-C is Completed — independently verified negative.** Its amended run completed 8/8 search fits and 12/12 three-seed full-window report refits; the independent verifier passed search and finalist checkpoint replay, source/split/preprocessing reconstruction, prediction/ensemble and policy recomputation, and official scorer parity. The selected `tcn_w64_d0.1` scored 0.535675 all4 mean AP (+0.000645 versus matched H2), but only improved on two of four origins. All4/mature3 P021 combined-F1 deltas were −0.001944/−0.002263, so the frozen development gate failed. Full scores, uncertainty, GPU resources, amendment history and exact commands are in `docs/EXPERIMENT_LOG.md` §42; artifacts and verification are under `outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01/`.

Both EXP-037 temporal finalists have passed independent verification. Before any diagnostic score is read, freeze and hash-bind the same-input control suite to the verified finalists and refreshed source. Then execute, sequentially with one trainer at a time, the flattened-history MLP, deterministic within-example joint seven-channel bin permutation, and sequence-only ablation; independently verify every seed/origin checkpoint, prediction, policy, metric and resource record. Preserve no October outcomes. The campaign ledger and state file must retain exact reproduction commands and verification hashes. The overall user-scoped goal remains active until these controls, final reporting and mandatory-artifact audit are complete.

## 34. EXP-036-T official implementation pin — Score-free

The selected official upstream source is Yandex Research TabM release v0.0.3 at commit [`a507095893d784c5702059d737ddfbd1299c41dd`](https://github.com/yandex-research/tabm/commit/a507095893d784c5702059d737ddfbd1299c41dd), licensed Apache-2.0. Its numerical-embeddings dependency is pinned to v0.0.12 at commit [`7381c5a4ed01253fbef9ea8cfed137f907265552`](https://github.com/yandex-research/rtdl-num-embeddings/commit/7381c5a4ed01253fbef9ea8cfed137f907265552), also Apache-2.0. Both sources are copied byte-identically with their licenses to a campaign-local import root; neither neural nor HGB global environments were modified. Source and installed-file hashes, exact package versions, runtime and constructor probe outputs are recorded in `reports/neural_refresh_20261022T0033Z/tabm_pin.json` and `outputs/neural_refresh_20261022T0033Z/third_party/tabm_install_receipt.json`.

The official `TabM.make` constructor resolves to `n_num_features=59`, no categorical features, `d_out=6`, `num_embeddings=None`, three blocks, width 128/256/512, dropout 0.1, official ReLU activation, ensemble size 16, `arch_type='tabm'`, and `start_scaling_init='random-signs'`. CPU API probes confirmed three official `MLPBackboneBatchEnsemble` blocks, no numerical embeddings, and `(batch,16,6)` outputs for every width. No adapter patch was needed. EXP-036-T retained exactly the six registered width × learning-rate cells and the shared weighted-BCE / member-probability averaging protocol. After EXP-036-R verification, its score-free preregistration was frozen at `outputs/neural_refresh_20261022T0033Z/exp036/tabm/preregistration.json` (SHA256 `0065a106719740ce9dbe3f4e9ab6feb066c60ad6c17bf5ddc8a86571698a1ee3`). The complete TabM search, finalist and verification are in §37; it failed the frozen development gate.

## 35. EXP-036-R implementation and score-free preregistration

The runnable sequential trainer, authenticated resumable checkpoint core, source-aware F0 preprocessing and focused contract tests are implemented. The exact residual MLP preregistration is recorded in log §34 and saved at `outputs/neural_refresh_20261022T0033Z/exp036/residual_mlp/preregistration.json`. It freezes the six-config search, AP/tie selection, primary signed-log preprocessing and one fixed raw-continuous contrast, the two August tuning folds, four September report origins, seeds 42/43/44, inner early stopping, fresh full-56-origin refits, H2 weights, CUDA/FP32 runtime and time/resource ceilings. The bounded synthetic sanity fit and later shuffled-label controls are process checks only, not predictive evidence. A pre-fit input-reader lookup failure was repaired before training; full results and verification are in log §36.

## 36. EXP-036-R execution status — Completed, development gate failed

All six search configurations, all four-origin × three-seed primary refits, and the fixed raw-continuous preprocessing contrast completed on the registered refreshed-source inputs. The primary MLP and contrast both failed the unchanged development gate versus matched H2. The complete numerical evidence, paired uncertainty, scorer/checkpoint verification, process sanity timing, verifier repair, resource use and exact reproduction commands are in EXPERIMENT_LOG.md §35. This is a completed negative result, not a nomination for later selection.

**Status: Completed — verified negative.** EXP-036-T's execution and full verification are recorded in log §36. EXP-041-tabular is next after both tabular finalists; the temporal models remain later in the registered order.

## 37. EXP-036-T official TabM execution — Completed, development gate failed

The official implementation pin and score-free family preregistration are recorded in §34 and EXPERIMENT_LOG.md §§33 and 36. All six configs × two August folds and all four-origin × three-seed September refits completed. Independent verification passed source/key/label checks, 24 checkpoint sample inferences (maximum error 1.2e-7), policy recomputation and scorer parity. The selected `w128_lr0.001` model gained +0.002539 all4 mean AP versus H2, but improved on only two of four report origins, had only +0.000330 all4 P021 F1 gain, and fell short of mature3 P021 parity. Both stable-user bootstrap intervals include zero; the frozen development gate fails. Detailed scores, per-target metrics, uncertainty, resource use and reproduction commands are in EXPERIMENT_LOG.md §36.

**Status: Completed — verified negative.** The next step is EXP-041-tabular: register its two-member child and frozen September 9/16 selection rule before evaluating the 12 blend cells, then replay the frozen choice on September 23/30. The full multi-family EXP-041 status remains Ongoing.

## 38. EXP-041 tabular-child registration — score-free

The two-member MLP/TabM subset is registered as `EXP-041-tabular-MLP-TabM-SeptemberBlend-v1` at `outputs/neural_refresh_20261022T0033Z/exp041/tabular_child/preregistration.json` (SHA256 `67851fb6cea19739c05342305b136c9c81d4850a65a068bd1835920193abdcda`). It authenticates source `2053302ba75751be975b9e779f567557b66f4071`, the eight table hashes, fresh H2 controls, both verified finalists and the two implementation hashes. The exact 12-member/alpha/layer cells, average-rank and stable-ID rules, P_persist7 k, EW7, selection metric, tie rule, transfer gate, bootstrap and reproduction commands are in the JSON registration and log §37.

Selection uses mean weekly combined F1 over Sep 9 and Sep 16 only. Ties within 0.001 resolve by smaller alpha, fewer neural parameters, then lexicographic full policy ID. The final selection lock is SHA256 `d1f30b755eb1544f5c718737f80c083756d882e49572bcf9a262c21e78c6abb7`; it selects `EXP-041-tabular-MLP-alpha=0.10-layer=P021`. The maximum mean was 0.535833 at MLP alpha 0.25/P021; three MLP/P021 cells fell within the 0.001 band and the registered lower-alpha rule selected alpha 0.10 (mean 0.534848). The pre-transfer check is in `selection_pretransfer_verification.json`. The first metadata-only preflight stopped on a member-to-parameter lookup. Later selection invocations found missing saved H2 P016/P021 and P_persist7 decision columns in the reader and one wrapper recursion. Those failures and repairs are preserved in amendments 01–03, with SHA256 values `afbdbf53c7b2d08bd905796efe486fd7782692cce10105783227bccc042acd79`, `d239d3997c4389cb59a92c44c9081786dec31e320a82229e08f53138c83e1fbd`, and `bcf41be968b627b733ae38f4892d4fa74e79079cff1975cd4d70cba4cc2e16de`. The first full transfer attempt loaded the four report populations but exited with Windows status `-1066598273` before persisting metrics; single-thread environment settings did not resolve it. A synthetic diagnostic isolated the native crash at `np.corrcoef`. Report-only amendment 04 (SHA256 `9b7bd76faae975ce19dcbec9f9ac99791bcb52870376e57cf42a1b6ca25e4ec5`) uses equivalent centered sums for diagnostic correlations and leaves the frozen selection and all scoring/gate formulas unchanged. The transfer replay and independent verification are still required.

The child is **Ongoing**: the Sep 9/16 grid and choice are frozen and independently tie/hash checked. Required next evidence is the four-origin replay for all 12 cells, transfer-gate disposition and independent scorer/key/label/bootstrap verification. The full multi-family EXP-041 study remains Ongoing after this bounded child.


## 41. Luna Max live status update — GRU complete; TCN active

This supersedes the EXP-037 rows in §33.2 and the pending-family status sentence in §39. It records measured evidence without changing the frozen experiment gates or the broader multi-family EXP-041 scope.

**EXP-037-G is Completed — verified negative.** Its four-config/two-origin search, 12 three-seed full-window refits, source/key/as-of and fit-only-preprocessing checks, 24 sampled checkpoint inferences, per-seed ensemble reproduction, official scorer parity, metrics, paired bootstrap and failed development gate are recorded in `docs/EXPERIMENT_LOG.md` §39. GRU all4 mean six-target AP was 0.532624 (−0.002406 versus H2); all4/mature3 P021 combined-F1 deltas were −0.000593/−0.000995. Verification SHA256: `91848f968bd7d2cba2d2d0a40c5b0f3431acd2bf0be9c13fa2f1b74d7334ead6`.

**EXP-037-C is Active — score-free preregistered.** The exact four-cell TCN grid and causal architecture are frozen at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/preregistration.json` (SHA256 `a7583591d2c1b7e08a1693a81e39992c3f61952eb8ca23c5ea4f7ed38d88423f`). No TCN candidate report scores have been accessed. Required evidence is its full search, frozen choice, four-origin × three-seed inner/full-56 refit replay, independent verification, and the preregistered flattened-history MLP, all-seven-channel temporal permutation, and sequence-only diagnostics. The campaign `reports/neural_refresh_20261022T0033Z/PROGRESS.md` and `state.json` carry exact resume commands, hashes, accepted artifacts, current stage, and owned-process state. One trainer at a time remains enforced.

Step 0, EXP-036-R, EXP-036-T and the named EXP-041 tabular child remain completed verified studies. The wider EXP-041 remains Ongoing; work outside the user-scoped goal is unchanged.

## 42. Luna Max live status update — TCN compute amendment registered

This supersedes the TCN status in §41 and records the incomplete first attempt without replacing its history.

**EXP-037-C remains Active — amended score-free registration; no amended fit has started.** The original registration at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/preregistration.json` remains immutable. Its first width-32/dropout-0.1 search fit exceeded the 2,700-second cap at epoch 2; its checkpoint, partial epoch history and console are preserved, and its partial AP is excluded from ranking. A synthetic-only, same-batch runtime probe found medians of 10.9649 seconds/step with deterministic cuDNN versus 0.02791 seconds/step with `cudnn.deterministic=False`; `cudnn.benchmark=False` in both cases. No campaign data or outcomes were used for this probe.

Before another TCN fit, compute amendment `EXP-037-C-resource-amendment-01` was registered at `outputs/neural_refresh_20261022T0033Z/exp037/tcn/resource_amendment_01/amendment.json`, bound to its own amended preregistration and fresh run root. Only the cuDNN deterministic execution flag changes. Seeds, precision, model/grid, inputs and preprocessing, batch/optimizer/loss, folds, fit caps and selection/gate rules stay fixed. Fixed seeds remain in use; nondeterministic kernels mean bitwise repeatability is not claimed, and checkpoint replay is verified at 1e-5.

Next: run the complete amended 8-fit search and 12 origin/seed finalist replays from scratch, independently verify the refreshed-source outputs, then register and run the three diagnostics only after both temporal finalists pass verification. `state.json` and `PROGRESS.md` carry the precise hashes, accepted outputs, process state, and exact run/verify commands. Step 0, both tabular families, EXP-041 tabular child and GRU remain complete verified negatives; global EXP-041 remains Ongoing.

## 44. Luna Max live status update — score-free temporal diagnostics registered

The three temporal diagnostics are now preregistered after successful GRU and amended TCN finalist verification. Registration `outputs/neural_refresh_20261022T0033Z/exp037/diagnostics/preregistration.json` has SHA256 `c6a0319e393da85c07724aadb4e5a00e66492a324a67b28657d1563324448a95`. The registered rule selects TCN `tcn_w64_d0.1` from verified all4 mean AP (0.535675 versus GRU 0.532624); flat-history MLP freezes the EXP-036-R width 256 / learning rate 0.0003 winner. Remaining controls are selected TCN with deterministic within-example joint seven-channel temporal permutation and selected TCN with F0 removed. Each recipe will fit 12 origin/seed pairs and 24 checkpoints; run the controls sequentially with one trainer.

The first score-free preflight encountered the EXP-036-R selection JSON's nested `config_id`; it stopped before creating a registration, scoring any diagnostic prediction, or fitting a model. The adapter was corrected and compilation passed before the frozen registration. The complete protocol and exact run/verification commands are in `docs/EXPERIMENT_LOG.md` §43. Current stage is sequential diagnostic training, followed by independent source/checkpoint/decision/official-scorer verification and the final report/artifact audit. October outcomes remain reserved.

## 45. Luna Max live status update — all scoped results verified

This supersedes the active status in §§43–44. The three registered temporal controls each completed all 12 origin/seed pairs (24 inner/refit fits), and the independent verifier passed source/split/preprocessing reconstruction, every saved checkpoint sample, per-seed and ensemble reproduction, policy/metric reconstruction and official scorer parity. The root diagnostic verifier SHA256 is `6acdf6add23342271027e1b5d3968aadbbb64a6ee40fc042e5bba6bd2eb67643`; descriptive seed, target, week and resource summaries are in log §44. Their all4 P021 combined-F1 deltas versus H2 were −0.002373 for flattened history, −0.001792 for order permutation and −0.007599 for sequence-only. Full confidence intervals, mature3 and per-target results are recorded in the experiment log and diagnostic artifacts.

The complete user-scoped work order—Step 0, EXP-036-R and its fixed preprocessing contrast, EXP-036-T, the EXP-041 tabular child, EXP-037-G, EXP-037-C and all three temporal controls—has passed its required real-data execution and verification. Each model-family development gate failed, so the accepted outcomes are verified negatives; no promotion is implied. October labels and outcomes remain reserved. The wider multi-family EXP-041 study and other out-of-scope studies remain Ongoing as before. Final report: `reports/neural_refresh_20261022T0033Z/FINAL_REPORT.md`; detailed evidence: `docs/EXPERIMENT_LOG.md` §§36–44 and the source-bound artifacts under `outputs/neural_refresh_20261022T0033Z/`. The ledger records exact reproduction commands, accepted outputs, hashes, final process state and remaining work (none within this goal).

## 46. Luna Max final close — all scoped packages accepted

This final ledger entry supersedes the in-progress status language in §§33, 39 and 41–45 for the bounded user goal. Step 0, EXP-036-R (including the fixed raw-F0 contrast), EXP-036-T, the named EXP-041 tabular child, EXP-037-G, amended EXP-037-C, and all three preregistered temporal diagnostic controls each have their required real-data results and independent verification artifacts. All neural model/blend gates failed; the completed negatives do not nominate or promote a model. The 12-cell blend child is complete; the larger six-family EXP-041 study remains Ongoing outside the requested scope.

Final report: `reports/neural_refresh_20261022T0033Z/FINAL_REPORT.md` (SHA256 `c2f1bb8200f6429406abd625eab2349c5795023eb664c66f4bf8347ca5adf4d6`). Final diagnostic root verifier: `6acdf6add23342271027e1b5d3968aadbbb64a6ee40fc042e5bba6bd2eb67643`. The machine state is `completed_verified`, lists 118 existing accepted-output paths, contains the exact command history, has an empty scoped `remaining_work` list and no owned trainer. The final audit resolved one stale accepted-output reference to an absent historical EXP-041 failed-transfer console; the failure and correction remain documented in the progress ledger/log, and the corrected transfer plus final verification artifacts exist.

All new-source results use instructor SHA `2053302ba75751be975b9e779f567557b66f4071`, with the source migration registered before rebuilding. No old-source feature/prediction was reused as a new-source control, no October outcome was built or read, and no historical study or reserved October outcome was changed. There is no further scoped execution or verification remaining.

## 47. October 23 strategy amendment: current forward priorities

### 47.1 Scope, precedence and evidence

**Effective planning amendment, 2026-10-23 (Asia/Taipei). No new model has been implemented, fitted, nominated or promoted by this edit.** This section adopts the revised priorities in the [second strategy review](<C:/Users/Davon/Desktop/Master/Semester 3/535703 Big Data Analytics/Assignment 2/reports/strategy_review_20261023/SECOND_REVIEW.md>). It supersedes conflicting remaining-work order, per-family spending expectations, screening requirements and calendar language in §§29–31. It also supersedes the earlier review's count-first order and requirement that a simple richer-input encoder win before a stronger relational learner is tried.

Completed studies and their gates remain unchanged. §32's refreshed source pin, §31's causal data/clock rules, exposure accounting and §31.10's final two-origin replacement rule remain controlling. New suffixed studies below receive new executable preregistrations; they cannot inherit completion or confirmation from a parent. This strategy amendment is not an executable experiment registration and does not itself launch work.

**Working reference:** refreshed-source H2/F0/P021 pooled all4 combined F1 **0.5476344735753119**, source `2053302ba75751be975b9e779f567557b66f4071`. Persistence remains subject to the existing submission-default rules. The completed MLP, TabM, GRU, TCN, tabular blend and temporal diagnostics did not qualify as replacements. The original feature arms remain rejected and B/AB remain untested. These are development findings, not independent November evidence.

The saved-prediction diagnostic's hindsight true-count gain (+0.024463) is not an attainable-gain estimate. Required count corrections reverse between weeks; EXP-018 already tested simple count forecasts. Likewise, low recall without same-type prior7 activity does not diagnose the main missing feature: P016 improves that stratum's recall for all six labels while reducing pooled F1 to **0.5383663378622162**. Score gains must preserve a useful precision/recall tradeoff. The existing 3.19-million-parameter MLP does not establish that relational computation has been tested at adequate capacity.

### 47.2 Current state and interpretation

| Work | Current state | Forward implication |
|---|---|---|
| Step 0, EXP-036-R/T, EXP-037-G/C, registered temporal controls, EXP-041 tabular child | Completed within the bounded Luna campaign; development gates failed, as recorded in §46 and the log | Do not list them as next or rerun them merely to fill the plan. Preserve their preregistrations and negative results. |
| EXP-038 / EXP-039 stronger relational families | No completed candidate result in the inspected record | Give one an early fair trial; no automatic full implementation of both. |
| EXP-040 event-preserving representation | Unexecuted | A possible distinct follow-up, not permanently gated on a positive handcrafted or pooled representation result. |
| EXP-037-X compact sequence Transformer | Original trigger is satisfied versus the F0 MLP | Admissible, but deferred behind the first relational trial on opportunity-cost grounds; may be selected as the one later follow-up. |
| Broader multi-family EXP-041 | Ongoing outside the completed tabular child | Evaluate complementarity as new accepted predictions arrive under a separately frozen child recipe. |
| Original EXP-013 | Ongoing/incomplete; no original candidate result | Not a predictive negative and not a prerequisite. Limit historical recovery to a short bounded task. |

### 47.3 Prioritized work and time boxes

Suffixes below name **new planned children**, not completed replications. Exact configs, grids, input hashes and nomination rules must be frozen before their fits or new candidate scores. Budgets include implementation, execution and review; they are not GPU-hour forecasts or score promises.

| Order / planned study | Status at this amendment | Budget and concrete work | Exit / next action |
|---|---|---|---|
| Preparation | Pending | **2–4 hours** for a versioned trainer/data/clock check, relevant replay/report repairs, existing learning curves and inference feasibility. Cap recovery of missing historical executed source at two hours. | Reuse authenticated caches and scoring. Record any unrecoverable replay limitation; do not require all old neural training to be repeated before new research. |
| Tabular block: **EXP-023-S**, **EXP-036-E**, optional **EXP-011-R** | Pending; new follow-ups | **One working day total**. Current HGB baseline plus at most two substantial capacity contrasts; one TabM numerical-embedding treatment with its matched embedding-off control; at most one CatBoost **or** LightGBM configuration under current H2/P021. Choose the optional family before results. | Close or retain cheap candidates on measured score, transfer, curves and cost. Omit the optional tree if it would delay the relational trial. No Cartesian sweep or second tree family automatically. |
| Optional count study: **EXP-018-R** | Deferred unless a valid OOT cache is already usable; revisit during the initial block | **2–4 hours**, at most one genuinely new regularized count/decision mechanism beyond EXP-018, with ranking fixed for the count contrast. | Stop if substantial cache reconstruction is needed or chronological transfer gives no useful signal. Oracle counts and September-fitted multipliers remain diagnostics. This is not a prerequisite for relational work. |
| Main high-upside trial: **EXP-038-S** | Pending | Protected **2–3 working days** for a stronger RelGNN-inspired model retaining F0, matched self-only/no-neighbor control and a cheap pooled-neighbor comparator. A score-free preflight may select **EXP-039-S** instead if the permitted RelGT implementation is materially easier; record that choice before candidate scores. | Complete a fair optimized screen, resource measurement and one prespecified H2 complementarity check. A losing simple pooling or handcrafted-feature probe does not cancel this block. |
| One distinct follow-up: **EXP-040-S**, **EXP-039-S** or **EXP-037-X-S** | Deferred until selected at the relational checkpoint; reason: preserve a finite research budget, not a predictive rejection | Reserve **2–3 further working days before the shortlist freeze** for one diagnosed mechanism: event-preserving input, materially different relational computation, or the triggered sequence Transformer. Choose an unused child; do not reopen the first completed screen under the same registration. | Record the question and selection rationale before its scores. A negative first architecture does not veto a distinct mechanism; an automatic multi-family tournament is excluded. |
| Complementarity: **EXP-041-S** | Pending new child; existing parent remains Ongoing | A small fixed blend recipe on accepted common-key predictions, inside the corresponding branch budget. | Freeze membership/weights or an earlier-to-later selection rule before the child's candidate scores; compare against H2/P021. No retrospective promotion of an old unselected blend cell. |
| Shortlist, confirmation and deliverables | Pending | Freeze at most **two new complete policies by November 2**, preferably earlier. Prepare eligible confirmation, exact-roster inference, fallback and graded explanation. | Apply the unchanged final replacement safeguards on eligible mature origins. Have the fallback ready before late releases; protect several days for slides/video and submission. |

Start the stronger relational trial after the short initial block. It must not wait for every calibration, numerical-encoding, sequence or pooled-encoder experiment to finish. The original §31.11 GPU ceilings are safety ceilings for their named studies, not obligations to spend the entire allocation on both graph families. A screen does not complete the original four-config/three-seed family schedule. Any expansion beyond these working-time boxes needs a recorded pre-score budget/scope amendment; no automatic three-day extension to polish +0.001.

### 47.4 Common protocol and fair architecture tests

- Retain instructor-only data, training from scratch, six targets, stable-ID keys, cutoff `C_s=S-4h`, decision clock `A_S=S-3h`, causal feature visibility, label-as-of rules, population definitions and H2 row weights. Do not use raw learnable user/repository-ID tables, future lifecycle attributes, external data or pretrained weights.
- Use a named pinned snapshot and fresh output directory. Continuing the September comparison starts from the refreshed source above. A newer snapshot requires matched controls and source-specific cache/maturity checks; do not subtract scores across releases as a model effect.
- Preserve the purged chronological fitting design: inner 42 origins, seven purged origins, seven validation origins, and a fresh final fit on all 56 allowed daily origins. Fit numerical bins/scalers/encoders only on permitted fitting rows. Use the fixed earlier August tuning origins and September all4 report pool; mature3 remains the first three registered September origins. September remains adaptive development.
- New studies may register architecture-appropriate preprocessing, optimization or scheduler choices. Keep equal data opportunity and disclose recipe differences. Retain the common weighted six-target objective for a matched architecture contrast; a different learning objective is a separate named recipe treatment. Do not change several factors and attribute the whole gain to parameter count.
- For the relational computation contrast, keep graph definition and user inputs fixed. The self-only control tests the neighbor-using system's incremental value. A pooled-neighbor control must receive the same permitted counterpart information before claiming that learned relation routing/composition adds value. Keep collapsed-versus-event-preserving input as a separately named representation contrast. A pooled control's loss is not a veto on the stronger model.
- Pin a reviewed paper/code pair and list mechanisms omitted by an adaptation. The literature review's RelGT v1 is not silently interchangeable with the subsequently retrieved v2. Name a substantial adaptation “inspired.” Use the existing graph scaffold where valid, with current-source and historical-cutoff checks.
- Check source-time meaning before a new temporal model: the existing sequence counts use observation time, and the “availability” channel means user first-seen status rather than extraction-outage completeness. This alone is not a bug or evidence that an outage-aware model will improve the score.
- Preserve source/checkpoint provenance and executed source bytes. Known replay and seed-dispersion findings require bounded verification before reusing affected infrastructure; this amendment does not claim they were repaired. One GPU trainer at a time; keep the accepted runtime isolated.

### 47.5 Screening, investment and nomination

For **new §47 children only**, use a cheap feasibility probe, a common **one-seed** chronological screen, and deeper work for the best one or two complete systems. A selected configuration is locked from earlier tuning before its September report predictions. Each report still receives causal inner epoch selection and a fresh 56-origin refit; screening does not justify withholding training opportunity relative to H2. Exact counts, resource caps, operating policies, blend recipe, nomination criteria and tie-breaks belong in the child's pre-fit registration.

Judge investment primarily by **official combined F1 against matched H2/P021**, weekly/target behavior, convergence, measured engineering/inference cost and prespecified complementary value. Record AP and P016 as ranking/operating diagnostics. Count error, returner recall, parameter count and paper benchmark scores do not substitute for the homework metric. Define activity/history strata causally; do not select a per-target architecture from September's best cells.

The earlier +0.003/+0.005 continuation figures are **investment guides, not automatic rejection or significance thresholds**. A cheap +0.001 already obtained may be retained for later confirmation; it does not buy three more days of tuning. A promising or clearly undertrained model gets at most one bounded follow-up within its block. A cap-hit/incomplete run is an execution limitation, not a predictive negative. Record when an adequately trained branch is stopped and what question its negative result answers.

Only neural finalists receive the new child's full three-seed replay (42/43/44); non-neural finalists use their registered matched replay and appropriate variability checks. Reserve fuller ablations, paired-user uncertainty and deeper checkpoint/source verification for finalists or suspected faults. New screening/nomination rules do not retroactively relax §31.10's AP/F1 gates for the completed Luna studies. Its AP-first development eligibility is not an automatic gate for these distinct new children; freeze their official-score-first nomination rules before fitting. No new nominee is accepted merely because this plan names it.

**Final replacement safeguards remain §31.10:** at least two common eligible mature origins, the frozen comparator/target/week guards and pooled margin, and the existing fallback logic. Shortlist and all policy choices must be frozen before direct or indirect confirmation-outcome exposure. A later revised candidate cannot inherit its parent's independent confirmation. User-cluster intervals describe observed-user uncertainty; they do not create additional independent temporal regimes.

### 47.6 Calendar, maturity and grading

| Taiwan dates | Main work / decision |
|---|---|
| Oct 23–26 | Bounded preparation and one-day tabular work; start the protected relational block without waiting on optional count or sequence studies. |
| Oct 27–Nov 1 | Finish the first relational screen and at most one distinct follow-up; after Oct 28 roster release, inspect permitted feature-only roster coverage/distributions and prepare exact-roster inference. |
| Nov 2, preferably earlier | Freeze at most two complete new policies and unchanged comparators, with actual exposure ledger, source adaptation and exact nomination/confirmation rules. |
| Nov 3–7, subject to actual maturity | Eligible confirmation and final verification; inference rehearsal and graded schema/entity, relational, temporal and pipeline explanation. |
| Nov 8–10 | Final fallback/output checks, slides and student-recorded video; optional late evidence only when measured coverage passes. |
| Nov 11 | Validated final instructor release and exact-roster inference; submit before **23:55 Taiwan**. No new architecture dependent on last-night debugging. |

Keep the `S+12d21h` operational maturity proxy. October 28 requires **November 10, 13:00 UTC = 21:00 Taiwan** coverage. If a 21:00 release contains observations only through 20:00, it may not qualify until the November 11 release. This is a scheduling risk, not a certainty about future coverage. Do not lower the maturity threshold late.

A new manifest may use an earlier **October 14/21** pair only if each candidate's full exposure ledger permits it and the rule is frozen before those outcome reads. Otherwise retain the valid original October 21/28 schedule. Label late-frozen historical tests as blinded retrospective evidence. Preserve v0 and prior manifests, do not delay their jobs to manufacture independence, and do not treat the October 12 sanity check as independent again. This edit opens no October outcomes.

Protect presentation/pipeline work: the assignment allocates **7% of course grade to prediction performance and 8% to slides/video**, with no specified linear grade conversion for +0.001 F1. Several days for a reproducible final method, the required figures, at most ten English slides and the student's 8–12-minute recording are part of maximizing the homework score.

### 47.7 Preservation and execution boundary

The exact pre-amendment plan is preserved at [EXPERIMENT_PLAN_before_section47.md](<C:/Users/Davon/Desktop/Master/Semester 3/535703 Big Data Analytics/Assignment 2/reports/strategy_review_20261023/plan_amendment_20261023/EXPERIMENT_PLAN_before_section47.md>), SHA256 `52d8faee4d941f449f4d8de051f662e21527ce6f39f9eab418f153f47d5fbe84`. Earlier review manifests describe their original review time; the archived bytes preserve the plan hash they recorded. Their historical hashes must not be rewritten to match this new canonical plan.

This amendment changes the opening navigation/current summary and appends §47. The original experiment register and §§3–46 remain preserved. `EXPERIMENT_LOG.md`, previous reviews/manifests, model outputs, checkpoints, sources and frozen scientific registrations are unchanged. A separate amendment manifest records before/after document hashes and preservation checks. New work remains planned until an exact executable specification, actual execution and the required log evidence exist; this document update supplies no new predictive result.

## 48. Overnight five execution — October 23, 2026

Execution of [OVERNIGHT_FIVE_EXECUTION_BRIEF_20261023.md](<C:/Users/Davon/Desktop/Master/Semester 3/535703 Big Data Analytics/Assignment 2/docs/exp/OVERNIGHT_FIVE_EXECUTION_BRIEF_20261023.md>) began from refreshed source `2053302ba75751be975b9e779f567557b66f4071`. The corrected preregistration is `outputs/overnight_five_20261023T0237TW/preregistration.json` (SHA256 `2a5bd501eaf934998d01f0bbbc57896287d9b6bf0a01a035d9ba1ab397801bb8`); it records the resolved configs, source snapshot, data/cache inventory, CatBoost 1.2.10 package inventory and brief hash. The H2/P021 control was reproduced at **0.5476344735753119**, with official evaluator parity on all four September origins. The feature inventory remains 100 origins and 403 label-cache keys, with 59 F0 columns; no October-origin labels were built or read.

The first preregistered runner stopped before its first HGB estimator fit because `threadpoolctl` raised Windows native error `-1066598273`. Its preregistration and partial graph caches remain preserved under `outputs/overnight_five_20261023T0217TW/`. The repaired runner sets the four-thread OpenMP/BLAS environment before numerical imports and uses CatBoost's registered `thread_count=4`; the HGB and CatBoost smoke fits passed. A fresh source snapshot was frozen before candidate fits. The obsolete first runner was stopped after its HGB failure to maintain one active GPU trainer; its interruption receipt records that no model checkpoint was produced.

### 48.1 EXP-023-S — complete and verified

The two August folds selected `more_rounds` (320 iterations, 15 leaves) at mean six-target AP **0.260036** versus **0.255767** for `more_leaves` (80 iterations, 63 leaves). The selected config was locked before September scoring. CPU runtime identity matched the saved H2 fit manifest, so no runtime-reference refit was needed. All four September fits completed 320 iterations on all six targets. Their predictions cover the exact common H2 key set; both P016 and P021 match the instructor evaluator.

| Origin | Users | HGB P021 F1 | H2 P021 F1 | Δ F1 | HGB six-target AP |
|---|---:|---:|---:|---:|---:|
| Sep 9 | 6,382 | 0.546104 | 0.549048 | −0.002944 | 0.548678 |
| Sep 16 | 5,862 | 0.515286 | 0.519272 | −0.003986 | 0.511052 |
| Sep 23 | 5,497 | 0.546912 | 0.554211 | −0.007298 | 0.492618 |
| Sep 30 | 5,178 | 0.550540 | 0.554036 | −0.003496 | 0.504247 |
| **Pooled** | **22,919** | **0.543451** | **0.547634** | **−0.004183** | **0.480058** |

Pooled target metrics (P021 F1 / delta versus H2/P021 / AP): open PR `0.488368 / −0.001604 / 0.440031`; open issue `0.467922 / −0.010852 / 0.413782`; issue comment `0.501436 / −0.002462 / 0.430469`; PR review `0.504027 / −0.005685 / 0.450395`; PR review comment `0.476687 / −0.006521 / 0.388341`; push `0.673566 / −0.001995 / 0.757332`. Total report-fit wall time was **375.29 seconds**; every selected model reached 320 iterations. This is a complete negative development screen against H2/P021. It does not nominate or promote a policy. Full confusion counts, P016 diagnostics, resource records and predictions are retained in the run directory.

### 48.2 Remaining cards — execution in progress

The RelGNN-inspired operator, empty-route/missing-FK behavior, deterministic cross-route sampler and finite backward pass passed the hand-calculated feasibility checks. The real as-of graph passed its 90-day audit at a strict `S−4h` cutoff; setup and the real forward/backward check took **9.20 seconds**, far below the two-hour ceiling. The first width-128 August fit reached its 2,700-second cap after **2,703.39 seconds**, saving three complete epochs with inner validation AP 0.2852537, 0.2873451, and 0.2848851. Epoch 2 was best (0.2873451), but the fit remains incomplete and is not eligible for width selection or predictive comparison. Its authenticated checkpoint, epoch history and cap receipt are preserved. This is an execution limitation, not a predictive negative. The batch runner advanced to EXP-036-E; the remaining card outcomes will be added from their saved run receipts.

The current run's pre-fit receipts are in `outputs/overnight_five_20261023T0237TW/pre_fit_checks.json`; its live state is `progress.json`. The run-level eight-hour deadline and per-fit safety caps remain unchanged. No experiment is nominated or promoted by this execution record.

## 49. Overnight five execution — final measured state, October 23, 2026

**Correction:** §50 below supersedes this section's AP interpretation, initialization reproducibility and audit-coverage claims. The observed F1 values remain unchanged.

This section supersedes §48.2's in-progress status using the final fresh run `outputs/overnight_five_20261023T0340TW/`. The run used the refreshed instructor snapshot `2053302ba75751be975b9e779f567557b66f4071`, the 59-column F0 schema, seed 42, and the preregistered August tuning/September report protocol. H2/P021 was reproduced at `0.5476344735753119`. Four September report origins supplied 22,919 common user-origin rows. The source, configuration and brief hashes were frozen in the preregistration before fitting. No October outcome labels were built or read. The runner finished in 1:18:51.5 with no runner failures.

| Card | Final state | Pooled P021 combined F1 | Δ vs H2 | Mean six-target AP | Decision / next action |
|---|---|---:|---:|---:|---|
| EXP-023-S HGB more_rounds | Complete, verified | 0.543451 | −0.004183 | 0.480058 | Complete negative screen; no promotion. |
| EXP-036-E TabM embedding_off | Complete, verified matched control | 0.548058 | +0.000424 | 0.461769 | Retain as the diagnostic control; not blend eligible. |
| EXP-036-E TabM embedding_on | Complete, verified | 0.542990 | −0.004644 | 0.496160 | Complete negative standalone screen. |
| EXP-038-S RelGNN-inspired full_attention | Incomplete; first fit hit 2,700-second cap | — | — | — | Do not interpret or score the partial fit. Continue only after a named pre-score resource amendment, in a fresh run directory. |
| EXP-011-R CatBoost | Complete, verified | 0.544216 | −0.003419 | 0.486631 | Complete negative screen; stop this registered recipe. |
| EXP-041-S blend HGB | Complete, verified | 0.546196 | −0.001439 | 0.538253 | Complete negative blend; no promotion. |
| EXP-041-S blend TabM embedding_on | Complete, verified | 0.548002 | +0.000368 | 0.537885 | Retain the small development gain for review; confirmation is required. |
| EXP-041-S blend CatBoost | Complete, verified | 0.547256 | −0.000378 | 0.538925 | Complete near-baseline negative blend. |

The HGB tuning selected 320 rounds/15 leaves over the 80/63-leaf candidate, then completed all 24 report target fits at 320 rounds. TabM completed the matched on/off arms at the fixed width-128 recipe; embedding-on raised mean AP about 0.03439 versus its control while reducing P021 combined F1 about 0.00507. CatBoost completed all 24 report fits under its one registered CPU recipe. The RelGNN equations, tiny graph, finite backward pass and real as-of graph checks passed, but its width-128 August fit saved only three epochs before the cap; it did not reach width selection, controls, or report scoring. The eligible blend pool contained the complete HGB, TabM-on and CatBoost candidates; the RelGNN blend was not run and the TabM-off control stayed excluded.

All executed blends passed alpha-zero H2/P021 decision parity and official evaluator parity. Independent verification checked the 24-file frozen source snapshot, common 22,919-row key sets, finite probabilities, pooled F1 from target confusion counts, and official scoring on each P016/P021 report origin. Its result is `PASS_WITH_RELGNN_INCOMPLETE`. The full comparison, all four weekly deltas, target confusion metrics, verification receipts and exact re-entry commands are in [`reports/overnight_five_20261023T0340TW/summary.md`](../reports/overnight_five_20261023T0340TW/summary.md).

These are September development results. No submission policy is nominated or promoted. The only measured positive blend delta is the small `+0.000368` TabM-on gain; it is retained for later review and does not by itself authorize extended investment. The ongoing RelGNN family is not completed by this first-night checkpoint. Its partial checkpoint, history and cap receipt remain available; same-fit continuation is not scientifically scoreable, so any further trial requires an explicit pre-score resource/scope amendment and a fresh directory. Until then, leave the completed screen decisions as recorded above and follow §47's one-distinct-follow-up budget rather than expanding a completed negative grid.

## 50. Repair amendment and RelGNN throughput priority — October 23, 2026

Use the isolated `experiments.overnight_five_v2` runner for future execution, with a fresh directory and its own pre-fit source/config freeze. It seeds neural initialization before construction, reports both mean-origin and pooled AP with explicit score provenance, and supplies the stronger saved-run verifier. The original package/run/reports remain historical evidence. Historical neural fits retain their initialization limitation; a new matched run is required for reproducible treatment attribution, not an automatic rerun of the negative tree screens.

**AP correction to §49:** the prior AP column was pooled across report rows. Mean-origin raw-model AP is H2 `0.535030`, HGB `0.514149`, TabM-off `0.538016`, TabM-on `0.519857`, CatBoost `0.520236`. Embedding-on decreased mean-origin AP by `0.018159` relative to off, as well as decreasing F1 by `0.005068`. The claim that embeddings improved the established AP diagnostic is withdrawn. Keep mean-origin and pooled AP separately named, and distinguish standalone probability AP from final blended rank AP. None of the original F1 scores or promotion decisions changes.

**Next RelGNN work:** give the relational branch its bounded implementation-throughput follow-up before a larger backbone or blind cap extension. The exact first-epoch census found all 42 fitting origins in every 2,048-row shuffled batch: 177 batches and 7,434 separate graph encodings per epoch. A constant-256-row August diagnostic took 0.218/0.464/0.949 seconds for 1/4/8 origins, with similar sampled graph volume. Fragmented per-origin encoding is therefore a measured high-priority target; CPU graph sampling alone does not explain the increase.

The first optimization candidate is a disjoint-union representation of the sampled origin graphs within each existing shuffled logical batch, vectorizing work by node type/relation. Keep origin-specific cutoffs, node identities, deterministic neighborhoods, routes/fanouts, F0 branch, H2 row weights, the 2,048-row gradient objective, clipping and optimizer step boundaries. Do not create cross-origin edges or silently substitute origin-homogeneous optimizer batches. Check logits and weighted-loss/gradient parity with dropout disabled, edge membership/cutoffs, memory and representative complete-step throughput before a pre-score registration and fresh fit. Cache deterministic neighbor/index tensors only where their origin/source/seed keys are complete. Any deliberate optimization change requires its own named amendment.

Keep this engineering attempt within §47's protected relational block and one distinct follow-up budget. First measure a small implementation prototype; expand only if its throughput makes a complete chronological screen feasible. The measured 4.35× small-batch contrast is a diagnosis, not a promised optimization gain. Do not resume the capped historical fit, score its partial checkpoint, expand width, or spend an overnight budget merely because the original batch had unused time. No new training budget is activated by this diagnostic amendment. H2 remains the reference and submission fallback; confirmation rules and October outcome restrictions remain unchanged.

Repairs passed seven focused regression tests and a saved-run audit with 70 official-score checks, 20 bounded checkpoint replay records and all 24 frozen sources matching. See [repair/profile report](../reports/overnight_five_repair_20261023/REPAIR_AND_PROFILE_REPORT.md) and Experiment Log §48 for scope and limitations.
