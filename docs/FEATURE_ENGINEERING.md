# Feature Engineering and Evaluation Specification

**Project:** NYCU 535703 — Big Data Analytics Techniques and Applications, Assignment 2: Predicting GitHub User Activity  
**Prepared:** 2026-10-15  
**Purpose:** Maintain a reproducible, prioritized catalogue of *candidate* features and a defensible experimental procedure for maximizing the official six-label combined F1 score.  
**Status:** Reviewed research/design specification, **not** a claim that these features were implemented, measured, or shown to improve the score.
**Audit scope (2026-10-15):** Cross-checked against the supplied Assignment 2 PDF, the live instructor schema/data/evaluator, the cited methodological literature, and the user-supplied EXP-006 research report. This is a *specification audit*, not a data-level validation or model experiment.

> **Decision rule:** A promising mechanism is a reason to *test* a feature, not evidence of incremental predictive value. Feature construction, model training, threshold selection, and validation must be separated experimentally. In particular, never equate feature importance in one fitted model with a reproducible gain in future-week F1.

## Contents

1. [Assignment contract and evidence](#1-assignment-contract-and-evidence)
2. [Definitions, cutoff rules, and reusable operators](#2-definitions-cutoff-rules-and-reusable-operators)
3. [Candidate feature catalogue](#3-candidate-feature-catalogue)
4. [Feature groups and staged implementation](#4-feature-groups-and-staged-implementation)
5. [Credible evaluation methodology](#5-credible-evaluation-methodology)
6. [Exact ablation and feature-selection plan](#6-exact-ablation-and-feature-selection-plan)
7. [Audit, reproducibility, and release checklist](#7-audit-reproducibility-and-release-checklist)
8. [Research and authoritative references](#8-research-and-authoritative-references)

---

## 1. Assignment contract and evidence

### 1.1 Target, population, dataset and metric

- Predict six binary outcomes for each official `UserID` over **2026-11-12 00:00 to 2026-11-19 00:00 Asia/Taipei**: `open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`. One means **at least one occurrence** within the seven-day period, not the number of events. [A2, EVAL]
- Use **only the instructor-provided data**, as released in the official GitHub repository: `Users`, `Repositories`, `PullRequests`, `Issues`, `IssueComments`, `PRReviews`, `PRReviewComments`, `Pushes`. The fixed cohort has 9,000 users; other referenced users do not have complete recorded activity. [A2, DATA, SCHEMA]
- No additional GitHub API calls, external GH Archive events, scraping, outside repository activity, profiles, stars, followers or third-party event enrichment. Academic literature and generic algorithm/software documentation may guide *methods* without being used as *additional activity data*. [A2]
- Training/validation must be split **by time, not randomly**. For feature construction, include only observations knowable **strictly before** the simulated prediction cutoff. Late-known/mutable fields require separate safeguards. [A2, DATA]
- Official score is `combined = (macro-F1 + micro-F1) / 2` across the six outputs. The supplied evaluator returns `F1=1` when `TP+FP+FN=0`, rather than scikit-learn's default empty-class convention. Exactly reproduce the instructor's evaluator. [EVAL]
- The official output must have the exact columns `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`; all prediction values are binary and IDs must match the provided test list. [EVAL]
- The required presentation asks specifically how the method improves **relational modeling (Q2)** and **temporal modeling (Q3)**; practical positive/negative ablations are therefore valuable even if the final predictor uses a simpler subset. [A2]

**Baseline context from the user's supplied Deep Research report [RPT], not newly independently reproduced:** EXP-006 pooled eligible-proxy combined scores were persistence `0.454677`, HistGradientBoosting `0.398767`, logistic regression `0.381531`. The report proposes EXP-007/008 evaluation forensics *before* further feature search; its reported 59 original feature definitions/source signatures were not available to the researcher. The reported baseline feature count and scores have **not** been verified against your current source tree or rerun here. Treat the current feature set as `F0` only after inventory and scorer-parity checks.

### 1.2 What the literature does and does not establish

| Evidence | What it supports | What it does **not** establish |
|---|---|---|
| [GAP] git-contribution activity forecasting via survival analysis | Testing personal event gaps, typical recurrence, inactivity and next-event features | That such features improve these six labels or their F1 |
| [DRIFT] concept-drift survey and [TEMP-CV] temporal evaluation study | Recency-weighted training and repeated forward testing for nonstationary activity | A universal optimal lookback or sample-weight half-life |
| [DFS] Featuretools and [RDL] relational learning literature | Safe graph joins/aggregations can expose useful multi-table signals | A specific GitHub neighborhood aggregation or GNN must beat persistence |
| [TS] time-series feature library | Definitions for autocorrelation, entropy, trend and variability operators | That hundreds of automated statistics are useful on sparse activity series |
| [SELECT], [STABLE], [PFI] feature-selection/importance research | Evaluate relevance, stability, conditional/redundant information, and grouped importance | That impurity/SHAP/permutation ranks are equivalent to prospective score gain |
| [TEMP-CV], [CV-STRUCT], [NESTED] | Chronological, separated selection/evaluation; dependence-aware assessment | That overlapping weekly folds are independent trials |
| [F1] threshold theory, [PR] imbalance evaluation | Calibrate selection to F1, assess rare positives with PR-oriented diagnostics | That independently optimizing each target F1 maximizes the joint assignment metric |

**Provenance codes used below** (one or more codes on *every* catalogue row): `S` = directly constructible from official schema/rules [SCHEMA, DATA]; `R` = proposed in supplied Deep Research report [RPT]; `G` = mechanism suggested by git activity forecasting [GAP]; `T` = defined by general time-series/statistical feature literature [TS]; `D` = relational feature synthesis and RDL literature [DFS, RDL]; `H` = **our testable task-specific hypothesis, not independently validated**. A citation supports a *method or mechanism*, not a claimed uplift. All rows use `S` because the source must be the permitted dataset.

---

## 2. Definitions, cutoff rules, and reusable operators

### 2.1 Atomic observations and historical availability

Define `u` as a stable `Users.github_user_id`, `r` as a stable `Repositories.github_repo_id`, `a` as one of the six targets, and `t` as the **feature cutoff in UTC**. Join row-index FKs **within one pinned snapshot** and map to stable `github_*_id` when saving cross-snapshot entities; per-release row indices may change. Define source-row visibility `v(e)` exactly as follows:

| Table | Row-visibility time `v(e)` | Key relationship paths | Specific cautions |
|---|---|---|---|
| Users | `first_seen_at` | `github_user_id`, `user_id` | `login` is latest-seen and may not be historically reconstructible; avoid as-of lexical/bot-name features without proof |
| Repositories | `first_seen_at` | `github_repo_id`, `repo_id` | Repository metadata and name/owner may not have historical rename records; stable repo ID preferred |
| PullRequests | `first_seen_at` | `author_id → Users`, `repo_id → Repositories`; `merged_by_id → Users` later | `created_at` can predate visibility; `merged_at`, `closed_at`, `merged_by_id` known later; some opening/title metadata may arrive after first PR visibility |
| Issues | `first_seen_at` | `author_id`, `assignee_id`, `repo_id`, `pull_request_id` | Includes PR conversations (`is_pull_request=true`); `created_at` may predate visibility; `closed_at`, `state_reason` known later |
| IssueComments | `created_at` | `author_id`, `issue_id`, `repo_id` | Distinguish comments on issues from PR conversations |
| PRReviews | `first_seen_at` | `reviewer_id`, `pr_id`, `repo_id` | `submitted_at` may predate visibility; `dismissed_at` known later |
| PRReviewComments | `created_at` | `author_id`, `pr_id`, `pull_request_review_id`, `in_reply_to_id`, `repo_id` | Parent/review edges require independently visible referenced rows |
| Pushes | `created_at` | `user_id`, `repo_id` | Contains `ref`, `branch`, `ref_type`, `is_new_ref`; this is a **push event**, not commit-count data |

**Hard invariants:** For every contributing row, `v(e) < t` (where `t` is the chosen exclusive *feature cutoff*). For relationships, **both/all endpoints must be visible**, and later-known attributes are masked when unavailable. Use `first_seen_at`, not `created_at`/`submitted_at`, to decide feature visibility for PRs, Issues and Reviews. Never treat retrospective issue/PR creation dates as permission to use those rows in earlier snapshots. [A2, DATA, SCHEMA]

**Do not equate row visibility with true target events.** A PR/issue/review discovered today may have actually been created/submitted long ago. An event series built from `first_seen_at` captures **first observation**, not necessarily opening/review occurrence. Keep two separate families and column-name namespaces: `(i)` `observed_pr/issue/review_first_seen_*` for safe as-of-observation signals, and `(ii)` `actual_open_pr/open_issue/review_submitted_*` **only** if precise event occurrence *and its historical discoverability* can be established. The supplied flattened tables do not guarantee that every opening/submission was directly observed, nor that the author/title/draft fields of a PR were available at initial PR-row visibility. When unavailable, omit the genuine-event feature; never quietly substitute `first_seen_at` while preserving an actual-event name. Historical target-label construction requires its own proof and/or sensitivity analysis.

**Different clocks must not be conflated:** (1) actual activity time (target-week membership where recoverable), (2) row-visibility time (whether features could know the row), (3) historical label-observation/as-of cutoff (whether a reconstructed label was mature), and (4) model fit time (what labels/training inputs were actually available). A historical weekly target reconstructed from *today's final tables* is not automatically a legitimate label available at the earlier simulated fit time. [RPT]

**Sparse/partial entities:** A missing recorded interaction is *not* proof that no such real-world interaction happened. Noncohort activity is incomplete. A feature's denominator must be a count of **observed eligible opportunities**, and a coverage/missingness companion should accompany sparse relation statistics.

**Final blind interval and two distinct boundaries:** The instructor's final dataset covers events through approximately **2026-11-11 20:00 Asia/Taipei**, while the prediction label interval starts at **2026-11-12 00:00 Asia/Taipei** and ends at **2026-11-19 00:00**. Define `c` = maximum *usable observed feature time* and `s` = *future target start*; for deployment `s−c≈4 hours`. Feature windows use an **exclusive** cutoff `c`: `visible_at < c`, where `c` is chosen after the last eligible timestamp in the pinned release and no later than the stated coverage boundary (respect available timestamp precision). The label is determined by real target-event occurrence in `[s,s+7d)`, subject to reconstruction limits. For historical replays, select paired `(c_i,s_i)` with the **same gap and weekday/time alignment**. A plain `count_7d` feature is `[c_i−7d,c_i)`, *not* the complete `[s_i−7d,s_i)` preceding outcome week; the latter includes four unobserved hours and must never be computed using those events. Do not set `c` equal to the final included event timestamp and then exclude that event by accident: distinguish the last observed event timestamp from the first unavailable feature-time boundary. [A2, MAIN, EVAL]

### 2.2 Notation and exact operators

| Symbol/operator | Definition |
|---|---|
| `E(u,a,t,W)` | Observed/visible source records associated with `a`, with visibility times in `[t−W,t)`; **not automatically true occurrence events for `open_pr`, `open_issue`, or `pr_review`** |
| `n_a(W)` | `card(E(u,a,t,W))`, **visible-source-record count**, not necessarily true openings/review submissions and not commit count |
| `b_a(W)` | `1[n_a(W)>0]` |
| `d_a(W)` | Number of **distinct active UTC dates** in `E(u,a,t,W)`; optionally add local-Taiwan-date variant |
| `h_a` | Elapsed days since last *visible source observation* associated with `a` before `t`; missing if never observed |
| `w_{a,j}` | Event count for `a` in `[t−7(j+1)d,t−7jd)`, `j=0` most recent |
| `z_{a,j}` | `1[w_{a,j}>0]`; week-level binary activity |
| `C(W)` | Visible events of all six target activities in `[t−W,t)` |
| `R_u(W)` | Set of repositories linked to visible activity of `u` in `[t−W,t)` |
| `R^*` / `U^*` | Repository/user sets **restricted to rows and edges visible before t** |
| `safe_ratio(x,y)` | `x/(y+ε)` with a declared `ε` (usually `1` for counts; otherwise e.g. `1e-6`), plus denominator-zero indicator where meaningful |
| `rate(x,W)` | `x / W` for a duration in days; normalize unequal windows before taking rate ratios |
| `EW(x_i, τ)` | `Σ_i exp(−age_days_i/τ) x_i`, with `age≥0` measured from `t`; for events `x_i=1` |
| `entropy(c_1,…,c_k)` | `−Σ_{i:c_i>0} p_i log(p_i)`, `p_i=c_i/Σ c_i`; output `0` for no mass and add sample-count coverage |
| `∆`, `CV` | `∆=new−old`; `CV=std/mean` only for positive mean (otherwise missing + flag) |
| `N_u(r,W)` | Number of visible interactions by `u` with repo `r` in `[t−W,t)`; neighbor activity is measured **excluding u** |

**Recommended exploratory parameter grids** (not established optima): event windows `W ∈ {1,3,7,14,28/30,56,90}` days, weekly lags `j=0..7`, decay `τ ∈ {7,14,30}` days, rare-event smoothing strength `α∈{5,20,50}` selected **inside chronological training/selection only**. Large Cartesian products are not necessary: begin with a prespecified small subset per feature group.

For historical weekly states or transition probabilities, distinguish **source activity indicators** (constructible at the prediction cutoff) from **supervised future-week labels** (usable only after the full outcome window and observation/maturity buffer). All supervised-derived statistics require earlier available labels and strictly forward-fitting/OOF encoding. Prefer unsupervised lagged activity flags when this distinction is uncertain.

### 2.3 Critical join semantics

- `IssueComments.issue_id → Issues.issue_id`: an `IssueComments` event is attributed to `issue_comment` even if `Issues.is_pull_request=true`; `open_issue` specifically excludes `Issues.is_pull_request=true`. A PR discussion comment is **not** a PR-review-comment event.
- `PRReviewComments` and `PRReviews` are different target categories; never count line-level comments as review submissions unless a separate visible `PRReviews` row establishes that target event.
- `PullRequests.head_repo_id` is a **GitHub repo ID**, whereas `PullRequests.repo_id` is an **internal FK row index**. Compare `head_repo_id` with joined `Repositories.github_repo_id`, not with `repo_id`. **Verified documentation inconsistency:** the supplied schema describes `is_cross_repo` as derived from `head_repo_id != repo_id`, which compares *different ID namespaces*. Disable the delivered `is_cross_repo` flag pending empirical verification; reconstruct a separate `cross_repo_verified = (head_repo_id != Repositories.github_repo_id)` after joining, only when both stable IDs are known, and compare the two flags on sampled rows. Do not assume their mismatch proves how the parquet generator is implemented, only that the published derivation is unsafe as written.
- `PRReviews.state` describes the recorded submitted review state, but `dismissed_at` cannot influence an earlier prediction. For `PullRequests.author_id`, `draft`, `title` and opening metadata that can arrive through a later observed conversation, require field-level availability validation; simply checking the PR row's `first_seen_at` may not suffice.
- For any `feature(user,repo,...)`, save the stable IDs and deduplicate by canonical event ID where possible (`event_id`, `github_push_id`, etc.) to prevent many-to-many join multiplication.

---

## 3. Candidate feature catalogue

**How to read:** `ID` is a conceptual feature definition (some expand into six targets and multiple windows). `Formula/build` is implementation-level guidance. `Sources` refers to official tables or the above provenance codes, plus a concise hypothesis; **H** means task-specific extrapolation. `Tier`: `0` low-cost/high-priority, `1` next, `2` exploratory. **No feature below is asserted to be already absent from F0 or empirically useful.**

### G01 — Baseline, event presence and multiscale activity (`F-TEMP-BASE`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| T01 | `count_{a}_{W}d` | `n_a(W)` for each `a`, `W=1,3,7,14,30,90` | `S,R`: direct counts; capture workload at several horizons | 0 |
| T02 | `any_{a}_{W}d` | `1[n_a(W)>0]`, especially `7d` | `S,R`: exact historical state mimicked by strong persistence | 0 |
| T03 | `active_days_{a}_{W}d` | Distinct calendar dates on which ≥1 visible event occurred within W | `S,T,H`: one burst vs recurring days | 0 |
| T04 | `activity_rate_{a}_{W}d` | `n_a(W)/W`; compare at equal units | `S,R`: rate normalizes window lengths | 0 |
| T05 | `total_events_{W}d` | `Σ_a n_a(W)`; six activity types only | `S,R`: global engagement; distinguishes dormant from busy | 0 |
| T06 | `num_active_targets_{W}d` | `Σ_a b_a(W)`, integer 0–6 | `S,H`: breadth of work categories | 0 |
| T07 | `nonzero_event_type_fraction` | `(Σ_a b_a(W))/6` | `S,H`: normalized behavioral diversity | 1 |
| T08 | `target_share_{a}_{W}d` | `n_a(W)/(1+Σ_b n_b(W))` | `S,H`: target specialization controlling overall volume | 0 |
| T09 | `target_entropy_{W}d` | `entropy(n_1(W),…,n_6(W))` | `S,T,H`: concentration vs diversified workflows | 1 |
| T10 | `event_count_log_{a}_{W}d` | `log1p(n_a(W))`; retain raw count for trees | `S,H`: stabilize high-count skew for linear models | 1 |
| T11 | `first_observed_age_days_{a}` | `(t−min visible time of a for u)` in days; NaN if none | `S,H`: length of target-specific observation history | 1 |
| T12 | `observed_user_tenure_days` | `(t−Users.first_seen_at)` if user visible; coverage flag | `S,H`: observed user history varies by cohort appearance | 0 |

**Implementation note:** T01–T57 and X01–X14 use `{a}` as a *visible-observation proxy* unless an audited event-occurrence stream exists. For `open_pr`, `open_issue`, and `pr_review`, name such columns `pr_first_seen_*`, `nonpr_issue_first_seen_*`, and `review_first_seen_*`; do not reuse official target names for historical observability proxies. For an actual-opening/submission variant, prove the event occurred in the window **and** was knowable at the feature cutoff. Use separate coverage features and test both representations when possible. `PRReviews.submitted_at` may differ from `first_seen_at`.

### G02 — Recency, rhythms, burstiness and inactivity (`F-TEMP-RHYTHM`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| T13 | `days_since_last_{a}` | `h_a`, plus explicit `never_seen_{a}` flag | `S,R,G`: recent-event recurrence and dormant state | 0 |
| T14 | `days_since_second_last_{a}` | `t − second most recent visible event` | `S,G,H`: distinguishes recurring users from one-offs | 0 |
| T15 | `median_event_gap_{a}` | Median gap between successive *visible observation timestamps* among last K=10/20 records; use actual event times only for a separately audited occurrence stream | `S,G`: personalized recurrence interval | 0 |
| T16 | `mean_event_gap_{a}` | Mean of same successive gaps, minimum 2 events | `S,G`: long-run event frequency | 1 |
| T17 | `gap_cv_{a}` | `std(gaps)/mean(gaps)`; NaN/flag if insufficient data | `S,G,T`: regular vs irregular recurrence | 1 |
| T18 | `gap_p90_{a}` | 90th percentile event gap on observed history | `S,G,H`: high-side normal inactivity | 1 |
| T19 | `normalized_inactivity_{a}` | `h_a/max(ε_gap,median_gap_a)` using a predeclared positive gap floor in **days** (e.g. 1/24 for an hourly floor), plus zero/insufficient-gap support flag | `S,G,H`: inactivity relative to personal norm | 0 |
| T20 | `last_gap_vs_typical_{a}` | `(last_event−second_last_event)/max(1,median historical gap)` | `S,G,H`: recent rhythm disruption | 1 |
| T21 | `max_inactive_run_days_{a,30}` | Maximum consecutive dates with zero activity within last 30 calendar days | `S,T,H`: distinguish persistently dormant vs episodic | 1 |
| T22 | `current_inactive_run_days_{a}` | Consecutive zero-activity days immediately preceding cutoff | `S,G`: length of present inactivity episode | 0 |
| T23 | `daily_count_cv_{a,28}` | Std/mean of 28 daily event counts, with guard at mean=0 | `S,T`: variability/burstiness | 1 |
| T24 | `max_daily_events_{a,30}` | Maximum daily visible event count in last 30 days | `S,T`: concentrated pushes/reviews | 1 |
| T25 | `top_day_event_share_{a,30}` | `max(daily counts)/(1+n_a(30))` | `S,T,H`: one-day spike vs routine work | 1 |
| T26 | `burst_days_{a,30}` | Count days above user's *prior* trailing historical daily p90, with threshold computed before t | `S,T,H`: unusually intense days | 2 |
| T27 | `active_day_fraction_{a,28}` | `active_days_{a,28}/28` | `S,G`: habitual presence regardless of volume | 0 |
| T28 | `events_per_active_day_{a,28}` | `n_a(28)/max(1,active_days_{a,28})` | `S,H`: heavy sessions vs numerous sessions | 1 |

**Event-gap warning:** For `open_issue`, `open_pr`, `pr_review` if event times are retrospectively reconstructed, gaps based on `first_seen_at` reflect **observation/discovery cadence**, not necessarily genuine action cadence. Label accurately and keep as a separate ablation; do not present these as true behavioral intervals without evidence.

### G03 — Week-state sequences, transitions and survival-style statistics (`F-TEMP-STATE`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| T29 | `weekly_count_{a,lag_j}` | `w_{a,j}` for j=0…7 | `S,R`: explicit sequence rather than pooled count | 0 |
| T30 | `weekly_active_{a,lag_j}` | `z_{a,j}` | `S,R`: persistence and weekly on/off states | 0 |
| T31 | `active_weeks_{a,4/8}` | `Σ_{j=0}^{K−1} z_{a,j}`, K=4,8 | `S,R`: habit frequency | 0 |
| T32 | `current_active_week_streak_{a}` | Number of consecutive ones from `z_{a,0}` backwards | `S,R`: sustained activity | 0 |
| T33 | `current_inactive_week_streak_{a}` | Number of consecutive zeros from `z_{a,0}` backwards | `S,R,H`: dormant sequence | 0 |
| T34 | `weekly_slope_{a,4}` | Ordinary least-squares slope of chronologically ordered `[w_3,w_2,w_1,w_0]` vs `[0,1,2,3]` | `S,R,T`: recent trajectory | 0 |
| T35 | `week_count_std_{a,8}` | Std of `[w_0,…,w_7]` | `S,T`: episodic vs steady weekly volume | 1 |
| T36 | `week_state_switches_{a,8}` | `Σ_{j=0}^{6} 1[z_{a,j} ≠ z_{a,j+1}]` | `S,H`: stability of activity state | 1 |
| T37 | `last_on_to_off_count_{a}` | Historical count of observed `1→0` transitions within 8 mature past weeks | `S,H`: likelihood of dropping activity | 1 |
| T38 | `last_off_to_on_count_{a}` | Historical count of observed `0→1` transitions within 8 mature past weeks | `S,H`: propensity to reactivate | 1 |
| T39 | `p_on_given_on_{a}` | `(N_{11,a}+α p_a)/(N_{1*,a}+α)`, using transitions of **fully observed earlier weeks** | `S,R,G,H`: recurrence probability; smoothed against low support | 0 |
| T40 | `p_on_given_off_{a}` | `(N_{01,a}+α p_a)/(N_{0*,a}+α)` | `S,R,G,H`: reactivation probability | 0 |
| T41 | `empirical_active_prob_{a,K}` | `(Σ_{j=0}^{K−1} z_{a,j}+α p_a)/(K+α)`, K=4/8/12 | `S,R,H`: stabilized historical activity probability | 0 |
| T42 | `reactivation_by_gap_bucket_{a}` | Historical rate of next-week onset when preceding inactive streak belonged to bin 0,1,2,3+; smoothing/coverage | `S,G,H`: non-Markov dormant recovery | 2 |
| T43 | `next_7d_empirical_hazard_{a}` | Fit survival/discrete hazard on prior completed event-gap histories; estimate `P(event in next 7d given current age)` via prior mature intervals only | `S,G,H`: directly targets probability of next event | 2 |
| T44 | `week_autocorr_{a,1/2}` | Autocorrelation of prior ≥8 weekly counts at lag 1 or 2; NaN when constant | `S,T`: repeated weekly cadence | 2 |
| T45 | `lagged_target_seasonality_{a}` | Compare mean counts at past same weekday/week position vs overall; only if enough history | `S,T,H`: weekly schedule/cadence; sparse-data risk | 2 |

**Important:** `p_on_*` and `empirical_active_prob` can be computed from **past source-activity states** if well-defined. If built from reconstructed *future-week supervised targets*, only use fully matured outcomes as of the simulated model fit time; do not use a test-week answer to compute a test-week predictor. `p_a` and smoothing coefficients are computed from earlier eligible training data, not from current reporting labels.

### G04 — Momentum, normalized rates, decay and personal seasonality (`F-TEMP-MOMENTUM`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| T46 | `recent_vs_prior_rate_{a,7v21}` | `(n_a(7)/7)/((n_a(28)−n_a(7))/21 + ε)` | `S,R,T`: separates acceleration from overall intensity | 0 |
| T47 | `recent_minus_prior_rate_{a}` | `n_a(7)/7 − (n_a(28)−n_a(7))/21` | `S,R`: growth without ratio instability | 0 |
| T48 | `short_to_long_{a}` | `n_a(7)/(1+n_a(30))` | `S,R`: proportion of actions occurring recently | 0 |
| T49 | `rolling_ema_count_{a,τ}` | `Σ_{e∈E(u,a,t,90)} exp(−age(e)/τ)`, τ=7,14,30d | `S,R,G`: smooth age-weighted events | 0 |
| T50 | `ema_fast_minus_slow_{a}` | `EW_a(τ=7)/7 − EW_a(τ=30)/30` (rescale consistently) | `S,T,H`: momentum crossover; normalization requires audit | 1 |
| T51 | `activity_acceleration_{a}` | `(rate last7−rate preceding7)−(rate preceding7−rate previous7)` | `S,T,H`: second difference | 2 |
| T52 | `weekday_event_fraction_{a,28}` | Visible weekday (Mon–Fri) events divided by total | `S,T,H`: business-day rhythm | 1 |
| T53 | `weekend_active_day_fraction_{a,56}` | Weekend active dates divided by # weekend dates in 56-day window | `S,T,H`: sustained off-hours/weekend routine | 1 |
| T54 | `hour_of_day_concentration_{a,30}` | Max event-hour histogram bin / total; UTC or Taiwan consistently | `S,T,H`: contribution schedule stability | 2 |
| T55 | `business_hours_share_{a,30}` | Count at predeclared local hours (e.g. 09–18) / total | `S,H`: regular development schedule; timezone hypothesis only | 2 |
| T56 | `event_hour_entropy_{a,30}` | Entropy of 24-bin hour histogram | `S,T,H`: concentration vs round-the-clock activity | 2 |
| T57 | `cross_target_ratio_{a,b,30}` | `n_a(30)/(1+n_b(30))` for **preselected pairs**, not all Cartesian pairs | `S,H`: comparative specialization | 1 |

**Seasonality caution:** Individual timezone is *not* present; local-Taiwan hour is a fixed feature convention, not a claim about a developer's real location. Calendar effects may be more useful for cohort-level activity than for individual users.

### G05 — Cross-target states, associations and workflow transitions (`F-CROSS`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| X01 | `pair_any_{a,b,7/30}` | `b_a(W)*b_b(W)` for selected workflow pairs | `S,H`: co-occurring behavioral modes | 0 |
| X02 | `push_pr_count_ratio_30` | `n_push(30)/(1+n_pr_first_seen(30))`; use actual open-PR denominator *only* when independently verified; label the proxy accordingly | `S,H`: development-to-PR conversion signature | 0 |
| X03 | `reviews_to_line_comments_30` | `n_pr_review(30)/(1+n_pr_review_comment(30))` | `S,H`: review workflow/style | 0 |
| X04 | `pr_comment_to_issue_comment_share` | Visible IssueComments on PR conversations / `(1+all IssueComments)` in 30d | `S,H`: PR conversation specialization | 0 |
| X05 | `push_without_pr_flag_14` | `1[n_push(14)>0 and n_pr_first_seen(14)=0]` as an *observation proxy*; cannot claim absence of actual PR openings | `S,H`: coding that may precede PR creation | 0 |
| X06 | `review_without_comment_flag_14` | `1[n_pr_review(14)>0 and n_pr_review_comment(14)=0]` | `S,H`: distinct review modes | 1 |
| X07 | `transition_a_to_b_7d` | For each **historical** visible a-event, count a subsequent b-event ≤7d later (strict order) | `S,H`: workflow sequence; define pairs explicitly | 1 |
| X08 | `transition_rate_a_to_b` | `#a-events followed by b within 7d / #a-events with complete 7d follow-up before t` | `S,H`: personal propensity, avoids right censoring | 1 |
| X09 | `mean_transition_lag_a_b` | Mean positive lag (hours/days) for matched prior a→b occurrences | `S,H`: workflow timing | 2 |
| X10 | `cooccur_weekly_count_a_b` | Number of past 8 fully observable weeks with `z_{a,j} z_{b,j}=1` | `S,H`: persistent multi-activity pairing | 0 |
| X11 | `conditional_b_after_a_week` | Smoothed `#historical a-active weeks followed by b-active week / #historical a-active weeks` | `S,H`: cross-target state forecast | 1 |
| X12 | `cross_activity_entropy_7v30` | `entropy(n_a(7) by a) − entropy(n_a(30) by a)` | `S,T,H`: changing work mix | 1 |
| X13 | `last_action_type_onehot` | One-hot of the most recent *visible* event's target type; also its age | `S,H`: most recent workflow state | 0 |
| X14 | `non_target_total_after_last_a` | # observed *other-type* events after latest a-event and before t | `S,H`: whether user's workflow shifted since a | 2 |

**Suggested preselected pairs:** `push→open_pr`, `open_pr→issue_comment`, `pr_review→pr_review_comment`, `issue_comment→pr_review`, `open_issue→issue_comment`, `pr_review_comment→pr_review` (last pair is association, **not** necessarily causal). Each pair must be evaluated rather than assumed. Counting `a→b` on different repos/users is only a user-level sequence; add a *same-repo/same-PR* restriction where supported to test genuine workflow adjacency.

### G06 — Repository breadth, concentration, continuity and specialization (`F-REPO-SELF`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| R01 | `distinct_repos_{W}d` | `card(R_u(W))`, W=7,30,90 | `S,R,D`: work breadth beyond raw activity volume | 0 |
| R02 | `top_repo_share_{W}d` | `max_r N_u(r,W)/(1+Σ_r N_u(r,W))` | `S,R,D`: one-project concentration | 0 |
| R03 | `repo_entropy_{W}d` | `entropy({N_u(r,W)})` | `S,R,D`: project diversity | 0 |
| R04 | `repo_effective_number_30` | `exp(repo_entropy_30)` | `S,T,H`: interpretable diversity scale | 1 |
| R05 | `repeated_repo_fraction_30` | # recently active repos observed for same user *before* `t−30d` / # recently active repos | `S,R`: returning to established work | 0 |
| R06 | `new_repo_fraction_30` | # user-repo pairs whose **first observed interaction** is in last30d / # active repos | `S,R`: exploration vs continuity | 0 |
| R07 | `same_repo_7v30_jaccard` | `card(R_u(7)∩R_u([t−30,t−7)))/card(R_u(7)∪R_u([t−30,t−7)))`; empty union→missing/flag | `S,H`: continuity of project focus | 1 |
| R08 | `top_repo_changed_flag` | `1[argmax_r N_u(r,7) ≠ argmax_r N_u(r,[t−30,t−7))]` if both exist | `S,H`: primary-project switch | 2 |
| R09 | `distinct_orgs_30` | Unique known `Repositories.org_id` among active repos; separate unknown count | `S,R`: organizational breadth; incomplete metadata | 1 |
| R10 | `owned_repo_fraction_30` | Fraction active repos where source `owner_login` matches historically verified user login; **omit if rename history unavailable** | `S,H`: own vs outside repos; high identity risk | 2 |
| R11 | `user_repo_event_count_max_30` | `max_r N_u(r,30)` | `S,H`: core project intensity | 1 |
| R12 | `repo_activity_type_diversity` | Mean/weighted # distinct visible activity types per user-repo in 30d | `S,D,H`: breadth of workflow within projects | 1 |
| R13 | `repo_return_gap_median` | Median days between successive observed interactions of u with same repo (require support) | `S,G,H`: stable vs sporadic repository participation | 2 |
| R14 | `repo_refocus_rate` | Share of last7d interactions assigned to repos absent from previous 21d | `S,H`: project-switch/novelty signal | 1 |

### G07 — Repository and cohort-neighbor momentum (`F-REPO-NEIGHBOR`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| R15 | `repo_recent_total_events` | For each r in `R_u(30)`, count visible **cohort** events on r in last7d excluding u; user-weighted mean | `S,R,D`: activity near user's projects | 0 |
| R16 | `repo_neighbor_active_users` | # distinct **other cohort users** with visible events on r in 7/30d; aggregate over u's repos | `S,R,D`: nearby observed collaborator count | 0 |
| R17 | `neighbor_event_7v30_rate_ratio` | Excluding u, mean/weighted `(events last7/7)/(events prior23/23+ε)` across linked repos | `S,R,H`: project momentum | 0 |
| R18 | `self_share_of_repo_events` | `N_u(r,30)/(1+total cohort events on r in 30d)`, activity-weighted across r | `S,H`: key contributor vs peripheral involvement | 1 |
| R19 | `neighbor_push_rate_7` | Cohort neighbors' `push` events on u's recent repos / 7; exclude u | `S,H`: active codebase demand | 1 |
| R20 | `neighbor_review_rate_7` | Neighbors' visible PRReview records on u's recent repos / 7 | `S,H`: current review activity of projects | 1 |
| R21 | `neighbor_issue_comment_rate_7` | Neighbors' visible IssueComments on u's recent repos / 7 | `S,H`: discussion activity of projects | 1 |
| R22 | `neighbor_positive_fraction_7` | # eligible neighbor cohort users with given observed activity / # observed eligible neighbors; no full-network inference | `S,H`: local activity ratio | 1 |
| R23 | `repo_active_neighbor_retention` | # other cohort users active in both prior 28d and last7d / # prior cohort active users | `S,H`: persistence of collaborative team | 2 |
| R24 | `repo_max_recent_acceleration` | Max across user's recent repos of change in neighbors' daily event rates | `S,H`: one surging project | 2 |
| R25 | `weighted_repo_trend` | `Σ_{r∈R_u(30)} p_{u,r} trend_r`, `p_{u,r}=N_u(r,30)/Σ N_u` | `S,R,H`: relative relevance of project trend | 0 |
| R26 | `repo_neighbor_per_own_event` | `neighbor_events_{u,30}/(1+own_events_{u,30})` | `S,H`: collaboration opportunity independent of personal volume | 1 |

**No false completeness:** Neighbor rows count only events present in the supplied instructor dataset, predominantly from cohort users. Never interpret `repo_total_events` as the repository's full GitHub traffic. All repo signals must be computed at the cutoff with `u`'s own events subtracted where the feature is specifically **neighbor** activity; otherwise own-volume can masquerade as relational gain.

### G08 — PR, branch and code-change workflow (`F-PR-WORKFLOW`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| P01 | `branch_pushes_7/14/30` | Count `Pushes` with `ref_type='branch'` over W | `S,H`: ongoing branch development | 0 |
| P02 | `new_ref_pushes_7/14` | Count visible pushes with `is_new_ref=true` in W | `S,H`: potential new line of work | 0 |
| P03 | `distinct_branches_30` | Distinct `(github_repo_id, branch)` for u's pushes in W | `S,H`: breadth of active branches | 0 |
| P04 | `non_default_named_branch_share` | Fraction pushed branches not literally named `main` or `master`; explicitly **not** a verified default-branch indicator | `S,H`: feature-branch heuristic | 2 |
| P05 | `tag_push_fraction_30` | # `Pushes.ref_type='tag'` / `(1+all pushes)` | `S,H`: releases/tags rather than ordinary coding | 1 |
| P06 | `new_ref_fraction_30` | `new_ref_pushes_30/(1+pushes_30)` | `S,H`: branch creation vs updates | 0 |
| P07 | `pr_first_seen_count_30` | # PR rows authored by u first visible within W, **only if `author_id` was available at that first observation**; otherwise exclude/mark unknown; not PR openings | `S,R`: project involvement, field-level author timing unproven | 2 |
| P08 | `pr_open_events_if_verified_30` | Count PR openings in W *only if opening occurrence and as-of visibility can be certified* | `S,H`: direct true-open event predictor | 1 |
| P09 | `pr_recent_open_state_count` | Among PRs visible and author-attribution valid at t, count those with **no `closed_at < t`**; mask future `closed_at` entirely, with separate known-closed/unknown coverage | `S,H`: ongoing authored PR workload; late field mask | 1 |
| P10 | `known_pr_merge_fraction_90` | # authored PRs with known `merged_at<t` / # authored visible PRs old enough for observation; coverage | `S,H`: prior completion tendency | 2 |
| P11 | `pr_discussions_per_authored_pr_30` | Historical IssueComments linked to authored PRs / `(1+# eligible visible authored PRs)` | `S,H`: likely continued PR discussion | 1 |
| P12 | `review_feedback_on_authored_pr_14` | # visible PRReview records referencing PRs by u, excluding u, in W | `S,H`: recent external feedback may prompt push/comment | 1 |
| P13 | `cross_repo_pr_fraction_90` | After joining PR target `repo_id` to Repositories, # PRs with both stable IDs known and `head_repo_id != github_repo_id` / # PRs with both known; **never trust raw `is_cross_repo` without audit** | `S,H`: fork-based contribution style; audit source type | 2 |
| P14 | `draft_pr_fraction_90` | # PRs with *verified as-of* `draft=true` / # PRs with observed draft values | `S,H`: WIP PR behavior; field availability high risk | 2 |
| P15 | `pr_branch_link_density` | Fraction of u's push branch names matching visible PR `head_ref` **and the corresponding repo/fork mapping**, not string match alone | `S,H`: pipeline push→PR; possible fork confusion | 2 |

**Strict limitation:** The dataset has `push` **event** rows, not a reliable number of commits/files/lines changed per push. Do not create `commit_count`, `code_churn`, or git-diff-size features that require unavailable push payloads or external API calls. `github_push_id` is not a guaranteed globally unique event row across all situations: prefer the documented `event_id` when present, or a validated stable composite key, for deduplication. Do not collapse legitimate distinct pushes.

### G09 — Issue and PR-conversation workload (`F-ISSUE-WORKFLOW`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| I01 | `visible_nonpr_issues_authored_30` | # `Issues.is_pull_request=false`, `author_id=u`, `first_seen_at` in W; not synonymous with true openings | `S,H`: issue-related workload | 0 |
| I02 | `issues_opened_if_verified_30` | Count verified issue **opening events** in W with validated visibility/as-of | `S,H`: direct issue-opening history | 1 |
| I03 | `known_open_assigned_issues` | Count visible non-PR issues assigned to u not yet known closed as of t; coverage flag | `S,H`: potential comment/management work | 1 |
| I04 | `recent_issues_commented_count_30` | Distinct issue IDs on which u authored visible IssueComments in W | `S,R,H`: active discussion threads | 0 |
| I05 | `pr_conversation_comment_fraction` | Comments with `is_pull_request=true` / `(1+all u's IssueComments)` in W | `S,H`: PR vs bug/task discussion style | 0 |
| I06 | `comments_per_touched_issue_30` | # issue comments by u / `(1+distinct issues commented on)` | `S,H`: thread engagement depth | 1 |
| I07 | `repeat_comment_thread_fraction` | Share of commented issues on which u posted ≥2 visible comments by t | `S,H`: continued conversations | 1 |
| I08 | `recent_issue_labels_count` | Distinct label tokens attached to first-visible *non-PR* issues u authored/interacted with, where actually present | `S,H`: type diversity, incomplete labels | 2 |
| I09 | `issue_type_distribution` | Counts/proportions of observed `issue_type` (Bug/Feature/Task/unknown) within visible user-related issues | `S,H`: workflow subtype | 2 |
| I10 | `issue_age_since_first_seen` | Median `t−Issues.first_seen_at` for visible issues u participates in | `S,H`: current observed thread age, **not creation age** | 1 |
| I11 | `recent_issue_closed_before_t` | # visible user-involved issues with `closed_at` in `[t−W,t)`; only observed historical closes, never the final-snapshot state reason | `S,H`: recent workflow resolution; field mask | 2 |
| I12 | `issues_with_known_milestone_fraction` | # visible eligible issues with `milestone_title` available / # eligible issues | `S,H`: scheduled work; first-seen metadata caveat | 2 |

### G10 — PR reviews, reviewer–author pairs and thread behavior (`F-REVIEW-COLLAB`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| V01 | `review_submissions_7/30/90` | Count visible PRReviews by `reviewer_id=u` in W; distinguish observed vs submitted time | `S,R`: direct review history | 0 |
| V02 | `review_state_distribution_90` | Counts of `state` in `approved`, `commented`, `changes_requested` for prior visible reviews | `S,H`: developer review style | 1 |
| V03 | `distinct_reviewed_prs_30` | Distinct PR IDs reviewed by u | `S,H`: review breadth | 0 |
| V04 | `distinct_reviewed_authors_90` | Distinct visible `PR.author_id` for u's reviewed PRs; missing author excluded + coverage | `S,R,D`: reviewer breadth | 1 |
| V05 | `repeat_reviewer_author_count` | Number of earlier visible reviews u performed of PRs by a **previously seen** author | `S,R,D`: repeat collaboration | 1 |
| V06 | `top_author_review_share_90` | Max reviewed-PR count per author / all eligible reviews | `S,H`: reviewer specialization | 1 |
| V07 | `reviewer_author_recency` | Days since last visible review by u on a PR from a known prior author | `S,H`: pair-level recency | 2 |
| V08 | `line_comments_7/30` | Count PRReviewComments authored by u in W | `S,R`: direct code-comment recurrence | 0 |
| V09 | `review_reply_fraction_30` | # review comments with visible `in_reply_to_id` / all visible review comments by u | `S,H`: thread reply propensity | 1 |
| V10 | `review_distinct_files_30` | Distinct `(repo,pr,path)` in u's review comments | `S,H`: review breadth across code locations | 2 |
| V11 | `review_file_extension_distribution` | Selected extension counts/proportions (`py`, `js`, `ts`, `md`, `other`, unknown) from prior comments | `S,H`: specialist reviewing; sparse feature caution | 2 |
| V12 | `review_comments_per_review_30` | # line comments by u / `(1+visible reviews by u)` in W; not necessarily one-to-one | `S,H`: intensity of reviewing | 0 |
| V13 | `pr_review_thread_depth` | Max/mean reply chain depth from **visible** `in_reply_to_id` ancestors with both endpoints visible | `S,D,H`: extended code discussions | 2 |
| V14 | `same_pr_repeat_participation` | # PRs for which u had ≥2 distinct visible review/comment events / # involved PRs | `S,H`: ongoing PR involvement | 1 |
| V15 | `neighbor_authored_pr_recent` | # previously visible PRs attributed to known collaborators **where PR author field availability at t is proven**, on recent linked repos | `S,R,H`: candidate review opportunities, high attribute-timing risk | 2 |

### G11 — Multi-hop interaction graph and organizational context (`F-GRAPH-2HOP`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| G01 | `user_repo_neighbor_count_30` | # distinct *other cohort users* connected to u through ≥1 recent visible repo, dedup users across repos | `S,R,D`: 2-hop proximity | 1 |
| G02 | `shared_repo_weighted_degree` | `Σ_{v≠u} #shared_recent_repos(u,v)` for cohort users | `S,D,H`: repeated local connectivity | 1 |
| G03 | `neighbor_distinct_repo_count` | # distinct repos in observed recent activity of u's cohort neighbors; exclude u's own edges | `S,D,H`: neighbor project breadth | 2 |
| G04 | `jaccard_repo_similarity_mean` | Mean or top-k `card(R_u(30)∩R_v(30))/card(R_u(30)∪R_v(30))` over cohort neighbors | `S,D,H`: similarity of project involvement | 2 |
| G05 | `weighted_neighbor_recent_activity` | `Σ_v similarity(u,v) * n_{v,a}(7)/Σ similarity` with v≠u, graph snapshot before t | `S,R,D`: target-specific peer activity | 1 |
| G06 | `neighbor_activity_acceleration` | Same weighted average of `(rate_v,last7 − rate_v,prior21)` | `S,H`: nearby behavior changing now | 2 |
| G07 | `two_hop_review_author_path_count` | # visible u→reviewed PR→author→other PR paths at t, dedup by stable entities | `S,D,H`: collaboration recurrence | 2 |
| G08 | `issue_coparticipant_count_30` | # distinct other cohort users who commented/reviewed same visible issue/PR as u | `S,R,D`: collaboration network | 1 |
| G09 | `repeat_coparticipant_count_90` | # co-participant pairs with ≥2 separate visible shared PR/issues | `S,D,H`: repeated collaboration | 2 |
| G10 | `collaborator_recency_mean` | Mean `t−last_shared_visible_event` across prior collaboration partners | `S,H`: freshness of user network | 2 |
| G11 | `repo_org_breadth` | # distinct known organizations across visible connected repositories; unknown separately | `S,R,H`: cross-organization breadth | 1 |
| G12 | `org_neighbor_activity` | Observed cohort activity on repos sharing a **known** `org_id` with u's active repos, excluding u | `S,D,H`: broader context than same repo | 2 |

**Do not treat graph size or centrality computed on the full current database as historical.** Build *snapshots at t* or maintain streaming counts with exact edge emergence times. For high-degree repos, cap or subsample neighbor joins reproducibly; report resulting approximation error and compare against an exact small-snapshot oracle.

### G12 — Global/cohort regime and population context (`F-REGIME`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| C01 | `cohort_active_fraction_{a,7}` | # cohort users with visible a activity last7d / # cohort users visible by t | `S,R,H`: target-wide regime prevalence | 0 |
| C02 | `cohort_event_rate_{a,7v30}` | `(cohort events_a last7/7)/(cohort events_a prior23/23+ε)` | `S,R,DRIFT`: global activity shift | 0 |
| C03 | `relative_user_activity_rate_{a}` | `rate(u,a,7)/(cohort mean user rate_a,7+ε)` | `S,R,H`: activity normalized to contemporaneous regime | 1 |
| C04 | `within_cohort_activity_percentile_{a}` | Percentile of user's count in **cutoff-t cohort count distribution** | `S,H`: relative engagement position | 1 |
| C05 | `cohort_activity_mix_entropy` | Entropy of six cohort event counts last7d | `S,T,H`: which activity types dominate current regime | 2 |
| C06 | `eligible_proxy_fraction` | # historically proxy-eligible cohort users / 9,000, according to *predefined past-90d rule* | `S,R,H`: population shift; proxy not future test-list truth | 1 |
| C07 | `user_eligible_proxy_flag` | 1 if u meets historical proxy rule at t; report both all cohort/proxy | `S,R`: matches sampled population condition | 0 |
| C08 | `calendar_week_index` | Integer weeks from fixed epoch; never use as unique proxy ID | `S,DRIFT,H`: captures secular shift; extrapolation may fail | 2 |
| C09 | `cohort_new_repo_fraction` | Fraction of currently used cohort repos first observed within last30d | `S,H`: novelty of development environment | 2 |
| C10 | `days_until_weekend_from_cutoff` | Deterministic calendar transform of t (not external GitHub data) | `S,H`: relevant if historical/final cutoffs vary by weekday | 2 |

**Historical training recency is a separate intervention, NOT itself a feature:** compare full-history, trailing 8/12/16 weeks, and exponential training-example weights with 4/8/12-week half-lives (EXP-009). Do not combine with new features in the same causal-style ablation; train-window decisions can materially change all feature outcomes. [RPT, DRIFT]

### G13 — Light metadata, text, and automatically synthesized features (`F-META-AUTO`)

| ID | Feature | Formula / generation | Source and rationale | Tier |
|---|---|---|---|---|
| M01 | `issue_title_length` | Mean/median character length of *verified historically visible* issue titles linked to u | `S,H`: descriptive complexity; truncated/missing fields | 2 |
| M02 | `issue_body_length` | Mean/median char length of visible truncated Issue bodies associated with u; missing flag | `S,H`: communication context; not raw unseen full body | 2 |
| M03 | `comment_length_stats` | Mean/std/max length of u's prior visible IssueComments and PRReviewComments bodies separately | `S,H`: participation mode | 2 |
| M04 | `comment_has_question_fraction` | Fraction of u's prior visible comment bodies containing `?`; no external tokenizer required | `S,H`: conversational form, unproven signal | 2 |
| M05 | `issue_label_token_counts` | Count fixed, **training-only selected** common label tokens among visible issues u interacts with | `S,H`: workflow categorization; rare/unstable vocab | 2 |
| M06 | `review_path_extension_share` | Fractions of u's review comments by `file_extension`, top training-only extensions plus `other` | `S,H`: specialization; no extra file access | 2 |
| M07 | `issue_pr_keyword_tfidf` | Learn TF-IDF vocabulary **on earlier training text only**, aggregate visible u-associated row vectors with mean/max | `S,H`: task-specific text without pretrained weights; costly | 2 |
| M08 | `auto_agg_count_nunique` | Whitelisted cutoff-safe `COUNT`, `NUM_UNIQUE` across `User→repo/PR/issue/review` relationships | `S,D,DFS`: systematic join synthesis | 2 |
| M09 | `auto_agg_second_order` | `MEAN/MAX/STD` of first-hop repository/user counts for whitelisted 2-hop paths | `S,D,DFS`: second-order relational statistics | 2 |
| M10 | `time_series_abs_sum_changes` | `Σ_{j=0}^{K−2} abs(w_{a,j}−w_{a,j+1})` (K=8) | `S,T`: overall weekly volatility | 2 |
| M11 | `time_series_short_autocorr` | Lag-1 correlation of 8 prior weekly counts when nonconstant | `S,T`: regular recurrence | 2 |
| M12 | `time_series_binned_entropy` | Entropy of predeclared bins of past 28 daily event counts | `S,T`: variation type; sparse pitfalls | 2 |

**Pretrained text models:** The homework does not explicitly authorize pretrained weights that encode outside information. Do not include downloaded pretrained GitHub/user embeddings, external corpora, or externally trained social-profile representations without TA approval. Lightweight TF-IDF trained solely on instructor text is the safer candidate. Raw issue bodies/diff hunks are truncated and may not justify development cost.

### 3.1 Catalog scale, redundancy, and mandatory coverage flags

The **173 catalogue rows are feature templates**, not 173 fixed engineered columns, and not an instruction to instantiate their complete Cartesian products. `T01` alone could produce 36 columns (six outcomes × six windows). Before generating any feature, assign a permanent manifest entry containing `feature_id`, `name`, `group`, exact formula, raw columns, source tables, visibility predicate, windows, missing-value policy, computation cost, provenance, version, and experiment ID.

For sparse/derived features, generate companion coverage/missingness where relevant:

- `has_{a}_history`, `n_{a}_observed_events`, `n_{a}_observed_weeks`, `inter_event_gap_support`.
- `n_visible_pr_author`, `n_visible_assignee`, `n_known_state`, `n_known_repo_owner`, `n_observed_neighbors`, `n_eligible_transition_opportunities`.
- A **zero count** means no eligible observed rows; **NaN** should mean undefined statistic, unavailable relationship, or insufficient support. Never blanket `fillna(0)` across all feature groups without documenting the semantic distinction.

**Explicit non-candidates:** commit count per push, line additions/deletions, full GitHub repo stars/forks, followers, private activity, CI outcomes, PR requested reviewers, full live issue status history, true repository default branch, and authentic per-user timezone are **not supplied** as reliable fields in this schema. Do not invent them or fetch them externally.

---
## 4. Feature groups and staged implementation

The goal is **not** to instantiate every feature template and train on all resulting columns immediately. Test coherent groups answering specific hypotheses; only expand groups whose out-of-time incremental value is supported. Feature selection is intended to improve predictive quality, cost and interpretability, and feature subsets may be unstable across samples. [SELECT, STABLE]

### 4.1 Feature-group register

| Group | Catalogue IDs | Hypothesis | Recommended initial subset | Priority | Dependencies |
|---|---|---|---|---|---|
| `F0` | Actual **existing 59 columns**, to inventory | Current measured baseline | Exactly current code/config | P0 | Verified scorer |
| `F-TEMP-BASE` | T01–T12 | Recent counts, active days and binary activity provide persistence signal | T01 (7/30d), T02 (7d), T03, T08, T12 | P0 | Visible event stream |
| `F-TEMP-RHYTHM` | T13–T28 | Inactivity relative to personal cadence adds beyond population-level recency | T13, T15, T19, T22, T27 | P0 | History support |
| `F-TEMP-STATE` | T29–T45 | Week-to-week state transitions matter more than cumulative history | T29–T34, T39–T41 | P0 | Mature historical weekly panel |
| `F-TEMP-MOMENTUM` | T46–T57 | Recent acceleration and decay help with regime change | T46–T49 and T52 | P0/P1 | Tested short-horizon representation |
| `F-CROSS` | X01–X14 | The six activities have complementary states and workflows | X01 (chosen pairs), X02–X05, X10, X13 | P1 | Winning temporal control |
| `F-REPO-SELF` | R01–R14 | Repo breadth and continuity matter beyond own volume | R01–R06, R12 | P1 | Correct FK joins |
| `F-REPO-NEIGHBOR` | R15–R26 | Activity of other cohort members on relevant repos adds information | R15–R17, R19–R21, R25 | P1 | Repo-self, cohort restriction |
| `F-PR-WORKFLOW` | P01–P15 | Push/branch/PR state informs future push and PR behavior | P01–P03, P05–P07, P12 | P1 | PR visibility audit |
| `F-ISSUE-WORKFLOW` | I01–I12 | Observable issue/discussion workload informs comment and issue targets | I01, I04–I07, I10 | P1 | Issue visibility audit |
| `F-REVIEW-COLLAB` | V01–V15 | Repeat reviewer and author relationships matter | V01–V05, V08, V12, V14 | P1/P2 | Review→PR→User links |
| `F-REGIME` | C01–C10 | Current cohort activity level helps account for drift | C01–C04, C07 | P1 | Historical eligible population |
| `F-GRAPH-2HOP` | G01–G12 | Multi-hop collaboration contributes beyond direct repo counts | G01, G02, G05, G08 | P2 | Proven neighbor lift |
| `F-META-AUTO` | M01–M12 | Visible text and synthesized 2-hop aggregations add residual signal | M01/M03/M06 *or* M08–M09 | P2 | Evidence plus budget |

**Inventory first:** Some rows may already exist among F0's **reported** 59 columns. Mark each candidate `new`, `exact duplicate`, `near duplicate`, or `replacement`; no “new feature” credit for recomputing a baseline under another name. `T01` represents multiple optional columns, not one guaranteed column.

### 4.2 Experimental dependency order

1. **EXP-007/008 — Integrity:** exact evaluator parity, per-target EXP-006 forensics, historical point-in-time correctness, realistic rolling-origin test and label maturity. No feature search while these are unresolved. [RPT, EVAL]
2. **EXP-009 — Training recency:** freeze F0 and the learner; test full history against trailing 8/12/16-week windows and 4/8/12-week exponential example weighting. This isolates distribution-shift handling before representation changes. [RPT, DRIFT]
3. **EXP-010 extended — Temporal groups:** using one fixed learner and training policy, separately test baseline + `F-TEMP-BASE`, `RHYTHM`, `STATE`, `MOMENTUM`. Promote only coherent groups; then evaluate their union.
4. **Workflow group — `F-CROSS`:** test selected cross-target pairs; do not generate all 15 possible ordered pairs across multiple lags indiscriminately.
5. **EXP-012 extended — Relational:** add repo-self, then repo-neighbor; next test PR, issue and review feature groups. Keep own-activity volume features fixed to reveal genuine relational increment. [RPT, RDL]
6. **High-cost gate — Graph/text/DFS:** investigate only if simple relational features have positive and stable incremental value or a distinct well-grounded hypothesis survives. Do not assume more feature depth implies more score.
7. **Model and threshold gate — EXP-011/013:** with feature groups provisionally frozen, compare strong boosting and a small target-specific persistence/learner ensemble; use chronological OOF predictions for threshold tuning. [RPT, F1]
8. **EXP-014 conditional:** compare a simple 2-layer relation-aware GNN with the best handcrafted relational model **only if** relational groups pass their earlier gate. Otherwise keep it as background, not the deployed predictor. [RPT]

### 4.3 Controlled complexity budget

- Initial screen: **one learner**, e.g. HGB, same fixed hyperparameters and seed, same calendar folds and label policy, approximately 10–25 *nonduplicate* new columns per group. These limits are project heuristics, not published optima.
- Refinement: try a **small preset** window/decay subset for the two to four strongest feature families; avoid a Cartesian search of every transform and window.
- Combined-group experiment: retrain on the retained union, then remove each family in turn to test necessity.
- Model-family experiment: keep selected features fixed while comparing LightGBM/CatBoost/XGBoost, not while adding hundreds of new features simultaneously.
- Final prediction policy: target-specific models and joint thresholds are also hyperparameters requiring their own strict historical selection.

---

## 5. Credible evaluation methodology

### 5.1 Research basis and problem structure

The homework explicitly requires a chronological train/validation split. Here, the same users recur weekly and activity rates change with calendar regime; random row CV mixes related observations and can yield optimistic errors. Cerqueira, Torgo and Mozetič report that nonstationary forecasting particularly benefits from ordered, repeated out-of-sample evaluation; Roberts et al. show that dependence requires structure-aware validation. [A2, TEMP-CV, CV-STRUCT]

**Score unit:** `(github_user_id, historical_cutoff)` corresponds to six decisions. Several reporting weeks together are user-*weeks*, not independent people. Evaluate variation between calendar reporting weeks; do not treat thousands of user-week rows as independent statistical replications.

### 5.2 Two-level forward-chaining evaluation and data maturity

Illustrative outer reporting fold `R` using the supplied report's structure:

```text
                  earlier mature samples   C1        C2        C3          gap          R
calendar time  --------------------------|---------|---------|----------|-----------|---->
                                                R-28d     R-21d     R-14d      R-7d     report
                  model training        calibration predictions/thresholds     embargo   outcome
                  Each reporting/calibration marker represents target-start s_i; features stop at c_i≈s_i−4h.
```

**Operational invariants:**

- Every outer evaluation example has *two* times: `c_i` for observable feature extraction and `s_i` for the official-style prediction-label start. Require `c_i < s_i`, preferably the same ~4-hour gap as the final release. Weekly lag features and source observation sequences end at `c_i`; supervised outcomes start at `s_i`. For a final decision at local midnight, do **not** accidentally add unseen 20:00–00:00 activity to features.
- Feature row `(u,c_i)` uses only rows visible strictly before `c_i`.
- Model training at simulated `T_fit` uses only **training labels mature/observable by `T_fit`**.
- Calibration probabilities come from models that did **not** train on their own target weeks **or use calibration weeks whose labels had not matured by their simulated prediction/selection time**; threshold choice must precede outer reporting labels. When simulating the deployment chronology, do not assume labels for `C1/C2/C3` were all knowable at an earlier model-fit date. Use time-consistent rolling-OOF training/threshold selection or explicitly declare a historical fixed-fit diagnostic that is *not* an executable prospective policy.
- Feature-group/model/hyperparameter selection uses earlier **inner** rolling folds, while outer reporting blocks estimate future performance. This selection/evaluation separation is essential: Varma & Simon and Cawley & Talbot document optimization bias when the same validation set serves both functions. [NESTED, SELECT-BIAS]
- Keep a ledger marking all previously inspected dates as **development**. The August 31, September 7 and September 14 blocks in EXP-006 are already inspected; do not call them a fresh sealed holdout. [RPT]

For a historical **target-start** time `s_i` (distinct from the earlier feature cutoff `c_i`), horizon `H=7d`, and target-specific observational maturation buffer `m_a`, an initially conservative fit-inclusion predicate is:

```text
  s_i + H + m_a <= T_fit
  AND source data proves the historical label was knowable by T_fit.
```

The report recommends auditing `m_a ∈ {0,1,3,5,7,14}` days. Choose the smallest **stable-observation** buffer where late-discovered positives/identities and model order stop changing materially, rather than choosing `m_a` to maximize the score. **This stability test requires actual dated snapshots or reliable release/as-of evidence.** Since the official repository force-pushes updates and does not retain old commits, a current snapshot plus `first_seen_at` alone may not prove the *original release in which each retrospectively filled field appeared*. Without dated copies, report these as **sensitivity analyses of reconstructed historical truth**, not validated retrospective label-availability facts. `open_issue` is particularly sensitive to late discovery. This inequality is an approximation, **not** a substitute for exact historical label as-of snapshots when available. [RPT, DATA]

**Final-cutoff alignment:** The last instructor data release covers activities only until approximately 20:00 Taiwan time on November 11, while the target begins at midnight November 12. Specify historical *paired feature and target start timestamps* rather than a single ambiguous `cutoff`, and reproduce the comparable four-hour feature-to-target blind interval in backtests or explicitly measure a sensitivity variant. Do **not** use external GitHub activity to fill it. [MAIN, EVAL]

**Minimum reporting coverage:** Aim for at least six usable forward reporting blocks *if mature data permit*, spanning different activity regimes. This is a planning target, not an independently verified sample-size rule. With few folds, make weaker uncertainty claims.

### 5.3 Official combined F1 and required secondary diagnostics

For each target `a`, compute confusion counts `TP_a, FP_a, FN_a` and:

```text
F1_a = 1.0                            if TP_a+FP_a+FN_a = 0
     = 2 TP_a/(2 TP_a+FP_a+FN_a)     otherwise
macro = mean(F1_a across six targets)
micro = official_F1(sum_a TP_a, sum_a FP_a, sum_a FN_a)
combined = (macro + micro)/2
```

**When pooling weeks**, concatenate their user-week predictions and sum counts **before** applying the F1 formula; averaging weekly F1 is not equivalent to pooled F1. Also show individual week metrics to reveal instability and the official `F1=1` behavior for all-negative week/target combinations. [EVAL, RPT]

Report for **each** model and fold: the official `combined`, macro, micro; six target F1s and TP/FP/FN; prevalence, predicted-positive rate, precision and recall per target; ranking diagnostics such as PR-AUC where defined; model/feature build cost; feature support/missingness. Precision-recall diagnostics are valuable for rare target events, where ROC-AUC can give a misleading view of positive prediction quality. [PR]

Report both **the 9,000-user cohort** and the prespecified **historical eligibility proxy** (at least five source activity records in the preceding 90 days for the historical cutoff, evaluated from allowable observed records); choose a primary development population beforehand rather than selecting whichever scores highest. As official test membership is fixed from the October 28 release, a future-evaluation proxy at final time need not coincide with the eventual test set. Only official published test IDs are used for final inference, not as imagined future labels. [RPT, DATA, EVAL]

### 5.4 Additive versus removal ablation (both required)

For a feature family `G` and fixed baseline feature set `B`:

- **Additive:** compare `B` with `B∪G` on **identical** reporting cutoffs, source snapshot, population, learner, hyperparameter budget, training recency, seed, label policy, and threshold *selection procedure*. This tests incremental value of the candidate group.
- **Removal:** after constructing a candidate union `U`, retrain and compare `U` with `U\G`. This checks whether a previously winning group remains useful **after other groups are included**.
- **Target-specific:** If a group only helps `open_issue` or `pr_review`, consider a separate target-specific feature subset instead of applying it to all six targets; this also requires chronological selection.
- **Interactions:** Only after main groups survive, test a few preselected pairs such as `(RHYTHM, STATE)` or `(REPO-SELF, REPO-NEIGHBOR)`; correlated groups may each look weak alone but complement each other.

Changing feature groups and learner, sample weights or thresholds **at the same time** invalidates claims that the measured difference came from the features. Keep controlled variables clear in the run manifest.

### 5.5 Paired per-fold gains and proposed promotion gates

Let `S_i(M)` be the official combined F1 of model/pipeline `M` on reporting week `i`. Compute:

`delta_i(G) = S_i(B+G) − S_i(B)`.

| Result to record | Calculation and use |
|---|---|
| `median_delta` | Median paired delta by reporting week; main typical-effect indicator |
| `mean_delta` | Mean paired delta, with week-count caveat |
| `win_fraction` | Fraction of usable weeks with strictly positive delta (report ties separately) |
| `nonnegative_fraction` | Fraction of weeks with delta ≥ 0 |
| `worst_delta`, `best_delta` | Detect brittle changes and jackpot weeks |
| `recent_period_delta` | Display latest 2–3 weeks and drift regimes separately |
| `delta_by_target` | Target-specific F1 and FP/FN trade-offs; count positive support |
| `all_vs_proxy` | Repeat with all-cohort and proxy-eligible populations |
| `time_and_memory` | Peak RAM, extraction time, fit and inference time, columns, missingness |

The supplied report's suggested heuristics (not published universal thresholds) are: for recency policy median `delta ≥ +0.010`, ≥70% of recent usable folds nonnegative and no catastrophic collapse; for relational group screening median `delta ≥ +0.005` with stable direction and at least one coherent per-target gain. Smaller but repeatedly reproduced low-cost gains may still be valuable; one unusually good week is insufficient. [RPT]

**Decision categories:** `PROMOTE` (stable out-of-time benefit and safe cost), `REFINE` (particular targets/lags help), `PARK` (too few independent weeks, ambiguity or high expense), `REJECT` (repeated nonpositive increment, redundancy, irreducible leakage). Store the **reason and exact run IDs**, not just a yes/no.

### 5.6 Feature importance and redundancy: not substitutes for ablation

Use feature attribution **after** measuring valid forward performance, not as an alternative to it:

1. **Retrained group ablation** is primary: measure score changing when a group is included or excluded.
2. **Held-out grouped permutation** is secondary: shuffle a coherent feature family jointly and measure official combined score drop in a *trained fixed model*. The scikit-learn documentation warns that individual permutation importance can be misleading when variables are correlated. Group permutation is itself a perturbation diagnostic, not a causal explanation. [PFI]
3. **Within-group filtering:** remove exact duplicates, constant columns, high missingness (when not informative), and only then cluster highly correlated columns using **training-only** rank correlations; evaluate representative subsets with forward ablations. Fit any filtering thresholds, encoders, imputation values, text vocabulary and correlation clusters *separately inside each historical training fold*. [SELECT, STABLE]
4. **Stability across periods:** Compare which families remain useful across different forward folds. Nogueira et al. show that selection stability is itself important, distinct from one-fold feature ranking. [STABLE]

Never choose features solely from impurity importance, SHAP rank, a single permutation chart, supervised mutual information, or default `RFECV`. `RFECV` with default classifiers may use non-temporal folds and must be supplied an appropriate custom chronological splitter; any supervised selection must stay inside the inner train/selection period. [SELECT, NESTED]

### 5.7 Threshold and model-selection effects

An observed F1 gain might be a **better threshold** rather than a better ranking function. Two diagnostics should be reported:

- **Controlled-threshold diagnostic:** reuse the baseline's fixed six-threshold vector to compare the effect on predicted labels. This isolates a decision-rule slice but may unfairly penalize shifted probability scales, so it is *diagnostic only*.
- **Realistic deployed-pipeline comparison:** for every candidate, run the **same prespecified inner-OOF six-dimensional joint threshold tuning** procedure, never using reporting labels. This is the criterion for model promotion.

The F1 threshold literature shows why 0.5 is not generally correct, and a joint macro/micro objective couples the six targets; six separately optimized per-target thresholds need not optimize the assignment score. [F1, RPT]

To tune thresholds legitimately, train on earlier mature history, predict the next mature calibration week out of time, concatenate several OOF calibration weeks, jointly optimize threshold coordinates, then evaluate on the next untouched reporting block. Check threshold stability across folds and probability-scale transfer after final refit; if refit breaks transfer, preserve a calibration separation. Select per-target persistence vs learner and optional ensembles only through earlier rolling folds.

### 5.8 Uncertainty, dependent data and adaptive feature search

- **Calendar weeks are primary forward blocks.** Adjacent report weeks and reused fitting histories are dependent, so ordinary IID tests treating folds or user-week rows as independent are misleading. [CV-STRUCT, TEMP-CV]
- **Repeated users:** Cluster-bootstrap stable `github_user_id` within a fixed report period (retaining all of each sampled user's rows) to assess conditional user-sampling sensitivity. Do **not** interpret this as a confidence interval over future calendar regimes. [RPT]
- **Very few weeks:** Avoid precise-looking “95% confidence” claims from three reporting weeks. Once enough nonoverlapping weeks exist, report week-block resampling as sensitivity with its assumptions; always show raw paired fold deltas.
- **Multiple tests:** Searching hundreds of features on the same weeks optimizes noise. Predeclare a small menu, freeze outer test periods, track every tested variant, and avoid cherry-picking. Formal multiple-test corrections such as Benjamini–Hochberg require valid p-value procedures and still do not repair adaptive reuse of a holdout. [FDR, SELECT-BIAS]
- **Scientific language:** Write “development-fold improvement” unless the positive direction was confirmed on a new untouched forward block.

---

## 6. Exact ablation and feature-selection plan

### 6.1 Minimum experiment matrix

| Run | Control | One intervention | Main question |
|---|---|---|---|
| `FE-00` | Instructor scorer | Current local scorer/panel | Scorer and temporal reconstruction identical? |
| `FE-01` | F0 + full history | F0 + recency window/weight | Did old training hurt? |
| `FE-02` | F0 + chosen train policy | + `F-TEMP-BASE` | Is persistence directly represented? |
| `FE-03` | Temporal foundation | + `F-TEMP-RHYTHM` | Do individualized gaps help? |
| `FE-04` | Temporal foundation | + `F-TEMP-STATE` | Do week-state transitions help? |
| `FE-05` | Temporal foundation | + `F-TEMP-MOMENTUM` | Does trend/decay add beyond counts? |
| `FE-06` | Winning temporal model | + `F-CROSS` | Do cross-target transitions matter? |
| `FE-07` | Winning temporal model | + `F-REPO-SELF` | Does user repo structure add signal? |
| `FE-08` | Winning temporal + repo-self | + `F-REPO-NEIGHBOR` | Does other cohort activity add signal? |
| `FE-09` | Accepted foundation | + `F-PR-WORKFLOW` | Do branch/PR states add lift? |
| `FE-10` | Accepted foundation | + `F-ISSUE-WORKFLOW` | Do issue discussions and workload help? |
| `FE-11` | Accepted foundation | + `F-REVIEW-COLLAB` | Do reviewer–author/review modes help? |
| `FE-12` | Accepted foundation | + `F-REGIME` | Can cohort prevalence improve drift transfer? |
| `FE-13` | Union of promoted families | Union minus each family | Which group is still necessary? |
| `FE-14` | Best group union + HGB | Same union + stronger tree models | Is algorithm choice now the bottleneck? |
| `FE-15` | Best single predictor | Target-specific selected models + joint thresholds | Are methods complementary? |
| `FE-16` | Best handcrafted relational model | Simple relation-aware GNN | Only if incremental relational benefit established |

Within the `FE-02`–`FE-12` comparisons **freeze** data snapshot and cutoffs, the training and maturity policy, scorer, population, model settings, random seed and threshold selection procedure. If any must change, treat the result as a *new pipeline comparison*, not isolated feature uplift.

### 6.2 Negative controls and falsification checks

In addition to forward ablations, include inexpensive checks that detect plausible leakage and overfitting rather than treating a higher score as automatically trustworthy:

- **Chronology canary:** add one synthetic record that is created in the past but first seen in the future. Any historical feature change is a test failure. Repeat for late-populated PR/issue/review attributes and author metadata, not just for new event rows.
- **Entity ID canary:** rewrite internal row IDs and corresponding FKs while preserving stable GitHub IDs; feature values must remain the same.
- **Neighbor self-exclusion:** compare computed repo-neighbor activity with and without the focal user's events on a small exact dataset; `neighbor_*` must exclude focal-user contributions and repeated shared-repo users must not be unintentionally double counted.
- **Placebo/irrelevant-feature control:** add a fixed seeded noise column or a time-respecting randomized synthetic feature. A claimed consistent large gain on the placebo is a reason to investigate experimental selection or leakage; it is not a formal p-value test by itself.
- **Out-of-regime challenge:** record performance of accepted groups separately during the dip and recovery regimes; a gain only on old inspected weeks does not establish future generalization.
- **Source-event truth check:** manually trace sampled push, PR opening, issue opening, review submission, issue comment and review comment events through source semantics and visibility filters. If a ground-truth historical occurrence cannot be reconstructed, document that target's label bias rather than silently treating missing as negative.

### 6.3 Structured experiment record

Persist one result row per `run_id × reporting_week × population` with at least:

```text
run_id, parent_run_id, data_release_sha, snapshot_max_visible_time,
feature_manifest_hash, feature_group_ids, exact_feature_names, n_features,
feature_cutoff_utc, report_cutoff_utc, label_maturity_policy,
training_window_or_half_life, population_policy,
model_family_and_hyperparameters, seed,
threshold_calibration_cutoffs, six_threshold_values,
macro_f1, micro_f1, combined_f1,
per_target_tp_fp_fn, per_target_f1, per_target_prevalence,
per_target_predicted_positive_rate, per_target_pr_auc_if_defined,
feature_extraction_seconds, fit_seconds, inference_seconds, peak_ram_mb,
missingness_by_group, leakage_test_artifacts, notes
```

Save **per-user/per-week OOF predicted probabilities** and their reporting predictions with stable GitHub IDs, cutoffs and run IDs, so threshold forensic checks need not retrain. Preserve every attempted variant including failures; selection bias is not visible if only winning runs survive in logs.

### 6.4 A disciplined screening cycle

1. Inventory F0, map all catalogue entries against existing features, and write a group manifest.
2. Validate a candidate group's raw columns, joins and strict cutoff logic on a small sample; unit-test later-known fields and future-record invariance.
3. Generate *only that family* for the same historical cutoffs as F0; report null/zero rate, coverage, correlation/duplication, counts and build cost.
4. Train control and additive candidate under identical replay; use **inner calibration only** for each six-target threshold vector.
5. Report pooled official combined F1, weekly paired deltas, target confusion counts and missingness.
6. If promising, subdivide the group: example `F-TEMP-RHYTHM` with `[recency]`, `[median gap]`, `[normalized inactivity]`, `[daily active fraction]`. Do not immediately multiply all parameter combinations.
7. Promote only stable subfamilies; construct the union and perform **leave-group-out retraining**.
8. Freeze the feature union for strong-learner comparison and target-specific models; confirm on the next genuinely uninspected forward block when available.

### 6.5 Example decision sheet (numbers are illustrative, not measured)

| Candidate | Median paired Δ combined | Improved weeks | Interpretation |
|---|---:|---:|---|
| Personal rhythm | +0.012 | 6/7 | Worth a leave-group-out test and later untouched holdout |
| All cross-activity pairs | −0.003 | 2/7 | Reject blanket pairs; test only a few justified transitions |
| Repo-neighbor recent activity | +0.006 | 5/7 | Investigate volume redundancy and target-level coherence |
| Expensive 2-hop graph statistics | +0.020 | 1/7 | Unstable despite impressive single-week win |
| PR title metadata | +0.001 | 4/7 | Ambiguous; probably park until higher-value groups exhausted |

Each family gets status `PROMOTE`, `REFINE`, `PARK`, or `REJECT`, along with exact code/data hashes, date split, per-target evidence, and rationale. Large but inconsistent gains do not automatically beat modest, consistently positive gains. The objective is expected November performance, not an adaptively maximized historical leaderboard.

### 6.6 Why “all features at once” is a poor first run

A one-shot union changes dozens of mechanisms simultaneously: recurrence encoding, changing training-window behavior, missing-data patterns, project interactions, and high-degree graph joins. If the score moves, attribution is unclear, and an innocuous feature may hide leakage. It is perfectly possible to try a large union **later as one candidate model** after all sources/fields pass the cutoff audit, but that run requires the same nested selection rules as any other hyperparameter search. [SELECT, NESTED]

---

## 7. Audit, reproducibility and release checklist

### 7.1 Feature-level tests — mandatory

- **Strict cutoff:** A row at exactly `v(e)=t` must contribute zero to every feature. A row at `t−W` must be included for `[t−W,t)` windows.
- **Future append invariance:** Appending events with `v(e)>t` changes *none* of the already-built `(u,t)` feature vectors, even through 2-hop repository/cohort joins; also test future-filled/mutated metadata on a previously visible PR/issue/review record.
- **Historical availability:** A PR/Issue/Review with `created_at`/`submitted_at <t` but `first_seen_at>t` is unavailable in features. Late-known `closed_at`, `merged_at`, `dismissed_at`, `merged_by_id` do not leak through state aggregation.
- **Mutable issue state:** Do not use `Issues.state_reason` historically without a defensible timestamp for that exact version.
- **Type-safe repo IDs:** Compare `PullRequests.head_repo_id` against joined `Repositories.github_repo_id`, **never** internal row index `Repositories.repo_id`.
- **Stable joins:** After internal FK row renumbering and rebuild, stable-ID keyed user features do not change; no raw internal FK crosses release boundaries.
- **No join amplification:** One event joined through several relationships contributes exactly once to a requested distinct-event statistic.
- **No fictitious noncohort completeness:** Neighbor feature explicitly uses observed cohort activity and excludes the subject's own events where specified.
- **Correct target distinctions:** Ordinary issue vs PR conversation; issue comment vs line-level PR review comment vs PR review submitted; observed `first_seen` vs real opening timestamps. Explicitly trace reconstructed **training labels** separately from historical observation-proxy features.
- **Sparse support:** Undefined ratios/gaps are missing with coverage flag, not silently zero; all-zero users are handled deterministically.
- **Supervised statistics:** Historical conditional-label features and any TF-IDF vocabulary/categorical encoding are fit only on chronological training samples available as of the simulated fit time.
- **Score parity:** Check local metric against exact official `evaluation/evaluate.py` for random binary matrices and corner cases including entirely all-negative targets where official F1=1.
- **Snapshot deterministic:** Same pinned release, cutoff policy, feature manifest and seed reproduce identical row counts and values.

### 7.2 QA and compute budgets

For each group, record row counts before/after each join, duplicate event counts, zero and NaN proportions, min/median/p95/p99/max, groups' training-only correlations with F0, proportion of cohort users covered, time/memory required, and at least five manually traced `(user,t)` examples. For graph groups audit repo-degree tails and any neighbor capping/sampling approximation. For text groups audit body truncation, missing text and vocabulary fit date.

Prefer cached cutoff-safe intermediate `(user, repo, week, activity)` aggregates and columnar Parquet over recomputing millions of joins per fold. Cache keys must contain source release SHA, feature implementation version, cutoffs, and label/as-of policy. **Never cache a full-dataset graph with future edges then merely truncate aggregate outputs later.**

### 7.3 November inference and homework deliverables

- Keep your own implementation/results outside the replaceable `assignment2/` clone. The instructor replaces its repository daily via force-push; save the final snapshot SHA and data coverage timestamp. [MAIN]
- Rebuild only features available by the final supplied data cutoff (approximately Nov 11, 20:00 Taiwan); do not add outside API records to fill the four-hour gap.
- Use the official `evaluation/test_input.csv` users when released, matching stable `github_user_id` identifiers; apply **frozen** model/feature/threshold policies.
- Emit exact `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push`, binary outcomes, correct full user set and no duplicate IDs. Validate with `python assignment2/evaluation/evaluate.py --pred output.csv --input assignment2/evaluation/test_input.csv`. [EVAL]
- **Deadline: 2026-11-11 23:55 Taiwan time.** Commit reproducible source to a public GitHub repository before the deadline. Submit `output_{studentID}.csv` and the prescribed PDF/video files to E3. The assignment PDF shows inconsistent examples for slide/video filename stems (`Asg2_...` versus `Asg_...`); confirm the final accepted naming and template on E3 rather than guessing. The PDF requires a **non-AI-generated** 8–12-minute Mandarin/English video and **English slides, maximum ten pages**; hallucinated experimental results and plagiarism are penalized. [A2]
- Presentation Q1: draw all eight tables with PK/FK arrows and the specified entity graph for users `22996444` and `28983449` from `2026-09-15 00:00` to `2026-09-17 00:00 UTC`. Q2: show relational-model enhancements (including tested negative findings when relevant). Q3: show temporal enhancements. Q4: show the actual complete pipeline; all described experiments must correspond to measured, reproducible runs. [A2]

---

## 8. Research and authoritative references

**Evidence rule:** `[A2]` plus `[SCHEMA]/[DATA]/[MAIN]/[EVAL]` govern compliance and raw columns. `[RPT]` preserves the supplied report's research assumptions and previously claimed measurements; **it is not an independent replication**. Published sources motivate operators, feature families and evaluation methodology—not claimed GitHub six-target score improvement.

| Key | Reference and stable URL | Relevance |
|---|---|---|
| `A2` | *Big Data Analytics Techniques and Applications — Assignment 2: Predicting GitHub User Activity*, 2026-10-14, provided `Big Data Course - Assignment 2.pdf`, §§2.4–3 | Only instructor data, chronological split, six outputs, grading |
| `SCHEMA` | Instructor [`data/db/SCHEMA.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/db/SCHEMA.md) | Eight-table schema, exact columns and FK structure |
| `DATA` | Instructor [`data/README.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/data/README.md) | Visibility time, cohort incompleteness, mutable history |
| `MAIN` | Instructor [`README.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/README.md) | Data updates, final coverage and stable IDs |
| `EVAL` | Instructor [`evaluation/README.md`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/README.md) and [`evaluation/evaluate.py`](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2/blob/main/evaluation/evaluate.py) | Official binary output and exact combined metric |
| `RPT` | User's supplied *High-Performance, Generalizable GitHub Activity Prediction for NYCU Big Data Assignment 2* (`Pasted text(3).txt`) | EXP-006 scores and EXP-007–014 roadmap; existing 59-feature code unavailable to original report |
| `GAP` | Decan et al. (2020), [*GAP: Forecasting commit activity in git projects*](https://doi.org/10.1016/j.jss.2020.110573), *Journal of Systems and Software* | Survival-based git developer activity, personalized gaps |
| `DRIFT` | Gama et al. (2014), [*A survey on concept drift adaptation*](https://doi.org/10.1145/2523813), *ACM Computing Surveys* | Window/weight adaptation in changing regimes |
| `TEMP-CV` | Cerqueira, Torgo & Mozetič (2020), [*Evaluating time series forecasting models: an empirical study on performance estimation methods*](https://doi.org/10.1007/s10994-020-05910-7), *Machine Learning* | Repeated out-of-sample temporal evaluation for nonstationarity |
| `CV-STRUCT` | Roberts et al. (2017), [*Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*](https://doi.org/10.1111/ecog.02881), *Ecography* | Dependence-aware performance estimation |
| `SELECT` | Guyon & Elisseeff (2003), [*An Introduction to Variable and Feature Selection*](https://www.jmlr.org/papers/v3/guyon03a.html), *JMLR* | Feature construction/selection objectives and validity |
| `NESTED` | Varma & Simon (2006), [*Bias in error estimation when using cross-validation for model selection*](https://doi.org/10.1186/1471-2105-7-91), *BMC Bioinformatics* | Separation of selection and unbiased reporting |
| `SELECT-BIAS` | Cawley & Talbot (2010), [*On Over-Fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*](https://www.jmlr.org/papers/v11/cawley10a.html), *JMLR* | Repeated search can fit validation noise |
| `STABLE` | Nogueira, Sechidis & Brown (2018), [*On the Stability of Feature Selection Algorithms*](https://jmlr.org/papers/v18/17-514.html), *JMLR* | Feature/group selection consistency |
| `PFI` | scikit-learn [*Permutation feature importance*](https://scikit-learn.org/stable/modules/permutation_importance.html) and [correlated-features example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html) | Model reliance diagnostic, redundancy caveat |
| `F1` | Lipton, Elkan & Narayanaswamy (2014), [*Thresholding Classifiers to Maximize F1 Score*](https://arxiv.org/abs/1402.1892) | Why 0.5 and independent thresholds may be suboptimal |
| `PR` | Saito & Rehmsmeier (2015), [*Precision–Recall vs ROC on imbalanced datasets*](https://doi.org/10.1371/journal.pone.0118432), *PLOS ONE* | Rare-target ranking diagnostics |
| `DFS` | Featuretools [*Handling Time*](https://docs.featuretools.com/en/stable/getting_started/handling_time.html) and [*Deep Feature Synthesis*](https://docs.featuretools.com/en/stable/getting_started/afe.html) | Automatic joins/aggregation; set `include_cutoff_time=False` when required |
| `TS` | tsfresh [*Overview on extracted features*](https://tsfresh.readthedocs.io/en/v0.20.3/text/list_of_features.html) | Standard time-series operators such as autocorrelation and entropy |
| `RDL` | Fey et al. (2024), [*Relational Deep Learning*](https://proceedings.mlr.press/v235/fey24a.html), *ICML/PMLR*; Robinson et al. (2024), [*RelBench*](https://arxiv.org/abs/2407.20060) | Typed temporal database graph methods; potential later research stage |
| `FDR` | Benjamini & Hochberg (1995), [*Controlling the False Discovery Rate*](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), *JRSS B* | Multiple-comparison caution; not a substitute for a sealed test set |

### Final principle

**Optimize for reproducible future-week combined F1, not the number of features or the best retrospectively selected historical score.** Begin with exact recency/state and behavioral hypotheses, prove incremental value by **paired forward additive/removal ablations**, then invest in repository/graph complexity and model tuning only where the evidence warrants it.
