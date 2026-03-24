# Technical Appendix

## Overview

This appendix documents the current evidence base behind the Zerve case study after the latest successful full pipeline run. It is anchored in [pipeline_log.txt](c:/Users/Balint/Documents/Pet_projects/zerve/outputs/pipeline_log.txt) and the downstream artifacts through phases `18-20`.

The earlier technical `.docx` in the repo remains historical context only. Where earlier notes and current outputs differ, the current outputs are authoritative.

## Pipeline Inventory

| Step | Script | Purpose | Key outputs |
|---|---|---|---|
| 01 | `01_eda.py` | dataset overview, event mix, funnel, geo, sessions | `outputs/01_*.html` |
| 02 | `02_feature_engineering.py` | build the user feature matrix | `outputs/user_features.parquet` |
| 03 | `03_user_segments.py` | rule-based behavioral segmentation | `outputs/user_features_segmented.parquet` |
| 04 | `04_cohort_analysis.py` | signup and adoption cohort views | `outputs/04_*.html` |
| 05 | `05_lifecycle_analysis.py` | lifecycle and funnel analysis | `outputs/05_*.html` |
| 06 | `06_kpi_and_modeling.py` | KPI matrix and success prediction | `outputs/06_*.html`, `model_rf_*` |
| 07 | `07_signup_hour_survival.py` | signup-time and return-window analysis | `outputs/07_*.html` |
| 08 | `08_india_hypothesis_success_def.py` | geo framing and success-definition argument | `outputs/08_*.html` |
| 09 | `09_fleet_cohort_model.py` | cohort-parallel modeling example | `outputs/09_fleet_results.csv` |
| 10 | `10_user_lifecycle.py` | time-to-builder, activation milestones, path divergence | `outputs/10_*.html` |
| 11 | `11_tool_sequences_session_progression.py` | tool transitions and session progression | `outputs/11_*.html` |
| 12 | `12_credit_error_propensity.py` | credit, error assist, PSM | `outputs/12_*.html` |
| 14 | `14_canvas_complexity.py` | repeat-canvas and complexity analysis | `outputs/canvas_complexity_features.parquet` |
| 15 | `15_churn_prediction.py` | active-user churn proxy model | `outputs/15_*.html`, `model_churn_rf.joblib` |
| 16 | `16_ngram_workflow_analysis.py` | event, session, and tool motifs | `outputs/16_ngram_tables.csv` |
| 17 | `17_geo_location_analysis.py` | country and region analysis | `outputs/17_country_metrics.csv` |
| 18 | `18_intervention_scoring.py` | intervention targeting layer | `outputs/18_*` |
| 19 | `19_quality_of_struggle.py` | productive vs abandonment-prone struggle | `outputs/19_*` |
| 20 | `20_path_branching_model.py` | early path branch-point analysis | `outputs/20_*` |

## Dataset Snapshot

Latest run highlights:

- `409,287` raw events
- `408,919` cleaned events after filtering
- `107` raw columns
- `141` event types
- `4,771` modeled users
- date range `2025-09-01` to `2025-12-08`

Important structural notes:

- `sign_up` is not present for every user
- backend tool events do not share the same session IDs as web events
- country is usable; finer-grain geo is less reliable
- credit intensity is best treated as event incidence rather than trusted exact consumption quantity

## Current Architecture And Repo State

The project is now organized into:

- [analytics](c:/Users/Balint/Documents/Pet_projects/zerve/analytics): shared helper package
- [docs](c:/Users/Balint/Documents/Pet_projects/zerve/docs): narrative, roadmap, reproduction, leakage, and submission docs
- [legacy](c:/Users/Balint/Documents/Pet_projects/zerve/legacy): archived exploratory files
- project root: runnable numbered scripts plus [orchestrator.py](c:/Users/Balint/Documents/Pet_projects/zerve/orchestrator.py)

This matters for reproducibility: the repo now supports both local orchestration and cleaner translation into Zerve blocks.

## Operational Success Definition

The main operational success proxy remains `Agent Builder`.

Observed segment mix in the latest run:

| Segment | Users | Share | Avg days active | Avg agent tools | Avg build calls |
|---|---:|---:|---:|---:|---:|
| Agent Builder | 285 | 6.0% | 4.75 | 156.99 | 39.15 |
| Manual Coder | 161 | 3.4% | 5.59 | 125.20 | 0.02 |
| Viewer | 1,288 | 27.0% | 1.69 | 26.07 | 0.01 |
| Ghost | 3,037 | 63.7% | 1.05 | 0.00 | 0.00 |

Why it remains the right operational target:

- it is behaviorally distinct
- it aligns with real product value creation
- it is stable under threshold sensitivity checks
- it maps to monetization pressure better than weaker usage definitions

## Feature Groups And Modeling

The current feature matrix has `79` engineered features.

Feature families include:

- volume and active-day features
- canvas usage features
- manual execution features
- agent chat and tool usage
- output/fullscreen behavior
- onboarding features
- time-to-first interaction features
- first-session depth
- return-pattern features
- signup timing and referrer features
- newer signal families:
  - iteration quality
  - session structure
  - canvas commitment

### Main predictive results

| Model setup | Population | Best AUC | Notes |
|---|---|---:|---|
| B1 full population | all users | 0.990 | mostly separates deep builders from the large low-engagement majority |
| B2 narrowed | agent users only | 0.906 | more interesting within already-engaged users |
| Churn proxy RF | active-user subset | 0.880 | 14-day survival-style churn proxy |

Most important narrowed-model features:

| Feature | Importance |
|---|---:|
| `first_session_event_types` | 0.211 |
| `time_to_return_hours` | 0.130 |
| `first_session_duration_min` | 0.120 |
| `first_session_events` | 0.101 |
| `had_second_session` | 0.061 |

Interpretation:

- first-session breadth and early return dominate success prediction
- the product's key battle is early activation depth, not mild later-stage optimization

## Key Findings By Theme

### 1. Activation and return timing

Strongest 48-hour milestones:

| Milestone | AB if yes | AB if no | Lift | User reach |
|---|---:|---:|---:|---:|
| `reached_3_build_48h` | 45.3% | 0.8% | 56.2x | 11.6% |
| `used_agent_tool_48h` | 43.6% | 0.8% | 53.8x | 12.1% |
| `created_block_48h` | 44.4% | 0.8% | 53.4x | 11.8% |
| `had_10plus_events_48h` | 24.1% | 0.5% | 52.0x | 23.3% |
| `refactored_48h` | 46.9% | 1.1% | 42.5x | 10.6% |

Return-window split:

| Threshold | Early users | AB early | Late users | AB late |
|---|---:|---:|---:|---:|
| `<= 6h` | 905 | 25.6% | 408 | 5.9% |
| `<= 12h` | 957 | 24.6% | 356 | 5.9% |
| `<= 24h` | 1,057 | 22.8% | 256 | 5.9% |
| `<= 48h` | 1,136 | 21.9% | 177 | 4.0% |

### 2. Cohorts

| Cohort | Users | % ever agent | Avg days active | Power-user share |
|---|---:|---:|---:|---:|
| 2025-09 | 999 | 21.0% | 2.91 | 19.1% |
| 2025-10 | 472 | 29.0% | 1.27 | 19.9% |
| 2025-11 | 2,000 | 11.0% | 1.33 | 0.0% |
| 2025-12 | 1,300 | 6.0% | 1.12 | 0.0% |

Interpretation:

- Sept/Oct are mature and strong
- Nov/Dec remain heavily right-censored by the dataset end date

### 3. Time-to-Agent-Builder

This section was previously fragile and is now materially improved.

Latest step-10 output:

- `32,784` build events under the corrected filter
- `625` users reached a third build call
- median time-to-Agent-Builder: `0.0` days
- `86.4%` within `1` day
- `92.8%` within `7` days
- `97.8%` within `30` days

Interpretation:

- once users truly convert into builder behavior, they usually do so quickly
- this strongly supports product emphasis on the first session and first two days

### 4. Workflow motifs

Most recurring Builder motifs:

- `CREATE -> RUN`
- `RUN -> CREATE`
- `RUN -> REFACTOR -> RUN`
- `FINISH -> SUMMARY -> GET`

Top Agent Builder trigram in step 11:

- `RUN -> CREATE -> RUN` with `4,700` occurrences

Interpretation:

- the successful workflow is iterative and correction-oriented
- users are not "graduating" from onboarding; they are entering a live development loop

### 5. Credit, error assist, and quasi-causal evidence

Credit burn by segment:

| Segment | Avg burn | % any burn |
|---|---:|---:|
| Agent Builder | 20.09 | 43.86% |
| Manual Coder | 64.91 | 66.46% |
| Viewer | 25.12 | 42.47% |
| Ghost | 0.02 | 0.86% |

Error assist:

- `95` users used it
- `47.37%` were Agent Builders
- they averaged `9.60` active days and `361.27` tool calls

Propensity-score matching:

- matched pairs: `586`
- ATT Agent Builder difference: `+46.4pp`
- ATT active days difference: `+1.79`
- ATT return difference: `+17.4pp`

Interpretation:

- strong directional evidence that agent-assisted building contributes to deeper usage
- still not a full causal proof because unobserved confounding can remain

## New Phase Summaries

### Phase 14: Canvas Complexity

Core repeat-canvas summary:

| Group | Users | Avg days active | Second-session rate | AB share | Avg complexity |
|---|---:|---:|---:|---:|---:|
| One-off canvas | 4,505 | 1.20 | 23.3% | 3.9% | 0.73 |
| Repeat canvas | 266 | 8.34 | 99.6% | 40.6% | 15.65 |

This remains one of the strongest behavioral separators in the project.

### Phase 15: Churn Prediction

Latest churn setup:

- population: `1,177` active users
- target: `14-day survival-style churn proxy`
- proxy rate: `83.3%`
- Random Forest AUC: `0.880`
- Logistic Regression AUC: `0.855`

Interpretation:

- useful for prioritization
- must not be described as literal long-run churn after final inactivity

### Phase 16: N-gram Workflow Analysis

Latest run:

- `4,771` user sequences
- `11,949` session sequences
- `292` tool-only sequences

Interpretation:

- very useful for qualitative workflow fingerprints
- still benefits from careful publication filtering because some lifts can be dominated by low baselines

### Phase 17: Geo Analysis

Country-level metrics above threshold remain directionally useful:

| Country | Users | AB % | Agent % | Repeat session % | Churn % |
|---|---:|---:|---:|---:|---:|
| IN | 2,025 | 6.27 | 12.49 | 27.56 | 18.07 |
| US | 838 | 4.30 | 11.69 | 21.36 | 16.59 |
| GB | 318 | 2.52 | 16.67 | 37.11 | 34.59 |
| IE | 220 | 8.18 | 13.18 | 41.36 | 23.18 |
| FR | 88 | 13.64 | 23.86 | 48.86 | 34.09 |

Interpretation:

- geography refines targeting
- it does not replace the core activation narrative

### Phase 18: Intervention Scoring

| Intervention | Users | Avg priority | Avg churn risk | Avg activation | Avg struggle | Avg builder momentum | Builder share |
|---|---:|---:|---:|---:|---:|---:|---:|
| Activation nudge | 3,393 | 0.044 | 0.010 | 0.072 | 0.002 | 0.129 | 0.85% |
| Retention rescue | 846 | 0.417 | 0.875 | 0.364 | 0.026 | 0.156 | 21.04% |
| Monitor | 518 | 0.183 | 0.169 | 0.365 | 0.043 | 0.201 | 13.71% |
| Builder acceleration | 8 | 0.444 | 0.628 | 0.396 | 0.157 | 0.528 | 62.5% |
| Productive struggle support | 6 | 0.429 | 0.217 | 0.559 | 0.654 | 0.390 | 33.3% |

Interpretation:

- the majority problem is still activation
- the most valuable smaller groups are highly actionable

### Phase 19: Quality Of Struggle

| Struggle class | Users | Avg quality score | Avg abandonment risk | Avg recovery intensity | Avg churn probability | Builder share |
|---|---:|---:|---:|---:|---:|---:|
| Abandonment-prone struggle | 121 | 0.316 | 0.687 | 0.358 | 0.891 | 33.88% |
| Mixed/uncertain struggle | 231 | 0.307 | 0.349 | 0.367 | 0.154 | 17.32% |
| No visible struggle | 4,404 | 0.063 | 0.624 | 0.000 | 0.164 | 4.50% |
| Productive struggle | 15 | 0.531 | 0.245 | 0.579 | 0.113 | 40.0% |

Interpretation:

- struggle is not one class of user
- some struggling users are valuable and recovering
- some clearly need rescue before churn

### Phase 20: Path Branching

This phase is useful, but should be interpreted carefully.

Strong branch-point examples with non-trivial support include:

- `AUTH -> ONBOARD -> AGENT_CHAT -> OTHER` followed by `AGENT_OTHER`
- `CREDITS -> OTHER -> CREDITS -> AGENT_OTHER` followed by `AGENT_BUILD`
- `OTHER -> AGENT_OTHER -> AGENT_BUILD -> AGENT_RUN` followed by `AGENT_BUILD` or `AGENT_REFACTOR`

Interpretation:

- the branch layer works
- but some top-ranked rows are still low-support `100%` gaps, so public storytelling should emphasize the higher-support motifs instead

## Limitations And Cautions

### 1. Churn definition

Step 15 is explicitly a `14-day survival-style churn proxy`, not literal churn after final inactivity.

### 2. Iteration-quality leakage risk

The newer iteration-quality features are analytically useful, but some are likely too close to the operational success definition to be used naively in predictive modeling.

This is documented separately in [LEAKAGE_REVIEW.md](c:/Users/Balint/Documents/Pet_projects/zerve/docs/LEAKAGE_REVIEW.md).

### 3. Geo instability

Small-country metrics remain directional only. Public claims should stay thresholded and uncertainty-aware.

### 4. Unknown geo bucket

The `Unknown` bucket is large enough to distort naive regional comparisons if mixed into simple rollups.

### 5. Session reconstruction

Some backend-tool analyses rely on time-gap session reconstruction because backend events do not share the same session IDs as web events.

### 6. Path branching publication risk

Phase 20 is analytically useful, but some top branch gaps are still driven by very low-support prefixes. Those should not be over-emphasized in external-facing summaries.

## What Is Most Defensible Externally

The strongest externally safe conclusions are:

1. Zerve's core problem is activation into real builder behavior.
2. The first 48 hours are the decisive window.
3. Quick return and repeated work on the same canvas are some of the strongest positive signals.
4. Agent-assisted building appears to drive deeper adoption.
5. Credit pain and error help can be signs of productive value extraction.
6. Zerve can now operationalize the findings through intervention, struggle, and branching layers.

## Recommended Next Moves

### Product-facing

- redesign onboarding around a live build-run moment
- trigger re-entry interventions before the 6h and 24h return windows expire
- treat credit and error events as support and monetization triggers
- accelerate repeat-canvas users with saved-context and next-step flows

### Analytics-facing

- tighten publishable filtering for phase 20 branch points
- keep churn-proxy labeling explicit everywhere
- add uncertainty-aware geo summaries where needed
- decide which leakage-sensitive new features remain descriptive only

### Submission-facing

- use [orchestrator.py](c:/Users/Balint/Documents/Pet_projects/zerve/orchestrator.py) as the local entrypoint
- use [ZERVE_REPRODUCTION_GUIDE.md](c:/Users/Balint/Documents/Pet_projects/zerve/docs/ZERVE_REPRODUCTION_GUIDE.md) and [SUBMISSION_CHECKLIST.md](c:/Users/Balint/Documents/Pet_projects/zerve/docs/SUBMISSION_CHECKLIST.md) to reproduce the submission on Zerve
- build the submission-safe canvas first, then add Fleet and presentation layers
