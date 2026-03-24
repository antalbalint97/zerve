# Next Steps Implementation

## What Was Implemented

This cycle moved the repo in four directions:

1. safer helper-based refactoring
2. correction of known weak analytical points
3. addition of new raw-data features
4. competition-ready reproducibility planning

## Refactor Progress

### Shared substrate
The shared helper layer now includes:

- `analytics/io.py`
- `analytics/events.py`
- `analytics/metrics.py`
- `analytics/viz.py`

These now centralize:

- raw-event loading
- feature loading
- output directory creation
- tool normalization
- canvas extraction
- session reconstruction
- complexity and n-gram scoring
- common chart writing

### Scripts partially migrated
The following scripts now use the helper layer at least for loading and/or output handling:

- `02_feature_engineering.py`
- `01_eda.py`
- `03_user_segments.py`
- `04_cohort_analysis.py`
- `05_lifecycle_analysis.py`
- `06_kpi_and_modeling.py`
- `07_signup_hour_survival.py`
- `08_india_hypothesis_success_def.py`
- `11_tool_sequences_session_progression.py`
- `12_credit_error_propensity.py`
- `13_orchestrator.py`
- `14_canvas_complexity.py`
- `15_churn_prediction.py`
- `16_ngram_workflow_analysis.py`
- `17_geo_location_analysis.py`

### Still worth refactoring next
The next high-value refactor targets are:

- `10_user_lifecycle.py`
- optional chart-style cleanup across all remaining scripts
- extracting a publish-ready summary artifact layer
- optional helper extraction for reusable modeling and cohort-chart code

## Analytical Corrections Applied

### Step 10
`10_user_lifecycle.py` now uses a stricter build-event filter and ranked build extraction for the third build timestamp.

This directly addresses the earlier contradiction where the script reported zero users reaching the third build despite phase-level evidence that some users clearly did.

### Step 14
Canvas growth logic now emphasizes repeated-canvas deepening instead of penalizing users who front-loaded their first day.

New repeated-canvas-oriented fields include:

- `repeat_growth_delta`
- `revisit_depth_score`
- `avg_revisit_depth`
- `avg_later_max_complexity`

### Step 15
The target is now named explicitly as a `14-day survival-style churn proxy`.

This avoids overstating it as literal churn and carries the target metadata into the scored-user artifact.

### Step 16
N-gram lift now uses smoothing and minimum baseline support, with a `publishable` flag to separate stable motifs from fragile low-baseline artifacts.

### Step 17
Geo analysis now:

- respects country minimum thresholds in the chart layer
- keeps `Unknown` out of regional rollups
- adds Wilson confidence interval columns for Agent Builder rates

## New Raw-Data Signals Added

The feature matrix now includes additional interpretable signals from raw events.

### Iteration quality
- `create_run_transitions`
- `run_refactor_transitions`
- `finish_summary_transitions`
- `agent_tool_transitions_total`
- `create_run_alternation_rate`
- `refactor_after_run_rate`
- `finish_summary_rate`

### Session structure
- `productive_sessions`
- `total_sessions_derived`
- `max_events_per_session`
- `productive_session_share`

### Canvas commitment
- `primary_canvas_event_share`
- `repeat_canvas_count_raw`
- `one_day_canvas_share`

These should be treated as candidate signals first. The next step is ablation testing to determine which ones genuinely improve prediction or intervention design.

## Recommended Next Coding Steps

### Wave 1
- complete helper-style cleanup of `10_user_lifecycle.py`
- add a compact summary artifact script that exports the top findings into one CSV/markdown bundle
- add a validation mode to the most important scripts for schema and dependency checks

### Wave 2
- run ablation tests on the newly added raw-data features
- test whether iteration-quality, productive-session, and canvas-commitment signals improve the narrowed Builder model
- decide which new fields are descriptive only vs worthy of downstream modeling

### Wave 3
- add a publishable summary table script that exports key metrics in one compact CSV or markdown artifact
- add intervention-oriented downstream phases:
  - `18_intervention_scoring.py`
  - `19_quality_of_struggle.py`
  - `20_path_branching_model.py`

## Recommended Next Analysis Improvements

- fix or remove any remaining metric names that imply literal churn
- add uncertainty bars or bootstrap intervals to the most important public geo charts
- test whether the new iteration-quality signals improve the narrowed Builder model
- add confidence-aware n-gram ranking so support and lift are both visible
- compare repeated-canvas users vs one-day-canvas users within the same acquisition channels

## Ordered Forward Strategy

### 1. Stabilize the architecture
- finish the helper-style cleanup of `10_user_lifecycle.py`
- keep numbered scripts as the public interface
- avoid moving core business logic into opaque shared packages

### 2. Validate the new signals
- run ablations on the 79-feature matrix
- identify which new signals are robust enough for narrative use
- keep only the strongest additions in model-facing feature sets

### 3. Add product-facing downstream phases
- `18_intervention_scoring.py`
  - rank users for onboarding nudge, struggle support, or builder acceleration
- `19_quality_of_struggle.py`
  - separate healthy engaged friction from abandonment-prone friction
- `20_path_branching_model.py`
  - quantify where Ghost, Viewer, and Builder paths diverge in the first few steps

Status:
- `18_intervention_scoring.py` implemented
- `19_quality_of_struggle.py` implemented
- `20_path_branching_model.py` implemented

Current next focus:
- translate the strongest analytical pieces into the actual Zerve competition canvas
- add a compact publishable summary artifact

### 4. Build the competition canvas on Zerve
- translate the pipeline into the block plan in `docs/ZERVE_REPRODUCTION_GUIDE.md`
- prioritize a submission-safe path first
- then add Fleet and hosted presentation layers once the base run is stable

### 5. Prepare a publishable package
- generate a compact summary artifact
- align names in charts and markdown with the final target definitions
- ensure the final presentation distinguishes descriptive, predictive, and quasi-causal findings

## Reproducibility Readiness

The repo now includes `docs/ZERVE_REPRODUCTION_GUIDE.md`, which maps the local pipeline to a competition-safe Zerve canvas layout.

That guide should be treated as the starting point for the actual submission project build-out.
