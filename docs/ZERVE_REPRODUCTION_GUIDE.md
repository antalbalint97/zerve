# Zerve Reproduction Guide

## Goal
Reproduce the full Zerve hackathon analysis inside a Zerve project using only the provided `zerve_events.csv` dataset and Python blocks.

This guide is written for the competition constraints:

- all development, analysis, and modeling must be done in Zerve
- no external datasets
- all results must be reproducible in a Zerve project
- Fleet usage should be demonstrated where it adds real value

## Inputs

- Raw data: `zerve_events.csv`
- Project code: numbered scripts `01_eda.py` through `20_path_branching_model.py`
- Pipeline entrypoint: `orchestrator.py`
- Shared helpers:
  - `analytics/io.py`
  - `analytics/events.py`
  - `analytics/metrics.py`
  - `analytics/viz.py`

## Required Python Packages

Use the repo `requirements.txt` as the source of truth. At minimum, the Zerve environment needs:

- `pandas`
- `numpy`
- `plotly`
- `pyarrow`
- `scikit-learn`
- `joblib`
- `lifelines`

## Recommended Zerve Canvas Structure

### Block 1: Ingest + EDA
Purpose:
- load `zerve_events.csv`
- inspect row counts, event mix, countries, sessions, and funnel shape

Local script mapping:
- `01_eda.py`

Writes:
- `outputs/01_event_frequency.html`
- `outputs/01_activity_timeline.html`
- `outputs/01_geo_distribution.html`
- `outputs/01_user_funnel.html`
- `outputs/01_session_depth.html`

### Block 2: Feature Engineering
Purpose:
- build the reusable user-level feature matrix

Local script mapping:
- `02_feature_engineering.py`

Writes:
- `outputs/user_features.parquet`
- `outputs/user_features.csv`

Notes:
- this is the central dependency block for nearly everything downstream
- it now includes additional raw-data signals for iteration quality, productive sessions, and canvas commitment

### Block 3: Segmentation + Cohorts
Purpose:
- assign user segments
- enrich with signup and adoption cohort views

Local script mapping:
- `03_user_segments.py`
- `04_cohort_analysis.py`

Writes:
- `outputs/user_features_segmented.parquet`
- `outputs/03_*`
- `outputs/04_*`

### Block 4: Lifecycle + Survival + Activation
Purpose:
- study early-life activation and return dynamics

Local script mapping:
- `05_lifecycle_analysis.py`
- `07_signup_hour_survival.py`
- `10_user_lifecycle.py`

Writes:
- `outputs/05_*`
- `outputs/07_*`
- `outputs/10_*`

Important note:
- `10_user_lifecycle.py` was corrected so the third-build timestamp logic is now based on actual ranked build events rather than the previous fragile extraction

### Block 5: Modeling
Purpose:
- predict Agent Builder outcomes
- compare broad vs narrowed populations

Local script mapping:
- `06_kpi_and_modeling.py`

Writes:
- `outputs/06_*`
- `outputs/model_rf_full.joblib`
- `outputs/model_rf_narrow.joblib`

### Block 6: Credit, Error, and Causal Signal
Purpose:
- analyze credit burn
- error-assist usage
- quasi-causal effect of agent adoption

Local script mapping:
- `08_india_hypothesis_success_def.py`
- `12_credit_error_propensity.py`

Writes:
- `outputs/08_*`
- `outputs/12_*`

### Block 7: Canvas Complexity + Active-User Churn Proxy
Purpose:
- measure workflow deepening on revisited canvases
- score active-user churn risk

Local script mapping:
- `14_canvas_complexity.py`
- `15_churn_prediction.py`

Writes:
- `outputs/canvas_complexity_features.parquet`
- `outputs/14_*`
- `outputs/15_*`
- `outputs/model_churn_rf.joblib`
- `outputs/feature_names_churn.joblib`

Important note:
- phase 15 models a `14-day survival-style churn proxy`, not literal post-last-activity churn

### Block 8: Workflow Motifs + Geo Analysis
Purpose:
- identify over-indexing workflow motifs
- compare country and region-level activation patterns

Local script mapping:
- `16_ngram_workflow_analysis.py`
- `17_geo_location_analysis.py`

Writes:
- `outputs/16_ngram_tables.csv`
- `outputs/16_*`
- `outputs/17_country_metrics.csv`
- `outputs/17_*`

Important note:
- n-gram lift is now smoothed and filtered for minimum baseline support
- geo outputs should exclude weak low-sample claims in presentation

### Block 9: Fleet Parallel Cohort Modeling
Purpose:
- demonstrate Zerve-native parallelism on independent signup cohorts

Local script mapping:
- `09_fleet_cohort_model.py`

Writes:
- `outputs/09_fleet_results.csv`
- `outputs/09_*`

Why this belongs in the submission:
- it is one of the clearest places where Zerve’s `spread()` / `gather()` story is genuinely product-relevant rather than decorative

### Block 10: Interpretation / Recommendation Generation
Purpose:
- turn metrics into stakeholder-ready product actions

Recommended implementation in Zerve:
- a Python block that reads the final artifacts and produces a markdown summary
- optionally a templated block that generates intervention recommendations by segment, geography, and churn-risk band

Suggested inputs:
- `outputs/user_features_segmented.parquet`
- `outputs/canvas_complexity_features.parquet`
- `outputs/15_churn_scored_users.parquet`
- `outputs/16_ngram_tables.csv`
- `outputs/17_country_metrics.csv`
- `outputs/18_intervention_summary.csv`
- `outputs/19_quality_of_struggle_summary.csv`
- `outputs/20_branching_summary.csv`

### Block 11: Hosted App / Dashboard
Purpose:
- present the submission as a polished reproducible artifact

Recommended contents:
- headline KPIs
- activation funnel
- milestone lift
- builder workflow motifs
- repeat-canvas vs one-off behavior
- churn proxy risk buckets
- thresholded geo comparison
- intervention buckets
- productive vs abandonment-prone struggle
- early branch-point summaries

## Suggested Execution Order

Run in this order:

1. `01_eda.py`
2. `02_feature_engineering.py`
3. `03_user_segments.py`
4. `04_cohort_analysis.py`
5. `05_lifecycle_analysis.py`
6. `06_kpi_and_modeling.py`
7. `07_signup_hour_survival.py`
8. `08_india_hypothesis_success_def.py`
9. `09_fleet_cohort_model.py`
10. `10_user_lifecycle.py`
11. `11_tool_sequences_session_progression.py`
12. `12_credit_error_propensity.py`
13. `14_canvas_complexity.py`
14. `15_churn_prediction.py`
15. `16_ngram_workflow_analysis.py`
16. `17_geo_location_analysis.py`
17. `18_intervention_scoring.py`
18. `19_quality_of_struggle.py`
19. `20_path_branching_model.py`
20. `python orchestrator.py --steps 1 2 3 ... 20` for an integrated rerun if needed

## Submission-Safe Variant

If time is tight, this reduced block set still reproduces the core case:

1. `01_eda.py`
2. `02_feature_engineering.py`
3. `03_user_segments.py`
4. `06_kpi_and_modeling.py`
5. `07_signup_hour_survival.py`
6. `10_user_lifecycle.py`
7. `12_credit_error_propensity.py`
8. `14_canvas_complexity.py`
9. `15_churn_prediction.py`
10. `16_ngram_workflow_analysis.py`
11. `17_geo_location_analysis.py`
12. `18_intervention_scoring.py`
13. `19_quality_of_struggle.py`
14. `20_path_branching_model.py`

This reduced flow still covers:

- the Ghost-majority problem
- Agent Builder as the success proxy
- the first 48h activation window
- repeat-canvas deepening
- active-user churn risk
- workflow motifs
- geo differences
- intervention targeting
- productive struggle vs abandonment-prone friction
- early path branching

## Artifact Contract

For reproducibility, preserve these core files:

- `outputs/user_features.parquet`
- `outputs/user_features_segmented.parquet`
- `outputs/canvas_complexity_features.parquet`
- `outputs/15_churn_scored_users.parquet`
- `outputs/16_ngram_tables.csv`
- `outputs/17_country_metrics.csv`
- `outputs/18_intervention_scored_users.parquet`
- `outputs/18_intervention_summary.csv`
- `outputs/19_quality_of_struggle_scored_users.parquet`
- `outputs/19_quality_of_struggle_summary.csv`
- `outputs/20_branching_summary.csv`
- `outputs/pipeline_log.txt`
- `outputs/pipeline_status.json`

## Competition Checklist

- Use only `zerve_events.csv`
- Keep every phase executable as a standalone Python block
- Keep `orchestrator.py` as the standalone pipeline entrypoint
- Show at least one meaningful Fleet use case
- Make assumptions explicit:
  - success proxy = `Agent Builder`
  - churn output = `14-day survival-style churn proxy`
  - geo claims are thresholded and `Unknown` is handled separately
- Ensure the final dashboard or markdown output can be regenerated from the blocks alone

## Recommended Final Submission Story

The strongest submission narrative is:

1. Most users are Ghosts or shallow explorers.
2. The first 48 hours determine whether users become builders.
3. Fast return, real build-run loops, and repeated canvas deepening are the strongest product signals.
4. Credit pain and error assist are often signs of productive struggle, not pure failure.
5. Product interventions can be targeted through activation, struggle, and branching signals.
6. The project is reproducible, modular, and implemented in Zerve-native building blocks.
