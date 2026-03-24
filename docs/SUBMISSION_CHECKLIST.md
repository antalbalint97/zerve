# Submission Checklist

## Goal

Use this checklist to move from the local repo to a reproducible Zerve hackathon submission without losing parity.

## Repo State To Freeze

Before committing, confirm:

- `python -m unittest tests.test_analytics_helpers` passes
- `python orchestrator.py --steps 18 19 20` passes
- `README.md` reflects the current repo layout
- `docs/ZERVE_REPRODUCTION_GUIDE.md` matches the current phases and outputs
- `outputs/pipeline_status.json` and `outputs/pipeline_log.txt` are present if you want a frozen reference snapshot

## Files To Commit

Commit:

- runnable scripts `01_eda.py` through `20_path_branching_model.py`
- `orchestrator.py`
- `13_orchestrator.py` as compatibility wrapper
- `analytics/`
- `docs/`
- `tests/`
- `requirements.txt`
- optionally `outputs/` if you want reference artifacts in the repo

## Zerve Build Order

Build the submission-safe version first:

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

Then add:

1. `04_cohort_analysis.py`
2. `05_lifecycle_analysis.py`
3. `08_india_hypothesis_success_def.py`
4. `09_fleet_cohort_model.py`
5. `11_tool_sequences_session_progression.py`
6. interpretation / markdown block
7. hosted app / dashboard block

## Required Narrative Labels

Keep these labels explicit in the final submission:

- success proxy = `Agent Builder`
- churn target = `14-day survival-style churn proxy`
- iteration-quality features = useful but leakage-sensitive for prediction
- geo analysis = thresholded and `Unknown` handled separately

## Final Parity Checks On Zerve

Verify these outputs regenerate:

- `outputs/user_features.parquet`
- `outputs/user_features_segmented.parquet`
- `outputs/canvas_complexity_features.parquet`
- `outputs/15_churn_scored_users.parquet`
- `outputs/17_country_metrics.csv`
- `outputs/18_intervention_summary.csv`
- `outputs/19_quality_of_struggle_summary.csv`
- `outputs/20_branching_summary.csv`

## Submission Story

The cleanest competition story is:

1. most users are Ghosts or shallow explorers
2. the first 48 hours drive builder activation
3. repeated canvas deepening and build-run loops are the strongest positive signals
4. some struggle is productive, some predicts abandonment
5. Zerve can operationalize this with targeted interventions
6. the full workflow is reproducible inside Zerve
