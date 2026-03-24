# Zerve Hackathon Analysis

## What This Repo Is

This repo contains a reproducible event-analysis pipeline for the Zerve hackathon case:

`What drives successful usage of Zerve?`

It combines:

- exploratory analysis
- user feature engineering
- segmentation
- lifecycle and survival analysis
- modeling
- credit/error/propensity analysis
- canvas complexity
- active-user churn proxy modeling
- workflow motif analysis
- geo analysis
- intervention scoring
- quality-of-struggle analysis
- early path branching analysis

## Main Entry Points

### Standalone pipeline runner
- `orchestrator.py`

### Legacy compatibility runner
- `13_orchestrator.py`

### Numbered analysis phases
- `01_eda.py`
- `02_feature_engineering.py`
- `03_user_segments.py`
- `04_cohort_analysis.py`
- `05_lifecycle_analysis.py`
- `06_kpi_and_modeling.py`
- `07_signup_hour_survival.py`
- `08_india_hypothesis_success_def.py`
- `09_fleet_cohort_model.py`
- `10_user_lifecycle.py`
- `11_tool_sequences_session_progression.py`
- `12_credit_error_propensity.py`
- `14_canvas_complexity.py`
- `15_churn_prediction.py`
- `16_ngram_workflow_analysis.py`
- `17_geo_location_analysis.py`
- `18_intervention_scoring.py`
- `19_quality_of_struggle.py`
- `20_path_branching_model.py`

## Shared Helper Layer

- `analytics/io.py`
- `analytics/events.py`
- `analytics/metrics.py`
- `analytics/viz.py`

These centralize data loading, event normalization, session reconstruction, scoring logic, and chart writing.

## Folder Structure

- `analytics/`
  Shared helper package for IO, event normalization, metrics, and plotting helpers.
- `docs/`
  Narrative writeups, roadmap, leakage review, reproduction guide, and the original technical document.
- `legacy/`
  Archived exploratory and draft files kept for reference but not part of the main pipeline.
- `outputs/`
  Generated artifacts, charts, parquet/csv outputs, and pipeline logs/status.
- `tests/`
  Deterministic helper-level tests.
- project root
  Runnable numbered scripts, `orchestrator.py`, `requirements.txt`, and `zerve_events.csv`.

## Current State

The repo has moved beyond the initial analysis stage.

Implemented and working:

- core pipeline through phase 20
- helper-based refactor across much of the pipeline
- fixes for the earlier weak analytical points
- intervention and struggle layers
- path branching model
- Zerve reproduction guide

Key project documents:

- `docs/CASE_NARRATIVE.md`
- `docs/TECHNICAL_APPENDIX.md`
- `docs/STATUS_AND_ROADMAP.md`
- `docs/NEXT_STEPS_IMPLEMENTATION.md`
- `docs/ZERVE_REPRODUCTION_GUIDE.md`
- `docs/LEAKAGE_REVIEW.md`

## How To Run Locally

### Full pipeline
```bash
python orchestrator.py
```

### Selected phases
```bash
python orchestrator.py --steps 14 15 16 17 18 19 20
```

### One script at a time
```bash
python 02_feature_engineering.py
python 03_user_segments.py
python 10_user_lifecycle.py
```

## Core Outputs

Important reusable artifacts:

- `outputs/user_features.parquet`
- `outputs/user_features_segmented.parquet`
- `outputs/canvas_complexity_features.parquet`
- `outputs/15_churn_scored_users.parquet`
- `outputs/16_ngram_tables.csv`
- `outputs/17_country_metrics.csv`
- `outputs/18_intervention_summary.csv`
- `outputs/19_quality_of_struggle_summary.csv`
- `outputs/20_branching_summary.csv`
- `outputs/pipeline_status.json`
- `outputs/pipeline_log.txt`

## Reproducing On Zerve

Start with:

- `docs/ZERVE_REPRODUCTION_GUIDE.md`

That guide explains:

- how to map the local pipeline into Zerve Python blocks
- which reduced submission-safe flow to use first
- what artifacts each block should write
- how to stay within the competition constraints

## Competition Constraints

This project is being packaged to satisfy the hackathon requirements:

- all development, analysis, and modeling must be reproducible in Zerve
- no external datasets
- results must be regenerable from the provided event CSV and code

## Recommended Next Move

The analysis-extension trio is complete. The next priority is no longer adding local phases by default, but packaging the current work into a clean, reproducible Zerve submission.

That means:

1. use `orchestrator.py` as the local rerun entrypoint
2. use `docs/ZERVE_REPRODUCTION_GUIDE.md` as the Zerve build plan
3. build the submission-safe Zerve canvas first
4. then add Fleet and hosted presentation layers
