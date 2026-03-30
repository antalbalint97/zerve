# Zerve Activation Lens

Interactive dashboard and analysis pipeline for Zerve user behavioral data — activation, retention, churn risk, and intervention targeting.

**Live dashboard:** [https://activatelens.streamlit.app/](https://activatelens.streamlit.app/)

---

## Project Structure

```
zerve/
├── app.py                  # Streamlit dashboard (presentation layer)
├── requirements.txt
├── analytics/              # Shared helper modules
│   ├── events.py           # Event categorization & session reconstruction
│   ├── io.py               # Data loading/saving, path constants
│   ├── metrics.py          # Complexity scores, statistical helpers
│   └── viz.py              # Plotly layout config, color palettes
├── src/                    # Analysis pipeline scripts (run in order)
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_user_segments.py
│   ├── ...
│   ├── 19_path_branching_model.py
│   └── orchestrator.py     # Runs the full pipeline
├── data/
│   └── zerve_events.csv    # Raw event data (input)
├── outputs/                # Generated CSVs, Parquets, HTML charts
├── docs/                   # Case narrative, technical docs
├── legacy/                 # Deprecated scripts (kept for reference)
└── tests/
    └── test_analytics_helpers.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

### Live (no setup needed)

[https://activatelens.streamlit.app/](https://activatelens.streamlit.app/)

### Locally

```bash
streamlit run app.py
```

The dashboard reads pre-computed files from `outputs/`. If you haven't run the pipeline yet, the sections that depend on those files will be hidden or degraded gracefully.

---

## Running the Analysis Pipeline

The pipeline reads `data/zerve_events.csv` and writes results to `outputs/`. Run everything from the **project root**.

### Full pipeline

```bash
python src/orchestrator.py
```

### Start from a specific step

```bash
python src/orchestrator.py --from-step 5
```

### Run specific steps only

```bash
python src/orchestrator.py --steps 2 3 6
```

### Dry run (validate without executing)

```bash
python src/orchestrator.py --dry-run
```

### Stop on first failure

```bash
python src/orchestrator.py --stop-on-failure
```

### Run a single script directly

```bash
python src/01_eda.py
python src/14_churn_prediction.py
```

All scripts are self-contained — they add the project root to `sys.path` automatically, so they can be run from any directory.

---

## Pipeline Steps

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_eda.py` | Exploratory data analysis — event frequencies, funnel, geo charts |
| 02 | `02_feature_engineering.py` | User-level feature matrix → `user_features.parquet` |
| 03 | `03_user_segments.py` | Behavioral segmentation (Agent Builder / Runner / Manual Coder / Ghost) |
| 04 | `04_cohort_analysis.py` | Signup-month cohorts + agent adoption cohorts |
| 05 | `05_lifecycle_analysis.py` | Conversion funnel and drop-off analysis |
| 06 | `06_kpi_and_modeling.py` | KPI matrix + early-signal RF model |
| 07 | `07_signup_hour_survival.py` | Signup-hour effects + Kaplan-Meier survival curves |
| 08 | `08_india_hypothesis_success_def.py` | India hypothesis validation + success definition |
| 09 | `09_fleet_cohort_model.py` | Fleet-style parallel RF model by cohort |
| 10 | `10_user_lifecycle.py` | Activation milestones and time-to-Agent-Builder |
| 11 | `11_tool_sequences_session_progression.py` | Tool bigram/trigram workflow fingerprints |
| 12 | `12_credit_error_propensity.py` | Credit burn rate, error-assist signal, propensity matching |
| 13 | `13_canvas_complexity.py` | Canvas complexity growth + repeat-canvas retention |
| 14 | `14_churn_prediction.py` | 14-day churn model for active users |
| 15 | `15_ngram_workflow_analysis.py` | Lifecycle and session motif analysis |
| 16 | `16_geo_location_analysis.py` | Country/region onboarding, churn, complexity comparison |
| 17 | `17_intervention_scoring.py` | Ranks users for activation nudge, struggle support, retention rescue |
| 18 | `18_quality_of_struggle.py` | Productive struggle vs abandonment-prone friction |
| 19 | `19_path_branching_model.py` | Where Ghost, Viewer, and Builder paths diverge |

Pipeline status and logs are written to `outputs/pipeline_status.json` and `outputs/pipeline_log.txt`.

---

## Deploying the Dashboard (Streamlit Community Cloud)

1. Connect this repository to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Set **Main file path** to `app.py`.
3. Streamlit installs dependencies from `requirements.txt` automatically.
