# Zerve Activation Lens Dashboard

## What This Repo Is
This repository contains a polished, presentation-only Streamlit dashboard for visualizing the results of the Zerve user behavioral analysis pipeline.

The dashboard highlights key findings regarding user activation, retention, churn risk, and intervention targeting for Zerve. It is built to serve as a presentation layer over precomputed Zerve pipeline outputs, and no longer contains or requires the heavy backend modeling logic.

## How to Run Locally

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Deploying on Streamlit Community Cloud

This repository is ready to be connected directly to [Streamlit Community Cloud](https://share.streamlit.io/).

1. Create a new app on Streamlit Community Cloud.
2. Select this repository and the main branch.
3. Set the Main file path to `app.py`.
4. Streamlit will automatically install dependencies from `requirements.txt` and launch the app.

## Expected Outputs

The application depends on precomputed data files that should be placed in the `outputs/` folder (or the repository root). The app expects files to have the `.csv` or `.parquet` extension. Note that the original filenames prefixed with `outputs_` are fully supported.

The following files are expected to be available:
- **Main user features (required):** e.g., `outputs_user_features_segmented.csv`
- **Churn scored users (optional):** e.g., `outputs_14_churn_scored_users.parquet`
- **Intervention scored users (optional):** e.g., `outputs_18_intervention_scored_users.parquet`
- **Quality of struggle scored users (optional):** e.g., `outputs_19_quality_of_struggle_scored_users.parquet`

If any of the optional datasets are missing, the dashboard will gracefully degrade and hide the sections relying on that missing data.
