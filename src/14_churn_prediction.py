import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.metrics import summarize_risk

warnings.filterwarnings("ignore")

DATA_PATH = "data/zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
CANVAS_COMPLEXITY_PATH = "outputs/canvas_complexity_features.parquet"
HORIZON_DAYS = 14
TARGET_COL = "is_14d_survival_churn_proxy"


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    events = load_events(DATA_PATH)
    canvas = pd.read_parquet(CANVAS_COMPLEXITY_PATH)
    canvas.index = canvas.index.astype(str)

    lifecycle = events.groupby("person_id")["timestamp"].agg(first_seen="min", last_seen="max")
    observation_end = events["timestamp"].max()
    lifecycle["days_observed_after_signup"] = (observation_end - lifecycle["first_seen"]).dt.total_seconds() / 86400
    lifecycle["is_censored"] = (lifecycle["days_observed_after_signup"] < HORIZON_DAYS).astype(int)
    lifecycle[TARGET_COL] = (
        ((lifecycle["last_seen"] - lifecycle["first_seen"]).dt.total_seconds() / 86400 < HORIZON_DAYS)
        & (lifecycle["is_censored"] == 0)
    ).astype(int)

    model_df = feat.join(canvas[[
        "avg_canvas_complexity",
        "max_canvas_complexity",
        "repeat_canvas_users",
        "repeat_canvas_count",
        "avg_canvas_growth",
        "share_of_canvases_with_growth",
        "avg_canvas_active_days",
    ]], how="left").join(lifecycle[["is_censored", TARGET_COL, "days_observed_after_signup"]], how="left")
    model_df = model_df.fillna(0)

    population = model_df[
        (model_df["is_censored"] == 0) &
        ((model_df["had_second_session"] == 1) | (model_df["days_active"] >= 2) | (model_df["manual_runs"] >= 1))
    ].copy()
    population[TARGET_COL] = population[TARGET_COL].astype(int)

    print(f"  Modeling population: {len(population):,} users")
    print(f"  14-day survival-style churn proxy rate: {population[TARGET_COL].mean()*100:.1f}%")

    feature_cols = [
        "manual_runs",
        "manual_run_days",
        "agent_conversations",
        "agent_messages",
        "agent_tool_calls_total",
        "agent_build_calls",
        "agent_run_calls",
        "agent_inspect_calls",
        "agent_finish_calls",
        "previewed_output",
        "signed_up",
        "completed_onboarding",
        "skipped_onboarding_form",
        "submitted_onboarding",
        "ttf_manual_run_min",
        "ttf_agent_tool_min",
        "ttf_agent_chat_min",
        "adopted_agent_early",
        "ever_used_agent",
        "ever_ran_manually",
        "first_session_events",
        "first_session_event_types",
        "first_session_duration_min",
        "first_session_had_agent",
        "first_session_had_run",
        "had_second_session",
        "time_to_return_hours",
        "signup_hour",
        "signup_is_weekend",
        "avg_canvas_complexity",
        "max_canvas_complexity",
        "repeat_canvas_users",
        "repeat_canvas_count",
        "avg_canvas_growth",
        "share_of_canvases_with_growth",
        "avg_canvas_active_days",
        "credit_events",
        "had_credit_exceeded",
        "had_addon_credits",
    ]
    feature_cols += [c for c in population.columns if c.startswith("ref_")]
    feature_cols = [c for c in feature_cols if c in population.columns]

    X = population[feature_cols].fillna(0)
    y = population[TARGET_COL]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nCross-validation:")
    for name, model in models.items():
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        ap = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        print(f"  {name:<20} AUC={auc.mean():.3f}+/-{auc.std():.3f}  AP={ap.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC", "Precision-Recall"])
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        probs = fitted.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ROC {roc_auc_score(y_test, probs):.3f}"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{name} PR {average_precision_score(y_test, probs):.3f}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=1)
    fig.update_layout(
        title=f"Active-User Churn Proxy Prediction Quality<br><sup>{HORIZON_DAYS}-day survival-style churn proxy, not literal post-last-activity churn</sup>",
        template="plotly_dark",
        height=420,
    )
    fig.write_html(f"{OUTPUT_DIR}/15_churn_roc_pr.html")
    print(f"  Saved: {OUTPUT_DIR}/15_churn_roc_pr.html")

    rf = models["Random Forest"]
    rf.fit(X, y)
    joblib.dump(rf, f"{OUTPUT_DIR}/model_churn_rf.joblib")
    joblib.dump(feature_cols, f"{OUTPUT_DIR}/feature_names_churn.joblib")

    fi = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False).head(15)
    fig_fi = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
        title="Top Churn Proxy Risk Drivers<br><sup>Higher importance means more predictive of the active-user 14-day churn proxy</sup>",
        template="plotly_dark",
        labels={"feature": "", "importance": "Importance"},
    )
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
    fig_fi.write_html(f"{OUTPUT_DIR}/15_churn_feature_importance.html")
    print(f"  Saved: {OUTPUT_DIR}/15_churn_feature_importance.html")

    scored = population.copy()
    scored["churn_probability"] = rf.predict_proba(X)[:, 1]
    scored["churn_risk_bucket"] = summarize_risk(scored["churn_probability"]).astype(str)
    scored["target_name"] = TARGET_COL
    scored["target_description"] = f"{HORIZON_DAYS}-day survival-style churn proxy from first-to-last observed activity"
    scored.reset_index().to_parquet(f"{OUTPUT_DIR}/15_churn_scored_users.parquet")
    print(f"  Saved: {OUTPUT_DIR}/15_churn_scored_users.parquet")

    risk = (
        scored.groupby("churn_risk_bucket")
        .agg(
            users=(TARGET_COL, "count"),
            actual_churn_pct=(TARGET_COL, "mean"),
            avg_days_active=("days_active", "mean"),
            avg_canvas_complexity=("avg_canvas_complexity", "mean"),
        )
        .reset_index()
    )
    risk["actual_churn_pct"] *= 100
    fig_risk = px.bar(
        risk.melt(id_vars="churn_risk_bucket", value_vars=["users", "actual_churn_pct", "avg_days_active", "avg_canvas_complexity"]),
        x="churn_risk_bucket",
        y="value",
        color="variable",
        barmode="group",
        title="Churn Proxy Risk Buckets<br><sup>Intervention targeting using the 14-day survival-style churn proxy</sup>",
        template="plotly_dark",
        labels={"churn_risk_bucket": "Risk bucket", "value": "Metric", "variable": ""},
    )
    fig_risk.write_html(f"{OUTPUT_DIR}/15_churn_risk_buckets.html")
    print(f"  Saved: {OUTPUT_DIR}/15_churn_risk_buckets.html")

    print("\n[OK] Churn proxy prediction complete.")


if __name__ == "__main__":
    main()
