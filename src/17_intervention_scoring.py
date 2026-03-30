import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.metrics import summarize_risk
from analytics.viz import write_html

warnings.filterwarnings("ignore")

DATA_PATH = "data/zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
CANVAS_PATH = "outputs/canvas_complexity_features.parquet"
CHURN_PATH_CANDIDATES = [
    "outputs/14_churn_scored_users.parquet",
    "outputs/15_churn_scored_users.parquet",
]


def normalize_series(series: pd.Series) -> pd.Series:
    series = series.fillna(0).astype(float)
    if series.max() <= series.min():
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def bucket_from_score(series: pd.Series) -> pd.Series:
    bins = [-0.01, 0.25, 0.5, 0.75, 1.0]
    labels = ["Low", "Medium", "High", "Critical"]
    return pd.cut(series.clip(0, 1), bins=bins, labels=labels)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    canvas = pd.read_parquet(CANVAS_PATH)
    canvas.index = canvas.index.astype(str)
    churn_path = next((p for p in CHURN_PATH_CANDIDATES if Path(p).exists()), None)
    if churn_path is None:
        raise FileNotFoundError(
            f"None of the expected churn outputs exist: {CHURN_PATH_CANDIDATES}"
        )
    print(f"  Using churn file: {churn_path}")
    churn = pd.read_parquet(churn_path).set_index("person_id")
    events = load_events(DATA_PATH)

    error_users = set(events.loc[events["event"] == "agent_open_error_assist", "person_id"].astype(str))
    credit_warning_events = {
        "credits_below_1",
        "credits_below_2",
        "credits_below_3",
        "credits_below_4",
        "credits_exceeded",
    }
    credit_warning_users = (
        events[events["event"].isin(credit_warning_events)]["person_id"].astype(str).value_counts()
    )

    intervention = feat.join(
        canvas[
            [
                "avg_canvas_complexity",
                "max_canvas_complexity",
                "repeat_canvas_users",
                "repeat_canvas_count",
                "avg_canvas_growth",
                "share_of_canvases_with_growth",
                "avg_canvas_active_days",
                "avg_revisit_depth",
            ]
        ],
        how="left",
    ).join(
        churn[["churn_probability", "churn_risk_bucket"]],
        how="left",
    )
    intervention = intervention.fillna(0)
    intervention["had_error_assist"] = intervention.index.isin(error_users).astype(int)
    intervention["credit_warning_events"] = intervention.index.map(credit_warning_users).fillna(0)

    activation_signal = (
        0.30 * intervention["had_second_session"].astype(float)
        + 0.20 * normalize_series(intervention["first_session_event_types"])
        + 0.20 * normalize_series(intervention["first_session_duration_min"])
        + 0.15 * intervention["first_session_had_agent"].astype(float)
        + 0.15 * intervention["first_session_had_run"].astype(float)
    )
    struggle_signal = (
        0.35 * intervention["had_error_assist"].astype(float)
        + 0.30 * normalize_series(intervention["credit_warning_events"])
        + 0.20 * intervention["had_credit_exceeded"].astype(float)
        + 0.15 * normalize_series(intervention["credit_events"])
    )
    builder_momentum_signal = (
        0.35 * normalize_series(intervention["avg_canvas_complexity"])
        + 0.20 * intervention["repeat_canvas_users"].astype(float)
        + 0.20 * normalize_series(intervention["avg_canvas_growth"])
        + 0.15 * normalize_series(intervention["avg_revisit_depth"])
        + 0.10 * intervention["share_of_canvases_with_growth"].astype(float)
    )
    churn_risk_signal = intervention["churn_probability"].fillna(0).astype(float)

    intervention["activation_signal"] = activation_signal.clip(0, 1)
    intervention["struggle_signal"] = struggle_signal.clip(0, 1)
    intervention["builder_momentum_signal"] = builder_momentum_signal.clip(0, 1)
    intervention["churn_risk_signal"] = churn_risk_signal.clip(0, 1)

    intervention["intervention_priority_score"] = (
        0.35 * intervention["churn_risk_signal"]
        + 0.25 * intervention["struggle_signal"]
        + 0.20 * intervention["activation_signal"]
        + 0.20 * intervention["builder_momentum_signal"]
    ).clip(0, 1)

    conditions = [
        (intervention["activation_signal"] < 0.35) & (intervention["had_second_session"] == 0),
        (intervention["struggle_signal"] >= 0.45) & (intervention["builder_momentum_signal"] >= 0.25),
        (intervention["builder_momentum_signal"] >= 0.45) & (intervention["churn_risk_signal"] >= 0.45),
        (intervention["churn_risk_signal"] >= 0.60),
    ]
    labels = [
        "Activation nudge",
        "Productive struggle support",
        "Builder acceleration",
        "Retention rescue",
    ]
    intervention["recommended_intervention"] = np.select(conditions, labels, default="Monitor")
    intervention["intervention_priority_bucket"] = bucket_from_score(
        intervention["intervention_priority_score"]
    ).astype(str)
    intervention["intervention_risk_bucket"] = summarize_risk(
        intervention["intervention_priority_score"]
    ).astype(str)
    intervention["churn_risk_bucket"] = intervention["churn_risk_bucket"].astype(str)
    intervention["recommended_intervention"] = intervention["recommended_intervention"].astype(str)

    scored = intervention.reset_index()
    scored.to_parquet(f"{OUTPUT_DIR}/18_intervention_scored_users.parquet")
    print(f"  Saved: {OUTPUT_DIR}/18_intervention_scored_users.parquet")

    summary = (
        intervention.groupby("recommended_intervention")
        .agg(
            users=("segment", "count"),
            avg_priority=("intervention_priority_score", "mean"),
            avg_churn_risk=("churn_risk_signal", "mean"),
            avg_activation=("activation_signal", "mean"),
            avg_struggle=("struggle_signal", "mean"),
            avg_builder_momentum=("builder_momentum_signal", "mean"),
            pct_agent_builder=("segment", lambda x: (x == "Agent Builder").mean()),
        )
        .reset_index()
    )
    summary["pct_agent_builder"] *= 100
    summary.to_csv(f"{OUTPUT_DIR}/18_intervention_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/18_intervention_summary.csv")

    fig1 = px.bar(
        summary.sort_values("avg_priority", ascending=False),
        x="recommended_intervention",
        y="users",
        color="avg_priority",
        title="Intervention Recommendation Mix<br><sup>How the user base maps to actionable product interventions</sup>",
        template="plotly_dark",
        labels={"recommended_intervention": "", "users": "Users", "avg_priority": "Avg priority"},
        color_continuous_scale="Teal",
        text="users",
    )
    fig1.update_traces(textposition="outside")
    write_html(fig1, f"{OUTPUT_DIR}/18_intervention_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/18_intervention_mix.html")

    signal_plot = summary.melt(
        id_vars="recommended_intervention",
        value_vars=["avg_churn_risk", "avg_activation", "avg_struggle", "avg_builder_momentum"],
        var_name="metric",
        value_name="value",
    )
    metric_order = ["avg_churn_risk", "avg_activation", "avg_struggle", "avg_builder_momentum"]
    metric_labels = {
        "avg_churn_risk": "Avg churn risk",
        "avg_activation": "Avg activation",
        "avg_struggle": "Avg struggle",
        "avg_builder_momentum": "Avg builder momentum",
    }
    metric_colors = {
        "avg_churn_risk": "#ff6b6b",
        "avg_activation": "#00b4d8",
        "avg_struggle": "#ffd166",
        "avg_builder_momentum": "#90e0ef",
    }
    fig2 = go.Figure()
    for metric in metric_order:
        sub = signal_plot[signal_plot["metric"] == metric]
        if len(sub) == 0:
            continue
        fig2.add_trace(go.Bar(
            x=sub["recommended_intervention"],
            y=sub["value"],
            name=metric_labels.get(metric, metric),
            marker_color=metric_colors.get(metric, "#888"),
        ))
    fig2.update_layout(
        barmode="group",
        title="Intervention Signal Profile<br><sup>Why each intervention group is being flagged</sup>",
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Average normalized score",
        legend_title_text="",
        height=420,
    )
    write_html(fig2, f"{OUTPUT_DIR}/18_intervention_signal_profile.html")
    print(f"  Saved: {OUTPUT_DIR}/18_intervention_signal_profile.html")

    top_users = scored.sort_values("intervention_priority_score", ascending=False).head(25).copy()
    fig3 = go.Figure()
    intervention_order = top_users["recommended_intervention"].drop_duplicates().tolist()
    palette = [
        "#00b4d8",
        "#48cae4",
        "#90e0ef",
        "#ffd166",
        "#ef476f",
        "#8338ec",
    ]
    color_map = {
        label: palette[i % len(palette)]
        for i, label in enumerate(intervention_order)
    }
    for label in intervention_order:
        sub = top_users[top_users["recommended_intervention"] == label]
        if len(sub) == 0:
            continue
        fig3.add_trace(go.Scatter(
            x=sub["activation_signal"],
            y=sub["churn_risk_signal"],
            mode="markers",
            name=label,
            marker=dict(
                color=color_map.get(label, "#888"),
                size=(sub["intervention_priority_score"].fillna(0) * 28 + 8),
                opacity=0.75,
                sizemode="diameter",
            ),
            customdata=np.stack(
                [
                    sub["person_id"].astype(str),
                    sub["segment"].astype(str),
                    sub["struggle_signal"].fillna(0),
                    sub["builder_momentum_signal"].fillna(0),
                    sub["intervention_priority_score"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Intervention=%{fullData.name}<br>"
                "Person ID=%{customdata[0]}<br>"
                "Segment=%{customdata[1]}<br>"
                "Activation signal=%{x:.3f}<br>"
                "Churn-risk signal=%{y:.3f}<br>"
                "Struggle signal=%{customdata[2]:.3f}<br>"
                "Builder momentum=%{customdata[3]:.3f}<br>"
                "Priority score=%{customdata[4]:.3f}<extra></extra>"
            ),
        ))
    fig3.update_layout(
        title="Top Priority Intervention Candidates<br><sup>High-priority users by activation and churn-risk profile</sup>",
        template="plotly_dark",
        xaxis_title="Activation signal",
        yaxis_title="Churn-risk signal",
        legend_title_text="",
        height=500,
    )
    write_html(fig3, f"{OUTPUT_DIR}/18_top_intervention_candidates.html")
    print(f"  Saved: {OUTPUT_DIR}/18_top_intervention_candidates.html")

    print("\nIntervention summary:")
    print(
        summary[
            [
                "recommended_intervention",
                "users",
                "avg_priority",
                "avg_churn_risk",
                "avg_activation",
                "avg_struggle",
                "avg_builder_momentum",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n[OK] Intervention scoring complete.")

# Zerve: call main() directly (no __main__ guard)
main()
