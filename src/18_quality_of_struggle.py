import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import write_html

warnings.filterwarnings("ignore")

DATA_PATH = "zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
CANVAS_PATH = "outputs/canvas_complexity_features.parquet"
CHURN_PATH_CANDIDATES = [
    "outputs/14_churn_scored_users.parquet",
    "outputs/15_churn_scored_users.parquet",
]
INTERVENTION_PATH = "outputs/18_intervention_scored_users.parquet"

STRUGGLE_EVENTS = {
    "agent_open_error_assist",
    "credits_exceeded",
    "credits_below_1",
    "credits_below_2",
    "credits_below_3",
    "credits_below_4",
}
RECOVERY_EVENTS = {
    "run_block",
    "run_all_blocks",
    "agent_tool_call_run_block_tool",
    "agent_tool_call_create_block_tool",
    "agent_tool_call_refactor_block_tool",
    "fullscreen_preview_output",
}


def normalize(series: pd.Series) -> pd.Series:
    series = series.fillna(0).astype(float)
    if series.max() <= series.min():
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    events = load_events(DATA_PATH)
    canvas = pd.read_parquet(CANVAS_PATH)
    canvas.index = canvas.index.astype(str)
    churn_path = next((p for p in CHURN_PATH_CANDIDATES if Path(p).exists()), None)
    if churn_path is None:
        raise FileNotFoundError(
            f"None of the expected churn outputs exist: {CHURN_PATH_CANDIDATES}"
        )
    print(f"  Using churn file: {churn_path}")
    churn = pd.read_parquet(churn_path).set_index("person_id")
    intervention = pd.read_parquet(INTERVENTION_PATH).set_index("person_id")
    if "person_id" not in events.columns:
        events["person_id"] = events.index.astype(str)

    events = events.sort_values(["person_id", "timestamp"]).copy()
    struggle_df = events[events["event"].isin(STRUGGLE_EVENTS)].copy()
    recovery_df = events[events["event"].isin(RECOVERY_EVENTS)].copy()

    first_struggle = struggle_df.groupby("person_id")["timestamp"].min().rename("first_struggle_at")
    struggle_counts = struggle_df.groupby("person_id").size().rename("struggle_events")
    error_assist_counts = (
        struggle_df[struggle_df["event"] == "agent_open_error_assist"].groupby("person_id").size().rename("error_assist_events")
    )
    credit_struggle_counts = (
        struggle_df[struggle_df["event"] != "agent_open_error_assist"].groupby("person_id").size().rename("credit_struggle_events")
    )

    post = (
        struggle_df[["person_id", "timestamp"]]
        .rename(columns={"timestamp": "first_struggle_at"})
        .drop_duplicates("person_id")
        .merge(
            events[["person_id", "timestamp", "event"]],
            on="person_id",
            how="left",
        )
    )
    post = post[post["timestamp"] > post["first_struggle_at"]].copy()
    post["hours_after_struggle"] = (post["timestamp"] - post["first_struggle_at"]).dt.total_seconds() / 3600
    post["is_recovery_event"] = post["event"].isin(RECOVERY_EVENTS).astype(int)

    recovery_summary = (
        post.groupby("person_id")
        .agg(
            any_post_struggle_event=("event", "count"),
            recovery_events_post_struggle=("is_recovery_event", "sum"),
            recovered_within_24h=("hours_after_struggle", lambda x: int(((x <= 24) & (x >= 0)).any())),
            events_within_24h=("hours_after_struggle", lambda x: int(((x <= 24) & (x >= 0)).sum())),
            active_days_after_struggle=("timestamp", lambda x: x.dt.normalize().nunique()),
        )
    )

    first_recovery = (
        post[post["is_recovery_event"] == 1]
        .groupby("person_id")["hours_after_struggle"]
        .min()
        .rename("hours_to_first_recovery")
    )

    struggle = feat.join(canvas[["avg_canvas_growth", "avg_revisit_depth"]], how="left").join(
        churn[[c for c in ["churn_probability", "is_14d_survival_churn_proxy"] if c in churn.columns]],
        how="left",
    ).join(
        intervention[
            [
                "activation_signal",
                "struggle_signal",
                "builder_momentum_signal",
                "intervention_priority_score",
                "recommended_intervention",
            ]
        ],
        how="left",
    )
    struggle = struggle.join(first_struggle, how="left")
    struggle = struggle.join(struggle_counts, how="left")
    struggle = struggle.join(error_assist_counts, how="left")
    struggle = struggle.join(credit_struggle_counts, how="left")
    struggle = struggle.join(recovery_summary, how="left")
    struggle = struggle.join(first_recovery, how="left")
    struggle = struggle.fillna(
        {
            "avg_canvas_growth": 0,
            "avg_revisit_depth": 0,
            "churn_probability": 0,
            "is_14d_survival_churn_proxy": 0,
            "activation_signal": 0,
            "struggle_signal": 0,
            "builder_momentum_signal": 0,
            "intervention_priority_score": 0,
            "recommended_intervention": "Monitor",
            "struggle_events": 0,
            "error_assist_events": 0,
            "credit_struggle_events": 0,
            "any_post_struggle_event": 0,
            "recovery_events_post_struggle": 0,
            "recovered_within_24h": 0,
            "events_within_24h": 0,
            "active_days_after_struggle": 0,
            "hours_to_first_recovery": 999,
        }
    )

    struggle["had_struggle_signal"] = struggle["struggle_events"].gt(0).astype(int)
    struggle["recovery_intensity"] = (
        0.35 * struggle["recovered_within_24h"].astype(float)
        + 0.25 * normalize(struggle["recovery_events_post_struggle"])
        + 0.20 * normalize(struggle["active_days_after_struggle"])
        + 0.20 * normalize(struggle["avg_revisit_depth"])
    ).clip(0, 1)
    struggle["abandonment_risk_after_struggle"] = (
        0.45 * struggle["churn_probability"].astype(float)
        + 0.25 * (1 - struggle["recovered_within_24h"].astype(float))
        + 0.15 * (1 - normalize(struggle["events_within_24h"]))
        + 0.15 * (1 - normalize(struggle["active_days_after_struggle"]))
    ).clip(0, 1)
    struggle["quality_of_struggle_score"] = (
        0.55 * struggle["recovery_intensity"] + 0.25 * struggle["builder_momentum_signal"] + 0.20 * struggle["activation_signal"]
    ).clip(0, 1)

    conditions = [
        struggle["had_struggle_signal"] == 0,
        (struggle["had_struggle_signal"] == 1)
        & (struggle["quality_of_struggle_score"] >= 0.45)
        & (struggle["abandonment_risk_after_struggle"] < 0.45),
        (struggle["had_struggle_signal"] == 1)
        & (struggle["abandonment_risk_after_struggle"] >= 0.6),
    ]
    labels = [
        "No visible struggle",
        "Productive struggle",
        "Abandonment-prone struggle",
    ]
    struggle["struggle_class"] = np.select(conditions, labels, default="Mixed/uncertain struggle")

    scored = struggle.reset_index()
    scored.to_parquet(f"{OUTPUT_DIR}/19_quality_of_struggle_scored_users.parquet")
    print(f"  Saved: {OUTPUT_DIR}/19_quality_of_struggle_scored_users.parquet")

    summary = (
        struggle.groupby("struggle_class")
        .agg(
            users=("segment", "count"),
            avg_quality_score=("quality_of_struggle_score", "mean"),
            avg_abandonment_risk=("abandonment_risk_after_struggle", "mean"),
            avg_recovery_intensity=("recovery_intensity", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_builder_momentum=("builder_momentum_signal", "mean"),
            pct_agent_builder=("segment", lambda x: (x == "Agent Builder").mean()),
        )
        .reset_index()
    )
    summary["pct_agent_builder"] *= 100
    summary.to_csv(f"{OUTPUT_DIR}/19_quality_of_struggle_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/19_quality_of_struggle_summary.csv")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=summary["struggle_class"],
        y=summary["users"],
        marker=dict(
            color=summary["avg_quality_score"],
            colorscale="Teal",
            showscale=True,
            colorbar=dict(title="Avg quality score"),
        ),
        text=summary["users"],
        textposition="outside",
        showlegend=False,
    ))
    fig1.update_layout(
        title="Quality of Struggle Classes<br><sup>Who is struggling productively vs stalling out?</sup>",
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Users",
        height=420,
    )
    write_html(fig1, f"{OUTPUT_DIR}/19_quality_of_struggle_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/19_quality_of_struggle_mix.html")

    signal_plot = summary.melt(
        id_vars="struggle_class",
        value_vars=[
            "avg_quality_score",
            "avg_abandonment_risk",
            "avg_recovery_intensity",
            "avg_churn_probability",
        ],
        var_name="metric",
        value_name="value",
    )
    metric_order = [
        "avg_quality_score",
        "avg_abandonment_risk",
        "avg_recovery_intensity",
        "avg_churn_probability",
    ]
    metric_labels = {
        "avg_quality_score": "Avg quality score",
        "avg_abandonment_risk": "Avg abandonment risk",
        "avg_recovery_intensity": "Avg recovery intensity",
        "avg_churn_probability": "Avg churn probability",
    }
    metric_colors = {
        "avg_quality_score": "#00b4d8",
        "avg_abandonment_risk": "#ff6b6b",
        "avg_recovery_intensity": "#90e0ef",
        "avg_churn_probability": "#ffd166",
    }
    fig2 = go.Figure()
    for metric in metric_order:
        sub = signal_plot[signal_plot["metric"] == metric]
        if len(sub) == 0:
            continue
        fig2.add_trace(go.Bar(
            x=sub["struggle_class"],
            y=sub["value"],
            name=metric_labels.get(metric, metric),
            marker_color=metric_colors.get(metric, "#888"),
        ))
    fig2.update_layout(
        barmode="group",
        title="Quality of Struggle Signal Profile<br><sup>Recovery, abandonment risk, and churn by struggle class</sup>",
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Average score",
        legend_title_text="",
        height=420,
    )
    write_html(fig2, f"{OUTPUT_DIR}/19_quality_of_struggle_signals.html")
    print(f"  Saved: {OUTPUT_DIR}/19_quality_of_struggle_signals.html")

    struggle_users = scored[scored["had_struggle_signal"] == 1].copy()
    if len(struggle_users) > 0:
        fig3 = go.Figure()
        class_order = struggle_users["struggle_class"].drop_duplicates().tolist()
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
            for i, label in enumerate(class_order)
        }
        for label in class_order:
            sub = struggle_users[struggle_users["struggle_class"] == label]
            if len(sub) == 0:
                continue
            fig3.add_trace(go.Scatter(
                x=sub["recovery_intensity"],
                y=sub["abandonment_risk_after_struggle"],
                mode="markers",
                name=label,
                marker=dict(
                    color=color_map.get(label, "#888"),
                    size=(sub["struggle_events"].fillna(0) * 4 + 8),
                    opacity=0.75,
                    sizemode="diameter",
                ),
                customdata=np.stack(
                    [
                        sub["person_id"].astype(str),
                        sub["segment"].astype(str),
                        sub["recommended_intervention"].astype(str),
                        sub["quality_of_struggle_score"].fillna(0),
                        sub["struggle_events"].fillna(0),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Struggle class=%{fullData.name}<br>"
                    "Person ID=%{customdata[0]}<br>"
                    "Segment=%{customdata[1]}<br>"
                    "Recommended intervention=%{customdata[2]}<br>"
                    "Recovery intensity=%{x:.3f}<br>"
                    "Abandonment risk=%{y:.3f}<br>"
                    "Quality score=%{customdata[3]:.3f}<br>"
                    "Struggle events=%{customdata[4]}<extra></extra>"
                ),
            ))
        fig3.update_layout(
            title="Recovery vs Abandonment After Struggle<br><sup>Users with visible struggle signals only</sup>",
            template="plotly_dark",
            xaxis_title="Recovery intensity",
            yaxis_title="Abandonment risk",
            legend_title_text="",
            height=500,
        )
        write_html(fig3, f"{OUTPUT_DIR}/19_recovery_vs_abandonment.html")
        print(f"  Saved: {OUTPUT_DIR}/19_recovery_vs_abandonment.html")

    print("\nQuality-of-struggle summary:")
    print(
        summary[
            [
                "struggle_class",
                "users",
                "avg_quality_score",
                "avg_abandonment_risk",
                "avg_recovery_intensity",
                "avg_churn_probability",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n[OK] Quality of struggle analysis complete.")

# Zerve: call main() directly (no __main__ guard)
main()
