import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.events import (
    AGENT_TOOL_EVENTS,
    DEPENDENCY_EVENTS,
    MANUAL_RUN_EVENTS,
    OUTPUT_EVENTS,
    STRUCTURAL_EVENTS,
    add_canvas_id,
    add_normalized_tool,
)
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.metrics import compute_canvas_complexity_score, split_first_vs_later_days

warnings.filterwarnings("ignore")

DATA_PATH = "zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"

SEGMENT_COLORS = {
    "Agent Builder": "#00b4d8",
    "Manual Coder": "#90e0ef",
    "Viewer": "#555566",
    "Ghost": "#2d2d3a",
}


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    df = add_normalized_tool(add_canvas_id(load_events(DATA_PATH)))

    canvas_df = df[df["canvas_id"].notna()].copy()
    canvas_df["event_day"] = canvas_df["timestamp"].dt.normalize()
    canvas_df["is_structural"] = canvas_df["event"].isin(STRUCTURAL_EVENTS).astype(int)
    canvas_df["is_execution"] = (
        canvas_df["event"].isin(MANUAL_RUN_EVENTS | {"agent_block_run"})
        | canvas_df["event"].isin(AGENT_TOOL_EVENTS) & canvas_df["tool"].fillna("").str.contains("run_block")
    ).astype(int)
    canvas_df["is_dependency"] = canvas_df["event"].isin(DEPENDENCY_EVENTS).astype(int)
    canvas_df["is_output"] = canvas_df["event"].isin(OUTPUT_EVENTS).astype(int)
    canvas_df["is_agent_build"] = (
        canvas_df["event"].isin(AGENT_TOOL_EVENTS)
        & canvas_df["tool"].fillna("").str.contains("create_block|refactor_block|create_edges")
    ).astype(int)

    print(f"  {len(canvas_df):,} canvas-linked events")

    canvas_metrics = (
        canvas_df.groupby(["person_id", "canvas_id"])
        .agg(
            total_events=("event", "count"),
            unique_active_days=("event_day", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            structural_actions=("is_structural", "sum"),
            execution_actions=("is_execution", "sum"),
            dependency_actions=("is_dependency", "sum"),
            output_actions=("is_output", "sum"),
            agent_build_actions=("is_agent_build", "sum"),
        )
        .reset_index()
    )
    canvas_metrics["active_days"] = canvas_metrics["unique_active_days"]
    canvas_metrics["canvas_complexity_score"] = canvas_metrics.apply(compute_canvas_complexity_score, axis=1)
    canvas_metrics["repeat_canvas"] = (canvas_metrics["unique_active_days"] >= 2).astype(int)

    growth = split_first_vs_later_days(canvas_df)
    canvas_metrics = canvas_metrics.merge(growth, on=["person_id", "canvas_id"], how="left").fillna(
        {
            "first_day_score": 0,
            "later_avg_day_score": 0,
            "later_max_day_score": 0,
            "later_active_days": 0,
            "growth_delta": 0,
        }
    )
    canvas_metrics["repeat_growth_delta"] = np.where(
        canvas_metrics["repeat_canvas"] == 1,
        canvas_metrics["growth_delta"],
        0.0,
    )
    canvas_metrics["revisit_depth_score"] = np.where(
        canvas_metrics["repeat_canvas"] == 1,
        canvas_metrics["later_max_day_score"],
        0.0,
    )
    canvas_metrics["complexity_growth_flag"] = (canvas_metrics["growth_delta"] > 0).astype(int)
    canvas_metrics["tenure_days_on_canvas"] = (
        (canvas_metrics["last_seen"] - canvas_metrics["first_seen"]).dt.total_seconds() / 86400
    ).clip(lower=0)

    user_canvas = (
        canvas_metrics.groupby("person_id")
        .agg(
            canvases_touched=("canvas_id", "nunique"),
            avg_canvas_complexity=("canvas_complexity_score", "mean"),
            max_canvas_complexity=("canvas_complexity_score", "max"),
            repeat_canvas_users=("repeat_canvas", "max"),
            repeat_canvas_count=("repeat_canvas", "sum"),
            avg_canvas_growth=("repeat_growth_delta", "mean"),
            share_of_canvases_with_growth=("complexity_growth_flag", "mean"),
            avg_canvas_active_days=("unique_active_days", "mean"),
            avg_revisit_depth=("revisit_depth_score", "mean"),
            avg_later_max_complexity=("later_max_day_score", "mean"),
        )
    )
    user_canvas["share_of_canvases_with_growth"] = user_canvas["share_of_canvases_with_growth"].fillna(0)
    user_canvas = feat.join(user_canvas).fillna(
        {
            "canvases_touched": 0,
            "avg_canvas_complexity": 0,
            "max_canvas_complexity": 0,
            "repeat_canvas_users": 0,
            "repeat_canvas_count": 0,
            "avg_canvas_growth": 0,
            "share_of_canvases_with_growth": 0,
            "avg_canvas_active_days": 0,
            "avg_revisit_depth": 0,
            "avg_later_max_complexity": 0,
        }
    )

    user_canvas.to_parquet(f"{OUTPUT_DIR}/canvas_complexity_features.parquet")
    print(f"  Saved: {OUTPUT_DIR}/canvas_complexity_features.parquet")

    fig1 = px.box(
        user_canvas.reset_index(),
        x="segment",
        y="avg_canvas_complexity",
        color="segment",
        points=False,
        title="Canvas Complexity Distribution by Segment<br><sup>One-off viewers vs returning builders</sup>",
        template="plotly_dark",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig1.write_html(f"{OUTPUT_DIR}/14_canvas_complexity_distribution.html")
    print(f"  Saved: {OUTPUT_DIR}/14_canvas_complexity_distribution.html")

    growth_by_seg = (
        user_canvas.groupby("segment")
        .agg(
            avg_growth=("avg_canvas_growth", "mean"),
            pct_growth=("share_of_canvases_with_growth", "mean"),
            users=("segment", "count"),
        )
        .reset_index()
    )
    growth_by_seg["pct_growth"] *= 100
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=growth_by_seg["segment"],
            y=growth_by_seg["avg_growth"].round(2),
            name="Avg growth delta",
            marker_color=[SEGMENT_COLORS.get(seg, "#888") for seg in growth_by_seg["segment"]],
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=growth_by_seg["segment"],
            y=growth_by_seg["pct_growth"],
            name="% canvases with growth",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#ffd166", width=3),
            marker=dict(size=9),
        )
    )
    fig2.update_layout(
        title="Canvas Growth by Segment<br><sup>Are users returning to the same canvas and deepening the workflow?</sup>",
        template="plotly_dark",
        height=420,
        yaxis=dict(title="Avg complexity growth"),
        yaxis2=dict(title="% canvases with growth", overlaying="y", side="right"),
    )
    fig2.write_html(f"{OUTPUT_DIR}/14_canvas_growth_by_segment.html")
    print(f"  Saved: {OUTPUT_DIR}/14_canvas_growth_by_segment.html")

    retention = (
        user_canvas.assign(repeat_canvas_label=np.where(user_canvas["repeat_canvas_users"] == 1, "Repeat canvas", "One-off canvas"))
        .groupby("repeat_canvas_label")
        .agg(
            users=("segment", "count"),
            avg_days_active=("days_active", "mean"),
            repeat_session_pct=("had_second_session", "mean"),
            agent_builder_pct=("segment", lambda x: (x == "Agent Builder").mean()),
        )
        .reset_index()
    )
    retention["repeat_session_pct"] *= 100
    retention["agent_builder_pct"] *= 100
    fig3 = px.bar(
        retention.melt(id_vars="repeat_canvas_label", value_vars=["avg_days_active", "repeat_session_pct", "agent_builder_pct"]),
        x="repeat_canvas_label",
        y="value",
        color="variable",
        barmode="group",
        title="Repeat Canvas Retention Signal<br><sup>Returning to the same canvas is a real engagement signal</sup>",
        template="plotly_dark",
        labels={"repeat_canvas_label": "", "value": "Metric", "variable": ""},
    )
    fig3.write_html(f"{OUTPUT_DIR}/14_repeat_canvas_retention.html")
    print(f"  Saved: {OUTPUT_DIR}/14_repeat_canvas_retention.html")

    print("\n[OK] Canvas complexity analysis complete.")


if __name__ == "__main__":
    main()
