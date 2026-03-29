import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 03_user_segments.py  --  Behavioral Segmentation
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features.parquet
Output: outputs/user_features_segmented.parquet
        outputs/03_segments_*.html

4 segments based on real Zerve usage patterns:

  Agent Builder   -- agent writes/refactors the workflow
  Agent Runner    -- agent runs the code, but also works manually
  Manual Coder    -- traditional notebook user, minimal agent usage
  Viewer/Ghost    -- browses or barely uses the platform
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import warnings
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features
from analytics.viz import write_html
warnings.filterwarnings("ignore")

INPUT_PATH = "outputs/user_features.parquet"
ensure_output_dir(OUTPUT_DIR)

# -- LOAD -------------------------------------------------
print("Loading features...")
feat = load_features(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

# -- RULE-BASED SEGMENTATION -------------------------------
# Primarily rule-based, because clustering would converge on the 95% ghost group
# -- the rules are Zerve-specific and interpretable

def assign_segment(row):
    has_agent      = row["ever_used_agent"] >= 1
    build_calls    = row.get("agent_build_calls", 0)
    run_calls      = row.get("agent_run_calls", 0)
    manual_runs    = row["manual_runs"]
    days           = row["days_active"]
    total          = row["total_events"]
    conversations  = row.get("agent_conversations", 0)

    # Ghost: almost no activity
    if total <= 5 or days == 0:
        return "Ghost"

    # Agent Builder: agent primarily builds/refactors
    if has_agent and build_calls >= 3:
        return "Agent Builder"

    # Agent Runner: agent runs code, but manual work also present
    if has_agent and (run_calls >= 3 or conversations >= 2):
        return "Agent Runner"

    # Manual Coder: runs code without agent or with minimal agent usage
    if manual_runs >= 3:
        return "Manual Coder"

    # Viewer/Explorer: browses, fullscreen, but doesn't run much
    return "Viewer"

feat["segment"] = feat.apply(assign_segment, axis=1)

# -- SEGMENT STATISTICS ------------------------------------
print("\n-- Segment distribution --")
seg_stats = (
    feat.groupby("segment").agg(
        users              = ("total_events", "count"),
        avg_days_active    = ("days_active", "mean"),
        avg_manual_runs    = ("manual_runs", "mean"),
        avg_agent_tools    = ("agent_tool_calls_total", "mean"),
        avg_build_calls    = ("agent_build_calls", "mean"),
        avg_run_calls      = ("agent_run_calls", "mean"),
        avg_conversations  = ("agent_conversations", "mean"),
        avg_canvases       = ("unique_canvases", "mean"),
        pct_credit_exceeded= ("had_credit_exceeded", "mean"),
    )
    .round(2)
)
seg_stats["pct_of_users"] = (seg_stats["users"] / len(feat) * 100).round(1)

# Sort by engagement
seg_order = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]
seg_stats = seg_stats.reindex([s for s in seg_order if s in seg_stats.index])

print(seg_stats[["users", "pct_of_users", "avg_days_active",
                  "avg_manual_runs", "avg_agent_tools", "avg_build_calls"]].to_string())

# Save segment order and colors
SEGMENT_COLORS = {
    "Agent Builder" : "#00b4d8",
    "Agent Runner"  : "#48cae4",
    "Manual Coder"  : "#90e0ef",
    "Viewer"        : "#555566",
    "Ghost"         : "#2d2d3a",
}

# -- VISUALIZATIONS ----------------------------------------

# 1. Segment size pie
fig1 = go.Figure(go.Pie(
    labels=seg_stats.index,
    values=seg_stats["users"],
    hole=0.45,
    marker_colors=[SEGMENT_COLORS.get(s, "#888") for s in seg_stats.index],
    textinfo="label+percent",
    textfont_size=12,
))
fig1.update_layout(
    title="Zerve User Segments<br><sup>Behavior-based segmentation</sup>",
    template="plotly_dark",
    height=450,
)
write_html(fig1, f"{OUTPUT_DIR}/03_segment_sizes.html")
print(f"\n  Saved: {OUTPUT_DIR}/03_segment_sizes.html")

# 2. Segment profile radar
radar_metrics = {
    "Days active"      : "avg_days_active",
    "Manual runs": "avg_manual_runs",
    "Agent tool calls" : "avg_agent_tools",
    "Agent build"      : "avg_build_calls",
    "Agent run"        : "avg_run_calls",
    "Canvases"         : "avg_canvases",
}

fig2 = go.Figure()
for seg in ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer"]:
    if seg not in seg_stats.index:
        continue
    row = seg_stats.loc[seg]
    vals = []
    for _, col in radar_metrics.items():
        col_max = seg_stats[col].max()
        vals.append(row[col] / col_max if col_max > 0 else 0)

    cats = list(radar_metrics.keys())
    fig2.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name=seg,
        line_color=SEGMENT_COLORS.get(seg, "#888"),
        opacity=0.7,
    ))

fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Segment Behavioral Profile<br><sup>Normalized values</sup>",
    template="plotly_dark",
    height=500,
)
write_html(fig2, f"{OUTPUT_DIR}/03_segment_radar.html")
print(f"  Saved: {OUTPUT_DIR}/03_segment_radar.html")

# 3. Agent tool mix by segment
tool_cols = [c for c in feat.columns if c.startswith("tool_")]
if tool_cols:
    tool_by_seg = (
        feat.groupby("segment")[tool_cols].mean()
        .reindex([s for s in seg_order if s in feat["segment"].unique()])
    )
    tool_by_seg.columns = [c.replace("tool_", "").replace("_", " ") for c in tool_by_seg.columns]
    # Top 6 tool
    top_tools = tool_by_seg.sum().nlargest(6).index.tolist()
    tool_by_seg = tool_by_seg[top_tools]

    fig3 = px.bar(
        tool_by_seg.reset_index().melt(id_vars="segment"),
        x="segment", y="value", color="variable",
        barmode="stack",
        title="Agent Tool Usage by Segment<br><sup>Avg tool calls per user</sup>",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Teal,
        labels={"value": "Avg tool calls", "variable": "Tool", "segment": ""},
    )
    write_html(fig3, f"{OUTPUT_DIR}/03_segment_tool_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/03_segment_tool_mix.html")

# 4. Signup cohort x segment heatmap
cohort_seg = (
    feat.groupby(["signup_cohort", "segment"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=[s for s in seg_order if s in feat["segment"].unique()])
)
cohort_seg_pct = cohort_seg.div(cohort_seg.sum(axis=1), axis=0) * 100

fig4 = px.imshow(
    cohort_seg_pct.T,
    text_auto=".1f",
    color_continuous_scale="Blues",
    title="Segment Proportions by Signup Cohort (%)<br><sup>Which cohort produced the most power users?</sup>",
    template="plotly_dark",
    labels=dict(x="Signup cohort", y="Segment", color="%"),
    aspect="auto",
)
write_html(fig4, f"{OUTPUT_DIR}/03_cohort_segment_heatmap.html")
print(f"  Saved: {OUTPUT_DIR}/03_cohort_segment_heatmap.html")

# 5. Agent adoption: early vs late
if "ttf_agent_tool_min" in feat.columns:
    non_ghost = feat[feat["segment"] != "Ghost"].copy()
    non_ghost["agent_adoption"] = "Never used agent"
    non_ghost.loc[non_ghost["ever_used_agent"] == 1, "agent_adoption"] = "Late adopter (1h+)"
    non_ghost.loc[non_ghost["adopted_agent_early"] == 1, "agent_adoption"] = "Early adopter (<1h)"

    adoption_seg = (
        non_ghost.groupby(["segment", "agent_adoption"])
        .size().unstack(fill_value=0)
    )
    melted = adoption_seg.reset_index().melt(id_vars="segment")
    # Safe color map: only include groups that actually exist in data
    _all_colors = {
        "Early adopter (<1h)": "#00b4d8",
        "Late adopter (1h+)": "#90e0ef",
        "Never used agent": "#555566",
    }
    _actual = set(melted["agent_adoption"].unique())
    _safe_colors = {k: v for k, v in _all_colors.items() if k in _actual}
    fig5 = px.bar(
        melted,
        x="segment", y="value", color="agent_adoption",
        barmode="group",
        title="Agent Adoption by Segment<br><sup>Early vs late vs never adopted</sup>",
        template="plotly_dark",
        color_discrete_map=_safe_colors,
        labels={"value": "Users", "segment": "", "agent_adoption": ""},
    )
    write_html(fig5, f"{OUTPUT_DIR}/03_agent_adoption_by_segment.html")
    print(f"  Saved: {OUTPUT_DIR}/03_agent_adoption_by_segment.html")

# -- SAVE -------------------------------------------------
feat.to_parquet(f"{OUTPUT_DIR}/user_features_segmented.parquet")
feat.to_csv(f"{OUTPUT_DIR}/user_features_segmented.csv")

print(f"\n  Saved: {OUTPUT_DIR}/user_features_segmented.parquet")
print(f"\n  Segment summary:")
for seg in seg_order:
    if seg in seg_stats.index:
        row = seg_stats.loc[seg]
        print(f"    {seg:<20} {int(row['users']):>5,} user  ({row['pct_of_users']:.1f}%)")
