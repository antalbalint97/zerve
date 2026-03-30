import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 04_cohort_analysis.py  --  Cohort Analysis
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/04_cohort_*.html

TWO SEPARATE COHORT ANALYSES:

  A) Signup month-based cohort
     -- Which month did users register in?
     -- How did the platform user base evolve?
     -- Which cohort became the most engaged?

  B) Cohort by agent adoption timing
     -- Early adopters (<1h after signup)
     -- Late adopters (1h+)
     -- Never adopters
     This is SEPARATE from the signup cohort analysis!
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features
from analytics.viz import write_html

INPUT_PATH = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading segmented features...")
feat = load_features(INPUT_PATH)
print(f"  {len(feat):,} users")

SEGMENT_COLORS = {
    "Agent Builder" : "#00b4d8",
    "Agent Runner"  : "#48cae4",
    "Manual Coder"  : "#90e0ef",
    "Viewer"        : "#555566",
    "Ghost"         : "#2d2d3a",
}
SEG_ORDER = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]

# =======================================================
# A) SIGNUP MONTH COHORT ANALYSIS
# =======================================================
print("\n-- A) Signup Cohort Analysis --")

cohort = feat.groupby("signup_cohort").agg(
    users              = ("total_events", "count"),
    avg_days_active    = ("days_active", "mean"),
    avg_manual_runs    = ("manual_runs", "mean"),
    avg_agent_tools    = ("agent_tool_calls_total", "mean"),
    avg_canvases       = ("unique_canvases", "mean"),
    avg_conversations  = ("agent_conversations", "mean"),
    pct_ever_agent     = ("ever_used_agent", "mean"),
    pct_early_adopter  = ("adopted_agent_early", "mean"),
    pct_credit_exceeded= ("had_credit_exceeded", "mean"),
    median_tenure_days = ("tenure_days", "median"),
).sort_index()

cohort["pct_ever_agent"]    *= 100
cohort["pct_early_adopter"] *= 100
cohort["pct_credit_exceeded"] *= 100

print(cohort[["users","avg_days_active","pct_ever_agent","avg_agent_tools","median_tenure_days"]].to_string())

# A1. Cohort size and engagement
fig_a1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Number of registrants by cohort",
        "Avg active days",
        "Agent adoption rate (%)",
        "Average agent tool calls / user",
    ]
)

x = cohort.index.tolist()
colors = ["#00b4d8", "#48cae4", "#90e0ef", "#0077b6"]

fig_a1.add_trace(go.Bar(x=x, y=cohort["users"], marker_color=colors,
                         name="Users", showlegend=False), row=1, col=1)
fig_a1.add_trace(go.Bar(x=x, y=cohort["avg_days_active"].round(2),
                         marker_color=colors, name="Avg days", showlegend=False), row=1, col=2)
fig_a1.add_trace(go.Bar(x=x, y=cohort["pct_ever_agent"].round(1),
                         marker_color=colors, name="% agent", showlegend=False), row=2, col=1)
fig_a1.add_trace(go.Bar(x=x, y=cohort["avg_agent_tools"].round(2),
                         marker_color=colors, name="Avg tools", showlegend=False), row=2, col=2)

fig_a1.update_layout(
    title="Signup Cohort Comparison<br><sup>Which month registered the most engaged users?</sup>",
    template="plotly_dark",
    height=600,
)
write_html(fig_a1, f"{OUTPUT_DIR}/04_cohort_signup_overview.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_signup_overview.html")

# A2. Segment composition by cohort
seg_cohort = (
    feat.groupby(["signup_cohort", "segment"]).size()
    .unstack(fill_value=0)
    .reindex(columns=[s for s in SEG_ORDER if s in feat["segment"].unique()])
)
seg_cohort_pct = seg_cohort.div(seg_cohort.sum(axis=1), axis=0) * 100

fig_a2 = go.Figure()
for seg in [s for s in SEG_ORDER if s in seg_cohort_pct.columns]:
    fig_a2.add_trace(go.Bar(
        x=seg_cohort_pct.index,
        y=seg_cohort_pct[seg],
        name=seg,
        marker_color=SEGMENT_COLORS.get(seg, "#888"),
    ))

fig_a2.update_layout(
    barmode="stack",
    title="Segment Mix by Signup Cohort (%)<br><sup>Has the power user ratio improved over time?</sup>",
    yaxis_title="Share (%)",
    template="plotly_dark",
    height=450,
)
write_html(fig_a2, f"{OUTPUT_DIR}/04_cohort_segment_mix.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_segment_mix.html")

# A3. Engagement metrics by cohort -- line chart
fig_a3 = make_subplots(specs=[[{"secondary_y": True}]])
fig_a3.add_trace(go.Scatter(
    x=x, y=cohort["avg_days_active"].round(2),
    mode="lines+markers", name="Avg active days",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=10),
), secondary_y=False)
fig_a3.add_trace(go.Scatter(
    x=x, y=cohort["pct_ever_agent"].round(1),
    mode="lines+markers", name="Agent adoption %",
    line=dict(color="#90e0ef", width=3, dash="dot"),
    marker=dict(size=10, symbol="diamond"),
), secondary_y=True)
fig_a3.update_layout(
    title="Engagement Trend by Cohort<br><sup>Is the platform learning? Are metrics improving?</sup>",
    template="plotly_dark",
    height=400,
)
fig_a3.update_yaxes(title_text="Avg active days", secondary_y=False)
fig_a3.update_yaxes(title_text="Agent adoption %", secondary_y=True)
write_html(fig_a3, f"{OUTPUT_DIR}/04_cohort_engagement_trend.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_engagement_trend.html")

# =======================================================
# B) AGENT ADOPTION COHORT -- SEPARATE ANALYSIS
# =======================================================
print("\n-- B) Agent Adoption Cohort Analysis --")
print("   (This is NOT the signup cohort -- this is a behavior-based group)")

feat["adoption_cohort"] = "Never adopted"
feat.loc[feat["ever_used_agent"] == 1, "adoption_cohort"] = "Late adopter (1h+)"
feat.loc[feat["adopted_agent_early"] == 1, "adoption_cohort"] = "Early adopter (<1h)"

ADOPTION_COLORS = {
    "Early adopter (<1h)" : "#00b4d8",
    "Late adopter (1h+)" : "#90e0ef",
    "Never adopted"   : "#444455",
}
ADOPTION_ORDER = ["Early adopter (<1h)", "Late adopter (1h+)", "Never adopted"]

adoption = feat.groupby("adoption_cohort").agg(
    users              = ("total_events", "count"),
    avg_days_active    = ("days_active", "mean"),
    avg_manual_runs    = ("manual_runs", "mean"),
    avg_agent_tools    = ("agent_tool_calls_total", "mean"),
    avg_build_calls    = ("agent_build_calls", "mean"),
    avg_canvases       = ("unique_canvases", "mean"),
    median_tenure      = ("tenure_days", "median"),
    pct_credit_exceeded= ("had_credit_exceeded", "mean"),
).reindex(ADOPTION_ORDER)

adoption["pct_credit_exceeded"] *= 100
adoption["pct_of_users"] = adoption["users"] / len(feat) * 100
print(adoption[["users", "pct_of_users", "avg_days_active",
                 "avg_agent_tools", "avg_build_calls"]].round(2).to_string())

# B1. Adoption cohort comparison
metrics_to_compare = {
    "Avg active days"      : "avg_days_active",
    "Avg manual runs"  : "avg_manual_runs",
    "Avg agent tool calls" : "avg_agent_tools",
    "Avg agent build"      : "avg_build_calls",
    "Avg canvases"         : "avg_canvases",
    "Credit exceeded %"      : "pct_credit_exceeded",
}

fig_b1 = go.Figure()
for cohort_name in ADOPTION_ORDER:
    if cohort_name not in adoption.index:
        continue
    row_data = adoption.loc[cohort_name]
    # Normalize to the max
    vals = []
    for _, col in metrics_to_compare.items():
        col_max = adoption[col].max()
        vals.append(row_data[col] / col_max if col_max > 0 else 0)

    cats = list(metrics_to_compare.keys())
    fig_b1.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name=cohort_name,
        line_color=ADOPTION_COLORS.get(cohort_name, "#888"),
        opacity=0.75,
    ))

fig_b1.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Agent Adoption Cohort Profile<br><sup>Early adopters vs late vs never -- behavioral differences</sup>",
    template="plotly_dark",
    height=520,
)
write_html(fig_b1, f"{OUTPUT_DIR}/04_adoption_cohort_radar.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_cohort_radar.html")

# B2. Adoption cohort x signup cohort cross-tab
cross = (
    feat.groupby(["signup_cohort", "adoption_cohort"]).size()
    .unstack(fill_value=0)
    .reindex(columns=ADOPTION_ORDER)
)
cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

fig_b2 = go.Figure()
for ac in ADOPTION_ORDER:
    if ac not in cross_pct.columns:
        continue
    fig_b2.add_trace(go.Bar(
        x=cross_pct.index,
        y=cross_pct[ac],
        name=ac,
        marker_color=ADOPTION_COLORS.get(ac, "#888"),
    ))

fig_b2.update_layout(
    barmode="stack",
    title="Agent Adoption by Signup Cohort<br><sup>Did agent adoption improve in later cohorts?</sup>",
    yaxis_title="Share (%)",
    template="plotly_dark",
    height=420,
)
write_html(fig_b2, f"{OUTPUT_DIR}/04_adoption_by_signup_cohort.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_by_signup_cohort.html")

# B3. Bar comparison -- key metrics by adoption timing
fig_b3 = make_subplots(rows=1, cols=3,
    subplot_titles=["Avg active days", "Avg agent tools", "Avg canvases"])

for i, (metric, col) in enumerate([
    ("Avg active days", "avg_days_active"),
    ("Avg agent tools", "avg_agent_tools"),
    ("Avg canvases",    "avg_canvases"),
], start=1):
    fig_b3.add_trace(go.Bar(
        x=adoption.index,
        y=adoption[col].round(2),
        marker_color=[ADOPTION_COLORS.get(s, "#888") for s in adoption.index],
        showlegend=False,
        text=adoption[col].round(1),
        textposition="outside",
    ), row=1, col=i)

fig_b3.update_layout(
    title="Early vs Late vs Never Agent Adopter -- Key Metrics<br>"
          "<sup>Did early adopters become substantially more engaged users?</sup>",
    template="plotly_dark",
    height=400,
)
write_html(fig_b3, f"{OUTPUT_DIR}/04_adoption_metrics_comparison.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_metrics_comparison.html")

# -- SAVE -----------------------------------------------
feat.to_parquet(f"{OUTPUT_DIR}/user_features_segmented.parquet")
print(f"\n  Updated: {OUTPUT_DIR}/user_features_segmented.parquet (adoption_cohort added)")
print("\nSummary:")
print(f"  Signup cohort analysis: 3 charts")
print(f"  Agent adoption cohort: 3 charts")
print(f"  Important: the TWO analyses are orthogonal and intentionally separated!")

