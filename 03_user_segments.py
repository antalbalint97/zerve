import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 03_user_segments.py  --  Viselkedési Szegmentáció
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features.parquet
Output: outputs/user_features_segmented.parquet
        outputs/03_segments_*.html

4 szegmens a valós Zerve használat alapján:

  Agent Builder   -- az agent írja/refaktorálja a workflow-t
  Agent Runner    -- az agent futtatja a kódot, de manuálisan is dolgozik
  Manual Coder    -- hagyományos notebook felhasználó, kevés agent
  Viewer/Ghost    -- nézelődik vagy szinte nem használja
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

# -- RULE-BASED SZEGMENTACIO -------------------------------
# Elsősorban szabály-alapú, mert a klaszterezés a 95%-os ghost csoportra
# konvergálna -- a szabályok Zerve-specifikusak és interpretálhatók

def assign_segment(row):
    has_agent      = row["ever_used_agent"] >= 1
    build_calls    = row.get("agent_build_calls", 0)
    run_calls      = row.get("agent_run_calls", 0)
    manual_runs    = row["manual_runs"]
    days           = row["days_active"]
    total          = row["total_events"]
    conversations  = row.get("agent_conversations", 0)

    # Ghost: szinte semmi aktivitás
    if total <= 5 or days == 0:
        return "Ghost"

    # Agent Builder: az agent főleg épít/refaktorál
    if has_agent and build_calls >= 3:
        return "Agent Builder"

    # Agent Runner: az agent futtat, de manuális is van
    if has_agent and (run_calls >= 3 or conversations >= 2):
        return "Agent Runner"

    # Manual Coder: fut kódot de agent nélkül vagy minimális agenttel
    if manual_runs >= 3:
        return "Manual Coder"

    # Viewer/Explorer: nézelődik, fullscreen, de nem futtat sokat
    return "Viewer"

feat["segment"] = feat.apply(assign_segment, axis=1)

# -- SZEGMENS STATISZTIKAK --------------------------------
print("\n-- Szegmens eloszlas --")
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

# Rendezés engagement szerint
seg_order = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]
seg_stats = seg_stats.reindex([s for s in seg_order if s in seg_stats.index])

print(seg_stats[["users", "pct_of_users", "avg_days_active",
                  "avg_manual_runs", "avg_agent_tools", "avg_build_calls"]].to_string())

# Szegmens sorrend és szín mentése
SEGMENT_COLORS = {
    "Agent Builder" : "#00b4d8",
    "Agent Runner"  : "#48cae4",
    "Manual Coder"  : "#90e0ef",
    "Viewer"        : "#555566",
    "Ghost"         : "#2d2d3a",
}

# -- VIZUALIZACIOK -----------------------------------------

# 1. Szegmens méret pie
fig1 = go.Figure(go.Pie(
    labels=seg_stats.index,
    values=seg_stats["users"],
    hole=0.45,
    marker_colors=[SEGMENT_COLORS.get(s, "#888") for s in seg_stats.index],
    textinfo="label+percent",
    textfont_size=12,
))
fig1.update_layout(
    title="Zerve User Segments<br><sup>Viselkedés-alapú szegmentáció</sup>",
    template="plotly_dark",
    height=450,
)
write_html(fig1, f"{OUTPUT_DIR}/03_segment_sizes.html")
print(f"\n  Saved: {OUTPUT_DIR}/03_segment_sizes.html")

# 2. Szegmens profil radar
radar_metrics = {
    "Napok aktív"      : "avg_days_active",
    "Manuális futtatás": "avg_manual_runs",
    "Agent tool hívás" : "avg_agent_tools",
    "Agent build"      : "avg_build_calls",
    "Agent run"        : "avg_run_calls",
    "Canvasok"         : "avg_canvases",
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
    title="Szegmens Viselkedési Profil<br><sup>Normalizált értékek</sup>",
    template="plotly_dark",
    height=500,
)
write_html(fig2, f"{OUTPUT_DIR}/03_segment_radar.html")
print(f"  Saved: {OUTPUT_DIR}/03_segment_radar.html")

# 3. Agent tool mix szegmensenként
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
        title="Agent Tool Használat Szegmensenként<br><sup>Átlag tool call / user</sup>",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Teal,
        labels={"value": "Avg tool calls", "variable": "Tool", "segment": ""},
    )
    write_html(fig3, f"{OUTPUT_DIR}/03_segment_tool_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/03_segment_tool_mix.html")

# 4. Signup kohorsz x szegmens heatmap
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
    title="Szegmens Arányok Signup Kohorszonként (%)<br><sup>Melyik kohorsz termelte a legtöbb power usert?</sup>",
    template="plotly_dark",
    labels=dict(x="Signup kohorsz", y="Szegmens", color="%"),
    aspect="auto",
)
write_html(fig4, f"{OUTPUT_DIR}/03_cohort_segment_heatmap.html")
print(f"  Saved: {OUTPUT_DIR}/03_cohort_segment_heatmap.html")

# 5. Agent adoption: early vs late
if "ttf_agent_tool_min" in feat.columns:
    non_ghost = feat[feat["segment"] != "Ghost"].copy()
    non_ghost["agent_adoption"] = "Nem használja"
    non_ghost.loc[non_ghost["ever_used_agent"] == 1, "agent_adoption"] = "Késői adoptáló"
    non_ghost.loc[non_ghost["adopted_agent_early"] == 1, "agent_adoption"] = "Korai adoptáló (<1h)"

    adoption_seg = (
        non_ghost.groupby(["segment", "agent_adoption"])
        .size().unstack(fill_value=0)
    )
    fig5 = px.bar(
        adoption_seg.reset_index().melt(id_vars="segment"),
        x="segment", y="value", color="agent_adoption",
        barmode="group",
        title="Agent Adoptáció Szegmensenként<br><sup>Korai vs késői vs nem adoptáló</sup>",
        template="plotly_dark",
        color_discrete_map={
            "Korai adoptáló (<1h)": "#00b4d8",
            "Késői adoptáló"      : "#90e0ef",
            "Nem használja"       : "#555566",
        },
        labels={"value": "Users", "segment": "", "agent_adoption": ""},
    )
    write_html(fig5, f"{OUTPUT_DIR}/03_agent_adoption_by_segment.html")
    print(f"  Saved: {OUTPUT_DIR}/03_agent_adoption_by_segment.html")

# -- MENTES -----------------------------------------------
feat.to_parquet(f"{OUTPUT_DIR}/user_features_segmented.parquet")
feat.to_csv(f"{OUTPUT_DIR}/user_features_segmented.csv")

print(f"\n  Saved: {OUTPUT_DIR}/user_features_segmented.parquet")
print(f"\n  Szegmens összefoglaló:")
for seg in seg_order:
    if seg in seg_stats.index:
        row = seg_stats.loc[seg]
        print(f"    {seg:<20} {int(row['users']):>5,} user  ({row['pct_of_users']:.1f}%)")
