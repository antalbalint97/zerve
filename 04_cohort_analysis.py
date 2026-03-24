import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 04_cohort_analysis.py  --  Kohorsz Elemzés
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/04_cohort_*.html

KET KULONALLO KOHORSZ ELEMZES:

  A) Signup hónap szerinti kohorsz
     -- Melyik hónapban regisztráltak?
     -- Hogyan fejlődött a platform user-bázisa?
     -- Melyik kohorsz lett a legelkötelezettebbé?

  B) Agent adoptáció időzítése szerinti kohorsz
     -- Korai adoptálók (<1h signup után)
     -- Késői adoptálók (1h+)
     -- Soha nem adoptálók
     Ez KULONALLO a signup kohorsz elemzéstől!
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
# A) SIGNUP HONAP KOHORSZ ELEMZES
# =======================================================
print("\n-- A) Signup Kohorsz Elemzés --")

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

# A1. Kohorsz méret és engagement
fig_a1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Regisztrálók száma kohorszonként",
        "Átlag aktív napok",
        "Agent adoptáció aránya (%)",
        "Átlag agent tool hívás / user",
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
    title="Signup Kohorsz Összehasonlítás<br><sup>Melyik hónapban regisztráltak a legelkötelezettebbek?</sup>",
    template="plotly_dark",
    height=600,
)
write_html(fig_a1, f"{OUTPUT_DIR}/04_cohort_signup_overview.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_signup_overview.html")

# A2. Szegmens összetétel kohorszonként
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
    title="Szegmens Összetétel Signup Kohorszonként (%)<br><sup>Javult-e a platformon a power user arány az idők során?</sup>",
    yaxis_title="Arány (%)",
    template="plotly_dark",
    height=450,
)
write_html(fig_a2, f"{OUTPUT_DIR}/04_cohort_segment_mix.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_segment_mix.html")

# A3. Engagement metrikák kohorszonként -- line chart
fig_a3 = make_subplots(specs=[[{"secondary_y": True}]])
fig_a3.add_trace(go.Scatter(
    x=x, y=cohort["avg_days_active"].round(2),
    mode="lines+markers", name="Átlag aktív napok",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=10),
), secondary_y=False)
fig_a3.add_trace(go.Scatter(
    x=x, y=cohort["pct_ever_agent"].round(1),
    mode="lines+markers", name="Agent adoptáció %",
    line=dict(color="#90e0ef", width=3, dash="dot"),
    marker=dict(size=10, symbol="diamond"),
), secondary_y=True)
fig_a3.update_layout(
    title="Engagement Trend Kohorszonként<br><sup>Tanul a platform? Javulnak a mutatók?</sup>",
    template="plotly_dark",
    height=400,
)
fig_a3.update_yaxes(title_text="Átlag aktív napok", secondary_y=False)
fig_a3.update_yaxes(title_text="Agent adoptáció %", secondary_y=True)
write_html(fig_a3, f"{OUTPUT_DIR}/04_cohort_engagement_trend.html")
print(f"  Saved: {OUTPUT_DIR}/04_cohort_engagement_trend.html")

# =======================================================
# B) AGENT ADOPTACIO KOHORSZ -- KULONALLO ELEMZES
# =======================================================
print("\n-- B) Agent Adoptáció Kohorsz Elemzés --")
print("   (Ez NEM a signup kohorsz -- ez a viselkedés alapján képzett csoport)")

feat["adoption_cohort"] = "Soha nem adoptálta"
feat.loc[feat["ever_used_agent"] == 1, "adoption_cohort"] = "Késői adoptáló (1h+)"
feat.loc[feat["adopted_agent_early"] == 1, "adoption_cohort"] = "Korai adoptáló (<1h)"

ADOPTION_COLORS = {
    "Korai adoptáló (<1h)" : "#00b4d8",
    "Késői adoptáló (1h+)" : "#90e0ef",
    "Soha nem adoptálta"   : "#444455",
}
ADOPTION_ORDER = ["Korai adoptáló (<1h)", "Késői adoptáló (1h+)", "Soha nem adoptálta"]

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

# B1. Adoptáció kohorsz összehasonlítás
metrics_to_compare = {
    "Átlag aktív napok"      : "avg_days_active",
    "Átlag manual futtatás"  : "avg_manual_runs",
    "Átlag agent tool hívás" : "avg_agent_tools",
    "Átlag agent build"      : "avg_build_calls",
    "Átlag canvasok"         : "avg_canvases",
    "Kredit túllépés %"      : "pct_credit_exceeded",
}

fig_b1 = go.Figure()
for cohort_name in ADOPTION_ORDER:
    if cohort_name not in adoption.index:
        continue
    row_data = adoption.loc[cohort_name]
    # Normalizálás a max-hoz
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
    title="Agent Adoptáció Kohorsz Profil<br><sup>Korai adoptálók vs késői vs soha -- viselkedési különbségek</sup>",
    template="plotly_dark",
    height=520,
)
write_html(fig_b1, f"{OUTPUT_DIR}/04_adoption_cohort_radar.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_cohort_radar.html")

# B2. Adoptáció kohorsz x signup kohorsz kereszttábla
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
    title="Agent Adoptáció Signup Kohorszonként<br><sup>Javult-e az agent adoptáció a later cohortokban?</sup>",
    yaxis_title="Arány (%)",
    template="plotly_dark",
    height=420,
)
write_html(fig_b2, f"{OUTPUT_DIR}/04_adoption_by_signup_cohort.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_by_signup_cohort.html")

# B3. Bar comparison -- kulcs metrikák adoptáció szerint
fig_b3 = make_subplots(rows=1, cols=3,
    subplot_titles=["Átlag aktív napok", "Átlag agent tools", "Átlag canvasok"])

for i, (metric, col) in enumerate([
    ("Átlag aktív napok", "avg_days_active"),
    ("Átlag agent tools", "avg_agent_tools"),
    ("Átlag canvasok",    "avg_canvases"),
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
    title="Korai vs Késői vs Soha Agent Adoptáló -- Kulcs Metrikák<br>"
          "<sup>A korai adoptálók lényegesen elkötelezettebb felhasználókká váltak?</sup>",
    template="plotly_dark",
    height=400,
)
write_html(fig_b3, f"{OUTPUT_DIR}/04_adoption_metrics_comparison.html")
print(f"  Saved: {OUTPUT_DIR}/04_adoption_metrics_comparison.html")

# -- MENTES -----------------------------------------------
feat.to_parquet(f"{OUTPUT_DIR}/user_features_segmented.parquet")
print(f"\n  Updated: {OUTPUT_DIR}/user_features_segmented.parquet (adoption_cohort hozzáadva)")
print("\nSummary:")
print(f"  Signup kohorsz elemzés: 3 chart")
print(f"  Agent adoptáció kohorsz: 3 chart")
print(f"  Fontos: a KET elemzes ortogonalis -- kulonvalasztas!")
