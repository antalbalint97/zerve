import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 05_lifecycle_analysis.py  --  Életút & Konverzió Elemzés
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/05_lifecycle_*.html
        outputs/05_conversion_*.html

Két elemzés:

  A) Életút -- mikor lépett be az agent használatba?
     Mennyi idő telt el signup és első agent tool közt?
     Milyen gyorsan "mélyülnek el" a userek?

  B) Konverzió funnel -- szegmensenként és kohorszonként
     Hány user jutott el sign_up -> run -> agent -> build?
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

print("Loading data...")
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
# A) ELETUT ELEMZES
# =======================================================
print("\n-- A) Életút Elemzés --")

# A1. Time-to-first-action eloszlás szegmensenként
active = feat[feat["segment"].isin(["Agent Builder", "Agent Runner", "Manual Coder"])].copy()

fig_a1 = go.Figure()
for seg in ["Agent Builder", "Agent Runner", "Manual Coder"]:
    sub = active[active["segment"] == seg]

    # Manual run TTF
    ttf_manual = sub["ttf_manual_run_min"].replace(0, np.nan).dropna()
    if len(ttf_manual) > 5:
        fig_a1.add_trace(go.Box(
            y=ttf_manual.clip(upper=1440),  # max 24h
            name=f"{seg} -- manual run",
            marker_color=SEGMENT_COLORS.get(seg, "#888"),
            boxmean=True,
            showlegend=True,
        ))

fig_a1.update_layout(
    title="Mikor futtatták az első kódot? (perc signup után)<br>"
          "<sup>Max 1440 perc (24h) -- csak aktív userek</sup>",
    yaxis_title="Perc signup után",
    template="plotly_dark",
    height=450,
)
write_html(fig_a1, f"{OUTPUT_DIR}/05_ttf_distribution.html")
print(f"  Saved: {OUTPUT_DIR}/05_ttf_distribution.html")

# A2. Agent adoptáció időzítése
agent_users = feat[feat["ever_used_agent"] == 1].copy()
ttf_agent = agent_users["ttf_agent_tool_min"].replace(0, np.nan).dropna()

bins   = [0, 15, 60, 240, 1440, 10_080]
labels = ["<15 perc", "15-60 perc", "1-4 óra", "4-24 óra", "1-7 nap"]
agent_users["agent_adoption_bucket"] = pd.cut(
    agent_users["ttf_agent_tool_min"], bins=bins, labels=labels, right=False
)

adoption_timing = agent_users["agent_adoption_bucket"].value_counts().sort_index()
adoption_timing_seg = (
    agent_users.groupby(["agent_adoption_bucket", "segment"]).size()
    .unstack(fill_value=0)
    .reindex(columns=[s for s in SEG_ORDER if s in agent_users["segment"].unique()])
)

fig_a2 = go.Figure()
for seg in [s for s in SEG_ORDER if s in adoption_timing_seg.columns]:
    fig_a2.add_trace(go.Bar(
        x=adoption_timing_seg.index.astype(str),
        y=adoption_timing_seg[seg],
        name=seg,
        marker_color=SEGMENT_COLORS.get(seg, "#888"),
    ))

fig_a2.update_layout(
    barmode="stack",
    title="Mikor adoptálták az agentet? (signup után)<br>"
          "<sup>Csak az agent-et valaha használó userek</sup>",
    xaxis_title="Idő az első agent tool hívásig",
    yaxis_title="Felhasználók száma",
    template="plotly_dark",
    height=420,
)
write_html(fig_a2, f"{OUTPUT_DIR}/05_agent_adoption_timing.html")
print(f"  Saved: {OUTPUT_DIR}/05_agent_adoption_timing.html")

# A3. Aktív napok eloszlása szegmensenként (tenure)
fig_a3 = go.Figure()
for seg in ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer"]:
    sub = feat[feat["segment"] == seg]["days_active"]
    if len(sub) < 3:
        continue
    fig_a3.add_trace(go.Violin(
        y=sub.clip(upper=30),
        name=seg,
        fillcolor=SEGMENT_COLORS.get(seg, "#888"),
        line_color="white",
        opacity=0.7,
        box_visible=True,
        meanline_visible=True,
    ))

fig_a3.update_layout(
    title="Aktív napok eloszlása szegmensenként<br><sup>Max 30 nap -- ki jön vissza rendszeresen?</sup>",
    yaxis_title="Aktív napok száma",
    template="plotly_dark",
    height=450,
)
write_html(fig_a3, f"{OUTPUT_DIR}/05_active_days_distribution.html")
print(f"  Saved: {OUTPUT_DIR}/05_active_days_distribution.html")

# =======================================================
# B) KONVERZIO FUNNEL ELEMZES
# =======================================================
print("\n-- B) Konverzió Funnel Elemzés --")

# Funnel lépések a feature mátrixból
FUNNEL_STEPS = [
    ("Regisztrált",          feat["signed_up"] >= 1),
    ("Manuálisan futtatott", feat["manual_runs"] >= 1),
    ("Agent chat indított",  feat["agent_conversations"] >= 1),
    ("Agent tool használt",  feat["agent_tool_calls_total"] >= 1),
    ("Agent épített",        feat["agent_build_calls"] >= 1),
    ("5+ aktív nap",         feat["days_active"] >= 5),
]

# Globális funnel
funnel_data = []
for label, condition in FUNNEL_STEPS:
    n = condition.sum()
    funnel_data.append({"step": label, "users": int(n)})

funnel_df = pd.DataFrame(funnel_data)
fig_b1 = go.Figure(go.Funnel(
    y=funnel_df["step"],
    x=funnel_df["users"],
    textinfo="value+percent initial",
    marker=dict(color=[
        "#023e8a", "#0077b6", "#0096c7",
        "#00b4d8", "#48cae4", "#90e0ef",
    ]),
))
fig_b1.update_layout(
    title="Globális Konverzió Funnel<br><sup>Sign-up-tól az elkötelezett Agent Builder-ig</sup>",
    template="plotly_dark",
    height=500,
)
write_html(fig_b1, f"{OUTPUT_DIR}/05_global_funnel.html")
print(f"  Saved: {OUTPUT_DIR}/05_global_funnel.html")
print(f"\n  Funnel:")
for _, row in funnel_df.iterrows():
    pct = row["users"] / funnel_df.iloc[0]["users"] * 100
    print(f"    {row['step']:<30} {row['users']:>5,}  ({pct:.1f}%)")

# Funnel kohorszonként
print("\n  Funnel signup kohorszonként:")
cohort_funnel = {}
for cohort_name, group in feat.groupby("signup_cohort"):
    steps = {}
    for label, _ in FUNNEL_STEPS:
        # Újraértékeljük a feltételt a csoportra
        pass

    # Egyszerűbb: aggregált metrikák
    cohort_funnel[cohort_name] = {
        "Regisztrált"         : len(group),
        "Manuálisan futtatott": (group["manual_runs"] >= 1).sum(),
        "Agent tool használt" : (group["agent_tool_calls_total"] >= 1).sum(),
        "Agent épített"       : (group["agent_build_calls"] >= 1).sum(),
        "5+ aktív nap"        : (group["days_active"] >= 5).sum(),
    }

cohort_funnel_df = pd.DataFrame(cohort_funnel).T
# Konverzió % az első lépéshez képest
for col in cohort_funnel_df.columns[1:]:
    cohort_funnel_df[col + " %"] = (
        cohort_funnel_df[col] / cohort_funnel_df["Regisztrált"] * 100
    ).round(1)

print(cohort_funnel_df[[c for c in cohort_funnel_df.columns if "%" in c]].to_string())

pct_cols = [c for c in cohort_funnel_df.columns if "%" in c]
fig_b2 = go.Figure()
for col in pct_cols:
    fig_b2.add_trace(go.Bar(
        x=cohort_funnel_df.index,
        y=cohort_funnel_df[col],
        name=col.replace(" %", ""),
    ))

fig_b2.update_layout(
    barmode="group",
    title="Konverzió Funnel Signup Kohorszonként (%)<br>"
          "<sup>Javul a platform konverziója az újabb kohorszoknál?</sup>",
    yaxis_title="Konverzió % (regisztráltakhoz képest)",
    template="plotly_dark",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
write_html(fig_b2, f"{OUTPUT_DIR}/05_cohort_funnel.html")
print(f"  Saved: {OUTPUT_DIR}/05_cohort_funnel.html")

# B3. Drop-off elemzés -- hol vesznek el a userek?
fig_b3 = make_subplots(rows=1, cols=1)
prev = funnel_df["users"].iloc[0]
dropoff_data = []
for i, row in funnel_df.iterrows():
    if i == 0:
        prev = row["users"]
        continue
    dropped = prev - row["users"]
    dropoff_data.append({
        "atlepett": f"{funnel_df.iloc[i-1]['step']} -> {row['step']}",
        "elveszett": dropped,
        "pct": dropped / funnel_df.iloc[0]["users"] * 100,
    })
    prev = row["users"]

dropoff_df = pd.DataFrame(dropoff_data)
fig_b3 = px.bar(
    dropoff_df,
    x="pct", y="atlepett",
    orientation="h",
    title="Hol vesznek el a felhasználók? (Drop-off %)<br>"
          "<sup>A teljes regisztrált bázishoz képest</sup>",
    template="plotly_dark",
    color="pct",
    color_continuous_scale="Reds",
    labels={"pct": "Elveszett userek %", "atlepett": ""},
    text="elveszett",
)
fig_b3.update_traces(texttemplate="%{text:,} user", textposition="outside")
fig_b3.update_layout(height=400, yaxis=dict(autorange="reversed"))
write_html(fig_b3, f"{OUTPUT_DIR}/05_dropoff_analysis.html")
print(f"  Saved: {OUTPUT_DIR}/05_dropoff_analysis.html")

print("\nLifecycle & Conversion analysis complete.")
print(f"  Életút: {OUTPUT_DIR}/05_ttf_distribution.html")
print(f"  Konverzió funnel: {OUTPUT_DIR}/05_global_funnel.html")
print(f"  Drop-off: {OUTPUT_DIR}/05_dropoff_analysis.html")
