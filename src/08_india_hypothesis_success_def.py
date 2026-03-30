import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 08_india_hypothesis_success_def.py
 Zerve Hackathon 2026
=============================================================
Input : zerve_events.csv
        outputs/user_features_segmented.parquet
Output: outputs/08_*.html

KET ELEMZES:

A) India hipotezis validalas
   -- UTC 0-6h regisztrálók valóban indiaiak-e?
   -- India vs tobbi orszag Agent Builder arany
   -- UTC ora x orszag kereszttabla

B) Siker definicio argumentacio
   -- Agent Builder szegmens belso konzisztenciaja
   -- Miert nem onkényes a hatar (>= 3 build call)
   -- Szegemns szeparabilitas vizualizacio
   -- Uzleti ertek: kredit tullépés, multi-day retention
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features, load_raw_events
from analytics.viz import write_html
warnings.filterwarnings("ignore")

DATA_PATH  = "data/zerve_events.csv"
FEAT_PATH  = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(FEAT_PATH)
df = load_raw_events(DATA_PATH)

print(f"  {len(feat):,} users  |  {len(df):,} events")

SEGMENT_COLORS = {
    "Agent Builder" : "#00b4d8",
    "Manual Coder"  : "#90e0ef",
    "Viewer"        : "#555566",
    "Ghost"         : "#2d2d3a",
}

# ============================================================
# A) INDIA HIPOTEZIS VALIDALAS
# ============================================================
print("\n-- A) India Hipotezis Validalas --")

# Signup eventek orszag adattal
signup_df = df[
    (df["event"] == "sign_up") &
    df["prop_$geoip_country_code"].notna()
].copy()
signup_df["signup_hour_utc"] = signup_df["timestamp"].dt.hour

print(f"  Signup eventek orszag adattal: {len(signup_df):,}")

# UTC 0-6h regisztrálók orszag eloszlasa
night_signups = signup_df[signup_df["signup_hour_utc"].between(0, 5)]
day_signups   = signup_df[signup_df["signup_hour_utc"].between(9, 17)]

print(f"\n  UTC 0-6h regisztrálók (n={len(night_signups):,}):")
night_countries = night_signups["prop_$geoip_country_code"].value_counts().head(8)
for country, n in night_countries.items():
    pct = n / len(night_signups) * 100
    print(f"    {country}: {n:>4,}  ({pct:.1f}%)")

print(f"\n  UTC 9-17h regisztrálók (n={len(day_signups):,}):")
day_countries = day_signups["prop_$geoip_country_code"].value_counts().head(8)
for country, n in day_countries.items():
    pct = n / len(day_signups) * 100
    print(f"    {country}: {n:>4,}  ({pct:.1f}%)")

# Chart A1: UTC ora x Top 5 orszag heatmap
top_countries = signup_df["prop_$geoip_country_code"].value_counts().head(6).index.tolist()
hour_country = (
    signup_df[signup_df["prop_$geoip_country_code"].isin(top_countries)]
    .groupby(["signup_hour_utc", "prop_$geoip_country_code"])
    .size()
    .unstack(fill_value=0)
)
# Normalizalas: soronkent (minden oran belul az orszag arany)
hour_country_pct = hour_country.div(hour_country.sum(axis=1), axis=0) * 100

fig_a1 = px.imshow(
    hour_country_pct.T,
    color_continuous_scale="Blues",
    title="Melyik orszagbol jonnek a regisztrálók oranként? (%)<br>"
          "<sup>India UTC 0-6h-ban dominans -- ez India reggeli munkaideje (5:30-11:30 IST)</sup>",
    labels=dict(x="UTC Ora", y="Orszag", color="Arany %"),
    template="plotly_dark",
    aspect="auto",
)
write_html(fig_a1, f"{OUTPUT_DIR}/08_country_hour_heatmap.html")
print(f"\n  Saved: {OUTPUT_DIR}/08_country_hour_heatmap.html")

# Chart A2: India vs tobbi - Agent Builder arany osszehasonlitas
feat_with_country = feat.copy()

# Orszag az elso signup eventbol
first_country = (
    df[df["event"] == "sign_up"]
    .dropna(subset=["prop_$geoip_country_code"])
    .sort_values("timestamp")
    .groupby("person_id")["prop_$geoip_country_code"]
    .first()
    .rename("country")
)
feat_with_country = feat_with_country.join(first_country)
feat_with_country["country_group"] = feat_with_country["country"].apply(
    lambda x: x if x in ["IN", "US", "GB", "IE", "FR", "NL"] else "Other"
    if pd.notna(x) else "Unknown"
)

country_stats = feat_with_country.groupby("country_group").agg(
    users     = ("segment", "count"),
    pct_ab    = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
    pct_ghost = ("segment", lambda x: (x == "Ghost").mean() * 100),
    pct_agent = ("ever_used_agent", "mean"),
).round(2)
country_stats["pct_agent"] *= 100
country_stats = country_stats.sort_values("pct_ab", ascending=False)

print("\n  Orszagonkent Agent Builder %:")
print(country_stats[["users", "pct_ab", "pct_ghost", "pct_agent"]].to_string())

fig_a2 = make_subplots(rows=1, cols=2,
    subplot_titles=["Agent Builder % orszagonkent", "Agent adoptacio % orszagonkent"])

colors_map = {"IN": "#00b4d8", "IE": "#48cae4", "FR": "#90e0ef",
              "US": "#555566", "GB": "#444455", "NL": "#333344", "Other": "#222233"}
bar_colors = [colors_map.get(c, "#333") for c in country_stats.index]

fig_a2.add_trace(go.Bar(
    x=country_stats.index, y=country_stats["pct_ab"].round(1),
    marker_color=bar_colors, text=country_stats["pct_ab"].round(1),
    texttemplate="%{text}%", showlegend=False,
), row=1, col=1)
fig_a2.add_trace(go.Bar(
    x=country_stats.index, y=country_stats["pct_agent"].round(1),
    marker_color=bar_colors, text=country_stats["pct_agent"].round(1),
    texttemplate="%{text}%", showlegend=False,
), row=1, col=2)

fig_a2.update_layout(
    title="India Hipotezis -- Orszagonkent Agent Builder es adoptacio arany<br>"
          "<sup>India a legnagyobb user bazis es vezeto Agent Builder arannyal?</sup>",
    template="plotly_dark", height=420,
)
write_html(fig_a2, f"{OUTPUT_DIR}/08_country_ab_comparison.html")
print(f"  Saved: {OUTPUT_DIR}/08_country_ab_comparison.html")

# ============================================================
# B) SIKER DEFINICIO ARGUMENTACIO
# ============================================================
print("\n-- B) Siker Definicio Argumentacio --")

# B1. Miert Agent Builder = siker?
# Bizonyitas: belso konzisztencia + uzleti ertek

seg_stats = feat.groupby("segment").agg(
    users                  = ("total_events", "count"),
    avg_days               = ("days_active", "mean"),
    median_days            = ("days_active", "median"),
    avg_agent_tools        = ("agent_tool_calls_total", "mean"),
    avg_build_calls        = ("agent_build_calls", "mean"),
    avg_tool_types         = ("agent_tool_types_used", "mean"),
    pct_multi_day          = ("days_active", lambda x: (x >= 3).mean() * 100),
    pct_credit_exceeded    = ("had_credit_exceeded", "mean"),
    pct_returned           = ("had_second_session", "mean"),
    avg_first_sess_events  = ("first_session_events", "mean"),
).round(2)
seg_stats["pct_credit_exceeded"] *= 100
seg_stats["pct_returned"]        *= 100

print("\n  Szegmens profil osszehasonlitas:")
print(seg_stats.to_string())

# B2. Agent Builder hatar szenzitivitas elemzes
# Van-e ugras a >= 3 build call hatarnal?
print("\n  Build call kuszob szenzitivitas:")
build_thresholds = range(1, 10)
for t in build_thresholds:
    above = feat[feat["agent_build_calls"] >= t]
    below = feat[(feat["agent_build_calls"] < t) & (feat["agent_build_calls"] > 0)]
    if len(above) == 0 or len(below) == 0:
        continue
    print(f"    >= {t} build call: {len(above):>4,} user, "
          f"avg_days={above['days_active'].mean():.1f}, "
          f"avg_tool_calls={above['agent_tool_calls_total'].mean():.0f}")

# B3. Szegmens szeparabilitas -- 2D scatter
# first_session_events vs time_to_return_hours, szin = szegmens
plot_feat = feat[feat["segment"].isin(["Agent Builder", "Manual Coder", "Viewer"])].copy()
plot_feat["time_to_return_clip"] = plot_feat["time_to_return_hours"].clip(0, 200)
plot_feat["first_sess_clip"]     = plot_feat["first_session_events"].clip(0, 50)

fig_b1 = px.scatter(
    plot_feat,
    x="first_sess_clip",
    y="time_to_return_clip",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    opacity=0.6,
    title="Szegmens Szeparabilitas -- Elso Session vs Visszateres<br>"
          "<sup>Agent Builderek elkulonulnek a tobbi felhasznalotol</sup>",
    labels={
        "first_sess_clip"    : "Elso session eventek (max 50)",
        "time_to_return_clip": "Visszateres ideje oraban (max 200)",
    },
    template="plotly_dark",
    height=500,
)
write_html(fig_b1, f"{OUTPUT_DIR}/08_segment_separability.html")
print(f"\n  Saved: {OUTPUT_DIR}/08_segment_separability.html")

# B4. Uzleti ertek argumentacio
# Agent Builderek kredit tullépés aranya = upgrade potential
print("\n  Uzleti ertek -- Agent Builder kredit nyomas:")
ab = feat[feat["segment"] == "Agent Builder"]
print(f"    Agent Builder kredit tullépés: {ab['had_credit_exceeded'].mean()*100:.1f}%")
print(f"    Agent Builder addon kredit:    {ab['had_addon_credits'].mean()*100:.1f}%")
print(f"    Ghost kredit tullépés:         {feat[feat['segment']=='Ghost']['had_credit_exceeded'].mean()*100:.1f}%")
print(f"    --> Agent Builderek {ab['had_credit_exceeded'].mean() / max(feat[feat['segment']=='Ghost']['had_credit_exceeded'].mean(), 0.001):.0f}x valoszinubb hogy kreditproblémajuk van")
print(f"    --> Ok a legfontosabb upgrade candidates")

# B5. Siker definicio chart -- miert nem lehetett mas
# Alternativ definiciok osszehasonlitasa
alt_defs = {
    "Agent Builder\n(>= 3 build calls)"       : (feat["agent_build_calls"] >= 3).mean() * 100,
    "Barmilyen agent\nhasznalat"               : (feat["ever_used_agent"] == 1).mean() * 100,
    "5+ aktiv nap"                             : (feat["days_active"] >= 5).mean() * 100,
    "Manual futtatott\n(>= 3x)"               : (feat["manual_runs"] >= 3).mean() * 100,
    "Visszatert\n(masodik session)"            : (feat["had_second_session"] == 1).mean() * 100,
}

# Mindegyik definiciohoz: avg days_active es avg agent tools
alt_stats = []
for label, pct in alt_defs.items():
    mask = eval({
        "Agent Builder\n(>= 3 build calls)"   : "feat['agent_build_calls'] >= 3",
        "Barmilyen agent\nhasznalat"           : "feat['ever_used_agent'] == 1",
        "5+ aktiv nap"                         : "feat['days_active'] >= 5",
        "Manual futtatott\n(>= 3x)"           : "feat['manual_runs'] >= 3",
        "Visszatert\n(masodik session)"        : "feat['had_second_session'] == 1",
    }[label])
    alt_stats.append({
        "definicio"   : label,
        "pct_of_users": round(pct, 1),
        "avg_days"    : round(feat[mask]["days_active"].mean(), 2),
        "avg_tools"   : round(feat[mask]["agent_tool_calls_total"].mean(), 1),
        "pct_credit"  : round(feat[mask]["had_credit_exceeded"].mean() * 100, 1),
    })

alt_df = pd.DataFrame(alt_stats)
print("\n  Alternativ siker definiciok osszehasonlitasa:")
print(alt_df.to_string(index=False))

fig_b2 = make_subplots(rows=1, cols=3,
    subplot_titles=["% of users (alacsonyabb = specifikusabb)",
                    "Avg aktiv napok", "Kredit tullépés %"])

x_labels = [d.replace("\n", " ") for d in alt_df["definicio"]]
for i, (col, title) in enumerate([
    ("pct_of_users", "% users"),
    ("avg_days", "Avg days"),
    ("pct_credit", "Credit %"),
], start=1):
    fig_b2.add_trace(go.Bar(
        x=x_labels,
        y=alt_df[col],
        marker_color=["#00b4d8" if "Builder" in x else "#555566" for x in x_labels],
        showlegend=False,
        text=alt_df[col],
        texttemplate="%{text}",
        textposition="outside",
    ), row=1, col=i)

fig_b2.update_layout(
    title="Miert az Agent Builder a legjobb siker definicio?<br>"
          "<sup>Alternativ definiciok osszehasonlitasa -- specifikus es uzletileg ertelmes</sup>",
    template="plotly_dark",
    height=480,
)
fig_b2.update_xaxes(tickangle=20)
write_html(fig_b2, f"{OUTPUT_DIR}/08_success_def_comparison.html")
print(f"  Saved: {OUTPUT_DIR}/08_success_def_comparison.html")

print("\n[OK] India hipotezis + Siker definicio argumentacio complete.")
