import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 07_signup_hour_survival.py
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/07_*.html

KET ELEMZES:

A) Signup Hour Analysis
   -- Melyik orasavban regisztralok a legsikeresebbeek?
   -- Munkaido vs ejtszaka vs hejtvege
   -- India timezone vs global

B) Survival Analysis (Kaplan-Meier)
   -- Meddig marad aktiv egy user?
   -- Van-e kritikus kuszob a visszateresnel?
   -- Szegmensenként és adoptáció szerint
   -- time_to_return_hours: 24h alatt visszajovok vs kesobb
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features
from analytics.viz import write_html
warnings.filterwarnings("ignore")

INPUT_PATH = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

SEGMENT_COLORS = {
    "Agent Builder" : "#00b4d8",
    "Manual Coder"  : "#90e0ef",
    "Viewer"        : "#555566",
    "Ghost"         : "#2d2d3a",
}

# ============================================================
# A) SIGNUP HOUR ELEMZES
# ============================================================
print("\n-- A) Signup Hour Elemzes --")

# Csak userek akiknel van signup_hour adat (-1 = nincs)
has_hour = feat[feat["signup_hour"] >= 0].copy()
print(f"  Userek signup_hour adattal: {len(has_hour):,}")

# Agent Builder arany orankennt
hour_stats = has_hour.groupby("signup_hour").agg(
    users           = ("segment", "count"),
    agent_builders  = ("segment", lambda x: (x == "Agent Builder").sum()),
    pct_ab          = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
    pct_ghost       = ("segment", lambda x: (x == "Ghost").mean() * 100),
    pct_agent_user  = ("ever_used_agent", "mean"),
    avg_first_sess  = ("first_session_events", "mean"),
).reset_index()
hour_stats["pct_agent_user"] *= 100

print("\n  Oranként Agent Builder % (top 5 ora):")
print(hour_stats.nlargest(5, "pct_ab")[
    ["signup_hour", "users", "pct_ab", "pct_ghost"]
].to_string(index=False))

# Orasav kategorizalas
def hour_category(h):
    if 9 <= h <= 17:  return "Munkaido (9-17h)"
    if 18 <= h <= 22: return "Este (18-22h)"
    if 0 <= h <= 6:   return "Ejszaka (0-6h)"
    return "Hajnal/reggel (7-8h, 23h)"

hour_stats["time_category"] = hour_stats["signup_hour"].apply(hour_category)

# Chart 1: Hourly heatmap -- AB % es ghost %
fig_a1 = make_subplots(
    rows=2, cols=1,
    subplot_titles=[
        "Agent Builder % oranként (UTC -- India +5:30, US -5/-8)",
        "Ghost % oranként",
    ],
    vertical_spacing=0.15,
)
fig_a1.add_trace(go.Bar(
    x=hour_stats["signup_hour"],
    y=hour_stats["pct_ab"].round(1),
    marker_color="#00b4d8",
    name="Agent Builder %",
    text=hour_stats["pct_ab"].round(1),
    texttemplate="%{text}%",
    textposition="outside",
), row=1, col=1)
fig_a1.add_trace(go.Bar(
    x=hour_stats["signup_hour"],
    y=hour_stats["pct_ghost"].round(1),
    marker_color="#ff6b6b",
    name="Ghost %",
    text=hour_stats["pct_ghost"].round(1),
    texttemplate="%{text}%",
    textposition="outside",
), row=2, col=1)
fig_a1.update_xaxes(tickmode="linear", tick0=0, dtick=1, title_text="Ora (UTC)")
fig_a1.update_layout(
    title="Mikor regisztrálnak a sikeres felhasználók?<br>"
          "<sup>Agent Builder es Ghost arany signup oras UTC szerint</sup>",
    template="plotly_dark",
    height=600,
    showlegend=False,
)
write_html(fig_a1, f"{OUTPUT_DIR}/07_signup_hour_ab_pct.html")
print(f"  Saved: {OUTPUT_DIR}/07_signup_hour_ab_pct.html")

# Chart 2: Orasav kategoria összehasonlitas
has_hour["time_category"] = has_hour["signup_hour"].apply(hour_category)

cat_stats = has_hour.groupby("time_category").agg(
    users          = ("segment", "count"),
    pct_ab         = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
    pct_ghost      = ("segment", lambda x: (x == "Ghost").mean() * 100),
    avg_first_sess = ("first_session_events", "mean"),
    pct_returned   = ("had_second_session", "mean"),
).reset_index()
cat_stats["pct_returned"] *= 100

print("\n  Orasav kategoria összehasonlitas:")
print(cat_stats[["time_category", "users", "pct_ab", "pct_ghost", "pct_returned"]].to_string(index=False))

fig_a2 = make_subplots(rows=1, cols=3,
    subplot_titles=["Agent Builder %", "Ghost %", "Visszateres %"])

cat_order = ["Munkaido (9-17h)", "Este (18-22h)", "Hajnal/reggel (7-8h, 23h)", "Ejszaka (0-6h)"]
cat_stats_ordered = cat_stats.set_index("time_category").reindex(
    [c for c in cat_order if c in cat_stats["time_category"].values]
).reset_index()

colors = ["#00b4d8", "#48cae4", "#90e0ef", "#2d2d3a"]
for i, (metric, title) in enumerate([
    ("pct_ab", "AB %"),
    ("pct_ghost", "Ghost %"),
    ("pct_returned", "Visszateres %"),
], start=1):
    fig_a2.add_trace(go.Bar(
        x=cat_stats_ordered["time_category"],
        y=cat_stats_ordered[metric].round(1),
        marker_color=colors[:len(cat_stats_ordered)],
        text=cat_stats_ordered[metric].round(1),
        texttemplate="%{text}%",
        textposition="outside",
        showlegend=False,
    ), row=1, col=i)

fig_a2.update_layout(
    title="Munkaidős vs esti vs ejszakai regisztrálók<br>"
          "<sup>Melyik napszakban regisztrálnak az elkötelezett userek?</sup>",
    template="plotly_dark",
    height=420,
)
fig_a2.update_xaxes(tickangle=15)
write_html(fig_a2, f"{OUTPUT_DIR}/07_signup_hour_categories.html")
print(f"  Saved: {OUTPUT_DIR}/07_signup_hour_categories.html")

# Chart 3: Hetvege vs hétköznap
if "signup_is_weekend" in feat.columns:
    wd_stats = feat.groupby("signup_is_weekend").agg(
        users     = ("segment", "count"),
        pct_ab    = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
        pct_ghost = ("segment", lambda x: (x == "Ghost").mean() * 100),
        avg_first = ("first_session_events", "mean"),
        pct_ret   = ("had_second_session", "mean"),
    ).reset_index()
    wd_stats["pct_ret"] *= 100
    wd_stats["signup_is_weekend"] = wd_stats["signup_is_weekend"].map(
        {0: "Hétköznap", 1: "Hétvége"})
    print("\n  Hétköznap vs Hétvége:")
    print(wd_stats.to_string(index=False))

# ============================================================
# B) SURVIVAL ANALYSIS (Kaplan-Meier)
# ============================================================
print("\n-- B) Survival Analysis --")

# Survival definicio:
# - "ido" = days_active (hány különböző napon volt aktív)
# - "event" = 1 ha volt legalabb 2 nap aktiv (= "nem churned azonnal")
# - Megkerdés: ki marad aktív legalabb N napon?

feat["survival_time"]  = feat["days_active"].clip(lower=1)
feat["had_engagement"] = (feat["days_active"] >= 2).astype(int)

print(f"  Users legalabb 2 aktiv nappal: {feat['had_engagement'].sum():,} "
      f"({feat['had_engagement'].mean()*100:.1f}%)")

# -- B1. Kaplan-Meier szegmensenként --
try:
    from lifelines import KaplanMeierFitter

    fig_b1 = go.Figure()
    kmf = KaplanMeierFitter()

    segments_to_plot = ["Agent Builder", "Manual Coder", "Viewer"]
    seg_colors = {"Agent Builder": "#00b4d8", "Manual Coder": "#90e0ef", "Viewer": "#555566"}

    for seg in segments_to_plot:
        sub = feat[feat["segment"] == seg]
        if len(sub) < 10:
            continue
        kmf.fit(sub["survival_time"], sub["had_engagement"], label=seg)
        sf = kmf.survival_function_.reset_index()
        ci = kmf.confidence_interval_

        col = seg_colors.get(seg, "#888")
        fig_b1.add_trace(go.Scatter(
            x=sf["timeline"],
            y=sf[seg],
            mode="lines",
            name=f"{seg} (n={len(sub):,})",
            line=dict(color=col, width=3),
        ))
        # Confidence interval
        lo_col = [c for c in ci.columns if "lower" in c]
        hi_col = [c for c in ci.columns if "upper" in c]
        if lo_col and hi_col:
            fig_b1.add_trace(go.Scatter(
                x=pd.concat([sf["timeline"], sf["timeline"][::-1]]),
                y=pd.concat([ci[hi_col[0]], ci[lo_col[0]][::-1]]),
                fill="toself",
                fillcolor=col,
                opacity=0.1,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

    fig_b1.update_layout(
        title="Survival Görbe -- Meddig marad aktív a user?<br>"
              "<sup>Kaplan-Meier: P(aktív marad legalabb N napon) szegmensenként</sup>",
        xaxis_title="Aktív napok száma",
        yaxis_title="Arány még aktív",
        template="plotly_dark",
        height=480,
    )
    write_html(fig_b1, f"{OUTPUT_DIR}/07_survival_by_segment.html")
    print(f"  Saved: {OUTPUT_DIR}/07_survival_by_segment.html")

    # -- B2. Survival: visszaterest alapjan (24h kuszob) --
    feat["return_cohort"] = "Soha nem tert vissza"
    feat.loc[feat["had_second_session"] == 1, "return_cohort"] = "Visszajott (>24h)"
    feat.loc[
        (feat["had_second_session"] == 1) &
        (feat["time_to_return_hours"] <= 24),
        "return_cohort"
    ] = "Gyorsan visszajott (<=24h)"

    return_colors = {
        "Gyorsan visszajott (<=24h)": "#00b4d8",
        "Visszajott (>24h)"         : "#90e0ef",
        "Soha nem tert vissza"      : "#555566",
    }
    return_order = list(return_colors.keys())

    print("\n  Return cohort eloszlas:")
    rc_stats = feat.groupby("return_cohort").agg(
        users  = ("days_active", "count"),
        pct_ab = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
        avg_days = ("days_active", "mean"),
    )
    print(rc_stats.to_string())

    fig_b2 = go.Figure()
    for rc in return_order:
        sub = feat[feat["return_cohort"] == rc]
        if len(sub) < 5:
            continue
        kmf.fit(sub["survival_time"], sub["had_engagement"], label=rc)
        sf = kmf.survival_function_.reset_index()
        col = return_colors.get(rc, "#888")
        fig_b2.add_trace(go.Scatter(
            x=sf["timeline"],
            y=sf[rc],
            mode="lines",
            name=f"{rc} (n={len(sub):,})",
            line=dict(color=col, width=3),
        ))

    fig_b2.update_layout(
        title="Survival Görbe -- Visszatérési minta kritikus-e?<br>"
              "<sup>Aki 24 órán belül visszajön vs. aki kesobb vs. aki soha</sup>",
        xaxis_title="Aktív napok száma",
        yaxis_title="Arány még aktív",
        template="plotly_dark",
        height=480,
    )
    write_html(fig_b2, f"{OUTPUT_DIR}/07_survival_by_return.html")
    print(f"  Saved: {OUTPUT_DIR}/07_survival_by_return.html")

    # -- B3. Survival: agent adoptacio szerint --
    fig_b3 = go.Figure()
    adoption_groups = {
        "Korai adoptalo (<1h)"  : feat["adopted_agent_early"] == 1,
        "Keso adoptalo (1h+)"   : (feat["ever_used_agent"] == 1) & (feat["adopted_agent_early"] == 0),
        "Soha nem adoptalta"    : feat["ever_used_agent"] == 0,
    }
    adoption_colors = {
        "Korai adoptalo (<1h)" : "#00b4d8",
        "Keso adoptalo (1h+)"  : "#90e0ef",
        "Soha nem adoptalta"   : "#555566",
    }
    for label, mask in adoption_groups.items():
        sub = feat[mask]
        if len(sub) < 5:
            continue
        kmf.fit(sub["survival_time"], sub["had_engagement"], label=label)
        sf = kmf.survival_function_.reset_index()
        col = adoption_colors.get(label, "#888")
        fig_b3.add_trace(go.Scatter(
            x=sf["timeline"],
            y=sf[label],
            mode="lines",
            name=f"{label} (n={len(sub):,})",
            line=dict(color=col, width=3),
        ))

    fig_b3.update_layout(
        title="Survival Görbe -- Agent adoptacio idozitese szamit-e?<br>"
              "<sup>Korai vs keso vs soha nem adoptalo</sup>",
        xaxis_title="Aktív napok száma",
        yaxis_title="Arány még aktív",
        template="plotly_dark",
        height=480,
    )
    write_html(fig_b3, f"{OUTPUT_DIR}/07_survival_by_adoption.html")
    print(f"  Saved: {OUTPUT_DIR}/07_survival_by_adoption.html")

    # -- B4. time_to_return_hours kuszob elemzés --
    # Van-e kritikus kuszob ahol a visszateres prediktiv ereje ugrik?
    thresholds = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 1d, 2d, 3d, 1 week
    threshold_stats = []
    for t in thresholds:
        returned_early = feat[
            (feat["had_second_session"] == 1) &
            (feat["time_to_return_hours"] <= t)
        ]
        returned_late = feat[
            (feat["had_second_session"] == 1) &
            (feat["time_to_return_hours"] > t)
        ]
        threshold_stats.append({
            "threshold_h" : t,
            "n_early"     : len(returned_early),
            "pct_ab_early": (returned_early["segment"] == "Agent Builder").mean() * 100
                            if len(returned_early) > 0 else 0,
            "n_late"      : len(returned_late),
            "pct_ab_late" : (returned_late["segment"] == "Agent Builder").mean() * 100
                            if len(returned_late) > 0 else 0,
        })

    thresh_df = pd.DataFrame(threshold_stats)
    print("\n  Visszaterest kuszob elemzés (Agent Builder % early vs late):")
    print(thresh_df.to_string(index=False))

    fig_b4 = go.Figure()
    fig_b4.add_trace(go.Scatter(
        x=thresh_df["threshold_h"],
        y=thresh_df["pct_ab_early"],
        mode="lines+markers",
        name="Gyorsan visszajott (<= N ora)",
        line=dict(color="#00b4d8", width=3),
        marker=dict(size=10),
    ))
    fig_b4.add_trace(go.Scatter(
        x=thresh_df["threshold_h"],
        y=thresh_df["pct_ab_late"],
        mode="lines+markers",
        name="Kesobb jott vissza (> N ora)",
        line=dict(color="#ff6b6b", width=3, dash="dot"),
        marker=dict(size=10),
    ))
    fig_b4.update_layout(
        title="Visszaterest kuszob -- mikor a legelkotelezettebbek?<br>"
              "<sup>Agent Builder % azon usereknel akik <= N orán belül visszajottek</sup>",
        xaxis_title="Kuszob (ora)",
        yaxis_title="Agent Builder %",
        template="plotly_dark",
        height=420,
        xaxis=dict(
            tickmode="array",
            tickvals=thresholds,
            ticktext=["6h", "12h", "24h", "48h", "72h", "1 het"],
        ),
    )
    write_html(fig_b4, f"{OUTPUT_DIR}/07_return_threshold.html")
    print(f"  Saved: {OUTPUT_DIR}/07_return_threshold.html")

    print("\n  Survival analysis complete (lifelines)")

except ImportError:
    print("  lifelines nincs telepitve -- pip install lifelines")
    print("  Survival analysis atugrorva, folytatas az osszes tobbi charttal...")

    # Fallback: egyszerű bar chart survival helyett
    days_bins = [1, 2, 3, 5, 7, 10, 14, 20, 30, 999]
    days_labels = ["1", "2", "3", "4-5", "6-7", "8-10", "11-14", "15-20", "21-30", "30+"]

    feat["days_bucket"] = pd.cut(feat["days_active"], bins=days_bins,
                                  labels=days_labels[:len(days_bins)-1], right=False)
    retention = feat.groupby(["days_bucket", "segment"], observed=True).size().unstack(fill_value=0)
    fig_fallback = px.bar(
        retention.reset_index().melt(id_vars="days_bucket"),
        x="days_bucket", y="value", color="segment",
        barmode="stack",
        title="Aktív napok eloszlása szegmensenként (fallback -- lifelines nelkul)",
        template="plotly_dark",
        color_discrete_map=SEGMENT_COLORS,
    )
    write_html(fig_fallback, f"{OUTPUT_DIR}/07_days_active_distribution.html")
    print(f"  Saved: {OUTPUT_DIR}/07_days_active_distribution.html")

print("\n[OK] Signup hour + Survival analysis complete.")
print(f"  Signup hour: {OUTPUT_DIR}/07_signup_hour_ab_pct.html")
print(f"  Survival szegmens: {OUTPUT_DIR}/07_survival_by_segment.html")
print(f"  Survival visszateres: {OUTPUT_DIR}/07_survival_by_return.html")
print(f"  Survival adoptacio: {OUTPUT_DIR}/07_survival_by_adoption.html")
print(f"  Visszateres kuszob: {OUTPUT_DIR}/07_return_threshold.html")
