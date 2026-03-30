import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 09_fleet_cohort_model.py  --  Fleet-stilus Kohorsz Modell
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/09_fleet_*.html
        outputs/09_fleet_results.csv

Zerve-n ez Fleet-tel futna:
  spread(cohorts) -> [model_sep, model_oct, model_nov, model_dec]
                  -> gather() -> osszehasonlitas

Lokálisan: sequentialis, de ugyanaz a logika.

KERDES: Valtozik-e az ami megjósolja az Agent Builder-re
valast kohorszonkent?

Ha szeptemberben az elso session melysege a fo predictor,
de decemberben mar a visszateres a fo predictor -- ez azt
jelenti hogy a platform felhasznaloi mintak valtoztak,
es az onboarding mas uzeneteket kell hogy kozvetitsen.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = "outputs/user_features_segmented.parquet"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
feat = pd.read_parquet(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

# Korai jelzok (leakage-mentes)
EARLY_FEATURES = [
    "ttf_manual_run_min", "ttf_agent_tool_min", "ttf_agent_chat_min",
    "adopted_agent_early", "ever_used_agent", "ever_ran_manually",
    "signed_up", "skipped_onboarding_form", "submitted_onboarding",
    "completed_onboarding",
    "first_session_events", "first_session_event_types",
    "first_session_duration_min", "first_session_had_agent",
    "first_session_had_run",
    "had_second_session", "time_to_return_hours",
    "signup_hour", "signup_is_weekend",
]
ref_cols = [c for c in feat.columns if c.startswith("ref_")]
ALL_FEATURES = [c for c in EARLY_FEATURES + ref_cols if c in feat.columns]

feat["is_agent_builder"] = (feat["segment"] == "Agent Builder").astype(int)
COHORTS = sorted(feat["signup_cohort"].unique())
print(f"  Kohorszok: {COHORTS}")

# ============================================================
# FLEET LOGIKA -- minden kohorszon kul modell
# Zerve-n: spread(COHORTS) -> parallel blocks -> gather()
# ============================================================
print("\n-- Fleet-stilus Kohorsz Modellezes --")
print("  (Zerve-n ez spread() / gather() lenne)")
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fleet_results = []

for cohort in COHORTS:
    cohort_df = feat[feat["signup_cohort"] == cohort].copy()
    n_ab      = cohort_df["is_agent_builder"].sum()
    n_total   = len(cohort_df)

    print(f"  [{cohort}] n={n_total:,}  Agent Builder={n_ab} ({n_ab/n_total*100:.1f}%)")

    if n_ab < 10:
        print(f"    -- skip: tul keves Agent Builder ({n_ab})")
        fleet_results.append({
            "cohort": cohort, "n": n_total, "n_ab": n_ab,
            "auc_rf": None, "auc_lr": None,
            "top_feature_1": "N/A", "top_feature_2": "N/A", "top_feature_3": "N/A",
            "fi_1": 0, "fi_2": 0, "fi_3": 0,
        })
        continue

    X = cohort_df[ALL_FEATURES].fillna(0)
    y = cohort_df["is_agent_builder"]

    # RF
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    auc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    # LR
    lr = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500,
                                    class_weight="balanced", random_state=42)),
    ])
    auc_lr = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    # Feature importance (RF, full fit)
    rf.fit(X, y)
    fi = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    top3 = fi.head(3)

    print(f"    RF AUC={auc_rf:.3f}  LR AUC={auc_lr:.3f}")
    print(f"    Top features: {top3.index[0]} ({top3.iloc[0]:.3f}), "
          f"{top3.index[1]} ({top3.iloc[1]:.3f}), "
          f"{top3.index[2]} ({top3.iloc[2]:.3f})")

    fleet_results.append({
        "cohort"      : cohort,
        "n"           : n_total,
        "n_ab"        : n_ab,
        "pct_ab"      : round(n_ab / n_total * 100, 1),
        "auc_rf"      : round(auc_rf, 3),
        "auc_lr"      : round(auc_lr, 3),
        "top_feature_1": top3.index[0],
        "top_feature_2": top3.index[1] if len(top3) > 1 else "",
        "top_feature_3": top3.index[2] if len(top3) > 2 else "",
        "fi_1"        : round(top3.iloc[0], 3),
        "fi_2"        : round(top3.iloc[1], 3) if len(top3) > 1 else 0,
        "fi_3"        : round(top3.iloc[2], 3) if len(top3) > 2 else 0,
    })

results_df = pd.DataFrame(fleet_results)
results_df.to_csv(f"{OUTPUT_DIR}/09_fleet_results.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR}/09_fleet_results.csv")

# ============================================================
# VIZUALIZACIOK
# ============================================================
valid = results_df.dropna(subset=["auc_rf"])

# Chart 1: AUC kohorszonkent
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=valid["cohort"], y=valid["auc_rf"],
    name="Random Forest AUC",
    marker_color="#00b4d8",
    text=valid["auc_rf"].round(3),
    textposition="outside",
))
fig1.add_trace(go.Bar(
    x=valid["cohort"], y=valid["auc_lr"],
    name="Logistic Regression AUC",
    marker_color="#90e0ef",
    text=valid["auc_lr"].round(3),
    textposition="outside",
))
fig1.add_hline(y=0.7, line_dash="dash", line_color="#ff6b6b",
               annotation_text="0.7 kuszob", annotation_position="right")
fig1.update_layout(
    barmode="group",
    title="Fleet Kohorsz Modell -- AUC kohorszonkent<br>"
          "<sup>Mennyire jósolható meg az Agent Builder-re valas kohorszonkent?</sup>",
    yaxis=dict(range=[0, 1.1], title="ROC-AUC"),
    template="plotly_dark",
    height=420,
)
fig1.write_html(f"{OUTPUT_DIR}/09_fleet_auc_by_cohort.html")
print(f"  Saved: {OUTPUT_DIR}/09_fleet_auc_by_cohort.html")

# Chart 2: Top feature kohorszonkent -- valtoznak-e?
# Minden kohorsz top 3 feature-jet megjeleniti
fig2 = go.Figure()
colors = ["#00b4d8", "#48cae4", "#90e0ef"]

for i, (fi_col, feat_col) in enumerate(zip(
    ["fi_1", "fi_2", "fi_3"],
    ["top_feature_1", "top_feature_2", "top_feature_3"]
), start=1):
    valid_rows = valid.dropna(subset=[fi_col])
    fig2.add_trace(go.Bar(
        x=valid_rows["cohort"],
        y=valid_rows[fi_col],
        name=f"Top {i} feature",
        marker_color=colors[i-1],
        text=valid_rows[feat_col],
        textposition="inside",
        textfont=dict(size=9),
    ))

fig2.update_layout(
    barmode="stack",
    title="Melyik feature a legfontosabb kohorszonkent?<br>"
          "<sup>Valtoznak-e a siker elorejelzoi idovel?</sup>",
    yaxis_title="Feature importance osszeadva",
    template="plotly_dark",
    height=480,
    legend=dict(orientation="h", y=-0.15),
)
fig2.write_html(f"{OUTPUT_DIR}/09_fleet_top_features.html")
print(f"  Saved: {OUTPUT_DIR}/09_fleet_top_features.html")

# Chart 3: Minden kohorsz feature importance -- grouped bar
# Top 6 global feature minden kohorszon belul
top_global = feat[ALL_FEATURES].corrwith(feat["is_agent_builder"]).abs().nlargest(8).index.tolist()

fig3 = go.Figure()
cohort_palette = ["#00b4d8", "#48cae4", "#90e0ef", "#0077b6"]

for ci, cohort in enumerate(COHORTS):
    cohort_df = feat[feat["signup_cohort"] == cohort]
    n_ab = cohort_df["is_agent_builder"].sum()
    if n_ab < 10:
        continue
    X = cohort_df[top_global].fillna(0)
    y = cohort_df["is_agent_builder"]
    rf_c = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_c.fit(X, y)
    fi_vals = rf_c.feature_importances_

    fig3.add_trace(go.Bar(
        name=cohort,
        x=[f.replace("first_session_", "fs_").replace("time_to_", "t2_") for f in top_global],
        y=fi_vals,
        marker_color=cohort_palette[ci % len(cohort_palette)],
    ))

fig3.update_layout(
    barmode="group",
    title="Feature Importance Valtozasa Kohorszonkent<br>"
          "<sup>Ugyanazok a factorok jósolnak szeptemberben es decemberben?</sup>",
    yaxis_title="Feature importance",
    template="plotly_dark",
    height=480,
    legend=dict(orientation="h", y=-0.15),
)
fig3.write_html(f"{OUTPUT_DIR}/09_fleet_feature_drift.html")
print(f"  Saved: {OUTPUT_DIR}/09_fleet_feature_drift.html")

# Chart 4: Agent Builder arany trendje kohorszonkent
fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(go.Bar(
    x=results_df["cohort"],
    y=results_df["n"],
    name="Osszes user",
    marker_color="#2d3a4a",
    opacity=0.8,
), secondary_y=False)
fig4.add_trace(go.Scatter(
    x=results_df["cohort"],
    y=results_df["pct_ab"],
    name="Agent Builder %",
    mode="lines+markers",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=12, symbol="diamond"),
), secondary_y=True)
fig4.update_layout(
    title="Platform Novekedes vs Agent Builder Arany<br>"
          "<sup>A platform nott, de az Agent Builder arany csokkent -- skalazhato az onboarding?</sup>",
    template="plotly_dark",
    height=420,
)
fig4.update_yaxes(title_text="Osszes user", secondary_y=False)
fig4.update_yaxes(title_text="Agent Builder %", secondary_y=True)
fig4.write_html(f"{OUTPUT_DIR}/09_fleet_growth_vs_quality.html")
print(f"  Saved: {OUTPUT_DIR}/09_fleet_growth_vs_quality.html")

# ============================================================
# FLEET OSSZEFOGLALO
# ============================================================
print("\n-- Fleet Eredmenyek Osszefoglaloja --")
print(results_df[["cohort", "n", "n_ab", "pct_ab", "auc_rf",
                    "top_feature_1", "top_feature_2"]].to_string(index=False))

print("\n  Zerve Fleet implementacios megjegyzes:")
print("  Zerve-n ez a kod spread(cohorts) -> 4 parallel worker lenne,")
print("  mindegyik egy kohorszon fittel egy RF-et, az Aggregator blokk")
print("  osszegyujti a feature importance tablakat es osszehasonlitja.")
print("  Ez 4x gyorsabb futasi ido es demonstralja a Fleet hasznalatat.")

print("\n[OK] Fleet kohorsz modell complete.")
