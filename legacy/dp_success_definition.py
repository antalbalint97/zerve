"""
=============================================================
 03_success_definition.py  —  Define & Label "Success"
 Zerve Hackathon 2026: "What Drives Successful Usage?"
=============================================================
Input : outputs/user_features.parquet
Output: outputs/user_features_labeled.parquet
        outputs/03_success_distribution.html
        outputs/03_success_criteria_heatmap.html

SUCCESS DEFINITION — Composite Engagement Score (CES)
------------------------------------------------------
A user is "successful" if they meet ≥ 2 of these 5 criteria:

  C1  DEPTH       — ran code AND created agent blocks
  C2  RETENTION   — active on ≥ 3 distinct days
  C3  COMPLEXITY  — used requirements_build OR previewed output
  C4  AI_ADOPTION — accepted ≥ 1 agent suggestion
  C5  REPRODUCIBILITY — ran the same canvas ≥ 3 times total

This is intentionally multi-dimensional:
  • It correlates with upgrade likelihood (business value)
  • It's not trivially gamed by a single burst of activity
  • It covers the three pillars Zerve cares about:
    analysis depth, reproducibility, workflow complexity
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

INPUT_PATH = "outputs/user_features.parquet"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────
print("Loading feature matrix...")
feat = pd.read_parquet(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

# ── DEFINE CRITERIA ───────────────────────────────────────

def safe_col(df, col, default=0):
    """Return column or zeros if it doesn't exist yet."""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)

# C1: Depth — ran code AND created agent blocks
feat["c1_depth"] = (
    (safe_col(feat, "total_code_runs") >= 1) &
    (safe_col(feat, "agent_blocks_created") >= 1)
).astype(int)

# C2: Retention — active ≥ 3 days
feat["c2_retention"] = (
    safe_col(feat, "days_active") >= 3
).astype(int)

# C3: Complexity — requirements built OR output previewed
feat["c3_complexity"] = (
    (safe_col(feat, "requirements_built") >= 1) |
    (safe_col(feat, "previewed_output") >= 1)
).astype(int)

# C4: AI Adoption — accepted ≥ 1 agent suggestion
feat["c4_ai_adoption"] = (
    safe_col(feat, "agent_suggestions_accepted") >= 1
).astype(int)

# C5: Reproducibility — ≥ 3 total code runs
feat["c5_reproducibility"] = (
    safe_col(feat, "total_code_runs") >= 3
).astype(int)

# Composite score (0–5)
criteria_cols = ["c1_depth", "c2_retention", "c3_complexity",
                 "c4_ai_adoption", "c5_reproducibility"]
feat["ces_score"]  = feat[criteria_cols].sum(axis=1)
feat["is_success"] = (feat["ces_score"] >= 2).astype(int)

# ── STATS ─────────────────────────────────────────────────
n_total   = len(feat)
n_success = feat["is_success"].sum()
pct       = n_success / n_total * 100

print(f"\n── Success label distribution ──")
print(f"  Successful users : {n_success:,}  ({pct:.1f}%)")
print(f"  Other users      : {n_total - n_success:,}  ({100-pct:.1f}%)")

print(f"\n── CES score distribution ──")
print(feat["ces_score"].value_counts().sort_index().to_string())

print(f"\n── Criteria hit rates ──")
for col in criteria_cols:
    hits = feat[col].sum()
    print(f"  {col:25s}: {hits:>5,}  ({hits/n_total*100:.1f}%)")

# ── PLOT 1: CES score histogram ───────────────────────────
score_counts = feat["ces_score"].value_counts().sort_index().reset_index()
score_counts.columns = ["ces_score", "users"]
score_counts["label"] = score_counts["ces_score"].apply(
    lambda s: "✅ Success" if s >= 2 else "❌ Not yet"
)

fig1 = px.bar(
    score_counts,
    x="ces_score", y="users",
    color="label",
    color_discrete_map={"✅ Success": "#00b4d8", "❌ Not yet": "#555555"},
    title="Composite Engagement Score Distribution<br>"
          "<sup>Score ≥ 2 = Successful User</sup>",
    labels={"ces_score": "CES Score (0–5)", "users": "Number of Users"},
    template="plotly_dark",
    text="users",
)
fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
fig1.write_html(f"{OUTPUT_DIR}/03_success_distribution.html")
print(f"\n  ✓ Saved: {OUTPUT_DIR}/03_success_distribution.html")

# ── PLOT 2: Criteria overlap heatmap ─────────────────────
criteria_labels = {
    "c1_depth"          : "C1: Depth\n(code + agent)",
    "c2_retention"      : "C2: Retention\n(≥3 days)",
    "c3_complexity"     : "C3: Complexity\n(req/preview)",
    "c4_ai_adoption"    : "C4: AI Adoption\n(accepted suggestion)",
    "c5_reproducibility": "C5: Reproducibility\n(≥3 runs)",
}

overlap = feat[criteria_cols].rename(columns=criteria_labels)
corr = overlap.corr()

fig2 = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Criteria Correlation Heatmap",
    template="plotly_dark",
    aspect="auto",
)
fig2.write_html(f"{OUTPUT_DIR}/03_success_criteria_heatmap.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/03_success_criteria_heatmap.html")

# ── PLOT 3: Radar chart — avg feature values by success ──
radar_features = [
    ("days_active",                "Days Active"),
    ("total_code_runs",            "Code Runs"),
    ("agent_total_events",         "Agent Events"),
    ("unique_canvases",            "Canvases Used"),
    ("agent_suggestions_accepted", "AI Suggestions Accepted"),
    ("avg_session_depth",          "Avg Session Depth"),
]

radar_data = {}
for col, label in radar_features:
    if col not in feat.columns:
        continue
    col_max = feat[col].quantile(0.95)
    if col_max == 0:
        continue
    success_mean = feat.loc[feat["is_success"] == 1, col].mean() / col_max
    other_mean   = feat.loc[feat["is_success"] == 0, col].mean() / col_max
    radar_data[label] = {"success": success_mean, "other": other_mean}

if radar_data:
    cats   = list(radar_data.keys())
    s_vals = [radar_data[c]["success"] for c in cats]
    o_vals = [radar_data[c]["other"]   for c in cats]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=s_vals + [s_vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="✅ Successful",
        line_color="#00b4d8",
    ))
    fig3.add_trace(go.Scatterpolar(
        r=o_vals + [o_vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="❌ Not Successful",
        line_color="#ff6b6b",
        opacity=0.6,
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Behaviour Profile: Successful vs. Not Successful Users<br>"
              "<sup>Values normalised to 95th percentile</sup>",
        template="plotly_dark",
        height=550,
    )
    fig3.write_html(f"{OUTPUT_DIR}/03_success_radar.html")
    print(f"  ✓ Saved: {OUTPUT_DIR}/03_success_radar.html")

# ── SAVE LABELED FEATURES ────────────────────────────────
out_path = f"{OUTPUT_DIR}/user_features_labeled.parquet"
feat.to_parquet(out_path)
feat.to_csv(f"{OUTPUT_DIR}/user_features_labeled.csv")
print(f"\n✅  Labeled features saved → {out_path}")
print(f"    Shape: {feat.shape}")
