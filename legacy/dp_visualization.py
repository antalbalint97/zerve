"""
=============================================================
 05_visualization.py  —  The Story: Key Insight Charts
 Zerve Hackathon 2026: "What Drives Successful Usage?"
=============================================================
Input : outputs/user_features_labeled.parquet
        outputs/model_rf.joblib
Output: outputs/05_*.html  — presentation-ready charts

MAIN NARRATIVE:
  "The 15-Minute Rule & The Agent Flywheel"

  Users who run their first block within 15 minutes of signing up
  AND engage with the AI agent are dramatically more likely to
  become long-term successful users.

  Three user archetypes emerge:
    🚀 Power Users    — agent + code + multi-day retention
    🔍 Explorers      — active but not yet using advanced features
    👻 Ghost Users    — signed up, never returned
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = "outputs/user_features_labeled.parquet"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────
feat = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(feat):,} users")

def safe(col, default=0):
    return feat[col] if col in feat.columns else pd.Series(default, index=feat.index)

# ── 1. THE 15-MINUTE RULE ────────────────────────────────
print("\n── 1. 15-Minute Rule analysis ──")

ttv = feat["time_to_first_run_min"].copy()
has_ttv = ttv > 0  # only users who signed up AND ran code

bins   = [0, 5, 15, 30, 60, 120, 240, 480, 10_080]
labels = ["<5m", "5-15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-8h", ">8h"]

feat["ttv_bucket"] = pd.cut(
    ttv.where(has_ttv),
    bins=bins, labels=labels, right=False
)

ttv_stats = (
    feat.dropna(subset=["ttv_bucket"])
        .groupby("ttv_bucket", observed=True)
        .agg(
            users       = ("is_success", "count"),
            success_rate= ("is_success", "mean"),
        )
        .reset_index()
)
ttv_stats["success_pct"] = ttv_stats["success_rate"] * 100

print(ttv_stats.to_string(index=False))

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(
    go.Bar(x=ttv_stats["ttv_bucket"], y=ttv_stats["users"],
           name="Users", marker_color="#2d3a4a", opacity=0.8),
    secondary_y=False,
)
fig1.add_trace(
    go.Scatter(x=ttv_stats["ttv_bucket"], y=ttv_stats["success_pct"],
               name="Success Rate %", mode="lines+markers",
               line=dict(color="#00b4d8", width=3),
               marker=dict(size=10)),
    secondary_y=True,
)
fig1.add_vline(x=1.5, line_dash="dash", line_color="#ff6b6b",
               annotation_text="15-min threshold", annotation_position="top right")
fig1.update_layout(
    title="The 15-Minute Rule: Time to First Code Run vs Success Rate<br>"
          "<sup>Users who run code within 15 minutes are significantly more successful</sup>",
    template="plotly_dark",
    height=450,
    legend=dict(orientation="h", y=1.1),
)
fig1.update_yaxes(title_text="Number of Users",  secondary_y=False)
fig1.update_yaxes(title_text="Success Rate (%)", secondary_y=True)
fig1.write_html(f"{OUTPUT_DIR}/05_15min_rule.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/05_15min_rule.html")

# ── 2. THE AGENT FLYWHEEL ────────────────────────────────
print("\n── 2. Agent Flywheel analysis ──")

agent_bins = [-1, 0, 1, 5, 20, 100, 99999]
agent_labels = ["0 (none)", "1", "2-5", "6-20", "21-100", "100+"]

feat["agent_bucket"] = pd.cut(
    safe("agent_total_events"),
    bins=agent_bins, labels=agent_labels
)

flywheel = (
    feat.groupby("agent_bucket", observed=True)
        .agg(
            users        = ("is_success", "count"),
            success_rate = ("is_success", "mean"),
            avg_code_runs= ("total_code_runs", "mean"),
        )
        .reset_index()
)
flywheel["success_pct"] = flywheel["success_rate"] * 100

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(
    go.Bar(x=flywheel["agent_bucket"], y=flywheel["users"],
           name="Users", marker_color="#1a2a3a", opacity=0.8),
    secondary_y=False,
)
fig2.add_trace(
    go.Scatter(x=flywheel["agent_bucket"], y=flywheel["success_pct"],
               name="Success Rate %", mode="lines+markers",
               line=dict(color="#00b4d8", width=3),
               marker=dict(size=10, symbol="diamond")),
    secondary_y=True,
)
fig2.add_trace(
    go.Scatter(x=flywheel["agent_bucket"], y=flywheel["avg_code_runs"],
               name="Avg Code Runs", mode="lines+markers",
               line=dict(color="#90e0ef", width=2, dash="dot"),
               marker=dict(size=8)),
    secondary_y=True,
)
fig2.update_layout(
    title="The Agent Flywheel: AI Agent Usage vs Success Rate<br>"
          "<sup>More agent interaction → higher success AND more code execution</sup>",
    template="plotly_dark",
    height=450,
)
fig2.update_yaxes(title_text="Number of Users",  secondary_y=False)
fig2.update_yaxes(title_text="Rate / Avg Runs",  secondary_y=True)
fig2.write_html(f"{OUTPUT_DIR}/05_agent_flywheel.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/05_agent_flywheel.html")

# ── 3. USER CLUSTERING (3 archetypes) ───────────────────
print("\n── 3. User clustering ──")

cluster_features = [
    "days_active", "total_code_runs", "agent_total_events",
    "unique_canvases", "avg_session_depth", "requirements_built",
    "previewed_output", "agent_suggestions_accepted",
]
cluster_features = [c for c in cluster_features if c in feat.columns]

X_clust = feat[cluster_features].fillna(0)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

# Elbow method to confirm k=3
inertias = []
K_range  = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig_elbow = px.line(
    x=list(K_range), y=inertias,
    markers=True,
    title="K-Means Elbow Plot",
    labels={"x": "k (clusters)", "y": "Inertia"},
    template="plotly_dark",
)
fig_elbow.write_html(f"{OUTPUT_DIR}/05_elbow.html")

# Final clustering with k=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
feat["cluster"] = km3.fit_predict(X_scaled)

# Name clusters by success rate
cluster_stats = (
    feat.groupby("cluster")
        .agg(
            users        = ("is_success", "count"),
            success_rate = ("is_success", "mean"),
            avg_days     = ("days_active", "mean"),
            avg_runs     = ("total_code_runs", "mean"),
            avg_agent    = ("agent_total_events", "mean"),
        )
        .reset_index()
        .sort_values("success_rate", ascending=False)
)

archetype_names = ["🚀 Power Users", "🔍 Explorers", "👻 Ghost Users"]
cluster_stats["archetype"] = archetype_names
# Map back to feat
id_to_name = dict(zip(cluster_stats["cluster"], cluster_stats["archetype"]))
feat["archetype"] = feat["cluster"].map(id_to_name)

print(cluster_stats[["archetype", "users", "success_rate",
                       "avg_days", "avg_runs", "avg_agent"]].to_string(index=False))

# Cluster profile bar chart
profile_cols = ["avg_days", "avg_runs", "avg_agent"]
profile_labels = {"avg_days": "Avg Days Active",
                  "avg_runs": "Avg Code Runs",
                  "avg_agent": "Avg Agent Events"}
profile_melt = cluster_stats.melt(
    id_vars="archetype", value_vars=profile_cols,
    var_name="metric", value_name="value"
)
profile_melt["metric"] = profile_melt["metric"].map(profile_labels)

fig3 = px.bar(
    profile_melt,
    x="archetype", y="value",
    color="metric",
    barmode="group",
    title="User Archetypes: Behaviour Profiles<br>"
          "<sup>3 clusters identified via K-Means</sup>",
    template="plotly_dark",
    color_discrete_sequence=["#00b4d8", "#90e0ef", "#caf0f8"],
)
fig3.write_html(f"{OUTPUT_DIR}/05_archetypes.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/05_archetypes.html")

# ── 4. ONBOARDING IMPACT ─────────────────────────────────
print("\n── 4. Onboarding impact ──")

onb_cols = [
    ("completed_onboarding_tour", "Completed Tour"),
    ("skipped_onboarding_form",   "Skipped Form"),
    ("explored_quickstart",       "Explored Quickstart"),
]
onb_rows = []
for col, label in onb_cols:
    if col not in feat.columns:
        continue
    did     = feat.loc[feat[col] == 1, "is_success"].mean() * 100
    did_not = feat.loc[feat[col] == 0, "is_success"].mean() * 100
    n_did   = (feat[col] == 1).sum()
    onb_rows.append({"action": label,
                     "did_success_rate"    : did,
                     "did_not_success_rate": did_not,
                     "n_users"             : n_did})

if onb_rows:
    onb_df = pd.DataFrame(onb_rows)
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Did the action",
        x=onb_df["action"], y=onb_df["did_success_rate"],
        marker_color="#00b4d8", text=onb_df["did_success_rate"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig4.add_trace(go.Bar(
        name="Did NOT do action",
        x=onb_df["action"], y=onb_df["did_not_success_rate"],
        marker_color="#555555", text=onb_df["did_not_success_rate"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig4.update_layout(
        barmode="group",
        title="Onboarding Actions vs Success Rate<br>"
              "<sup>Does completing the tour actually predict long-term success?</sup>",
        yaxis_title="Success Rate (%)",
        template="plotly_dark",
        height=450,
    )
    fig4.write_html(f"{OUTPUT_DIR}/05_onboarding_impact.html")
    print(f"  ✓ Saved: {OUTPUT_DIR}/05_onboarding_impact.html")

# ── 5. SURVIVAL ANALYSIS (when do users disengage?) ──────
print("\n── 5. Retention curve ──")

try:
    from lifelines import KaplanMeierFitter

    feat["tenure_days"] = safe("days_active")
    feat["observed"]    = (feat["total_events"] > 1).astype(int)

    kmf = KaplanMeierFitter()
    fig5 = go.Figure()

    for label, mask, color in [
        ("✅ Successful",    feat["is_success"] == 1, "#00b4d8"),
        ("❌ Not Successful", feat["is_success"] == 0, "#ff6b6b"),
    ]:
        sub = feat[mask]
        kmf.fit(sub["tenure_days"], sub["observed"], label=label)
        sf = kmf.survival_function_.reset_index()
        ci = kmf.confidence_interval_.reset_index()

        fig5.add_trace(go.Scatter(
            x=sf["timeline"], y=sf[label],
            mode="lines", name=label,
            line=dict(color=color, width=3),
        ))
        # CI band
        lo_col = [c for c in ci.columns if "lower" in c]
        hi_col = [c for c in ci.columns if "upper" in c]
        if lo_col and hi_col:
            fig5.add_trace(go.Scatter(
                x=pd.concat([sf["timeline"], sf["timeline"][::-1]]),
                y=pd.concat([ci[hi_col[0]], ci[lo_col[0]][::-1]]),
                fill="toself", fillcolor=color,
                opacity=0.1, line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

    fig5.update_layout(
        title="Retention Curve: How Long Do Users Stay Active?<br>"
              "<sup>Kaplan-Meier survival estimate by success group</sup>",
        xaxis_title="Days Active",
        yaxis_title="Proportion Still Engaged",
        template="plotly_dark",
        height=450,
    )
    fig5.write_html(f"{OUTPUT_DIR}/05_retention_curve.html")
    print(f"  ✓ Saved: {OUTPUT_DIR}/05_retention_curve.html")

except ImportError:
    print("  (lifelines not installed — skipping survival analysis)")

# ── 6. COMBINED DASHBOARD ────────────────────────────────
print("\n── 6. Building summary dashboard ──")

fig_dash = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Success Rate by Time-to-First-Run",
        "Success Rate by Agent Usage",
        "User Archetype Sizes",
        "CES Score Distribution",
    ],
    specs=[
        [{"secondary_y": True}, {"secondary_y": True}],
        [{"type": "pie"},       {"type": "bar"}],
    ],
)

# Top-left: 15-min rule (simplified)
fig_dash.add_trace(
    go.Bar(x=ttv_stats["ttv_bucket"], y=ttv_stats["success_pct"],
           marker_color="#00b4d8", name="Success %", showlegend=False),
    row=1, col=1, secondary_y=False,
)

# Top-right: agent flywheel (simplified)
fig_dash.add_trace(
    go.Bar(x=flywheel["agent_bucket"], y=flywheel["success_pct"],
           marker_color="#90e0ef", name="Success %", showlegend=False),
    row=1, col=2, secondary_y=False,
)

# Bottom-left: archetype pie
fig_dash.add_trace(
    go.Pie(
        labels=cluster_stats["archetype"],
        values=cluster_stats["users"],
        hole=0.35,
        marker_colors=["#00b4d8", "#90e0ef", "#555555"],
        showlegend=True,
    ),
    row=2, col=1,
)

# Bottom-right: CES distribution
ces_dist = feat["ces_score"].value_counts().sort_index()
fig_dash.add_trace(
    go.Bar(x=ces_dist.index, y=ces_dist.values,
           marker_color=["#555555", "#555555", "#00b4d8", "#00b4d8", "#00b4d8", "#00b4d8"],
           name="Users", showlegend=False),
    row=2, col=2,
)

fig_dash.update_layout(
    title_text="Zerve User Success Analysis — Summary Dashboard",
    template="plotly_dark",
    height=750,
    showlegend=True,
)
fig_dash.write_html(f"{OUTPUT_DIR}/05_dashboard.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/05_dashboard.html")

# Save cluster labels for deployment
feat[["archetype", "cluster", "ces_score", "is_success"]].to_csv(
    f"{OUTPUT_DIR}/user_segments.csv"
)

print("\n✅  Visualization complete.")
print("    Key files:")
print(f"    → {OUTPUT_DIR}/05_dashboard.html   (main overview)")
print(f"    → {OUTPUT_DIR}/05_15min_rule.html  (key insight #1)")
print(f"    → {OUTPUT_DIR}/05_agent_flywheel.html (key insight #2)")
print(f"    → {OUTPUT_DIR}/05_archetypes.html  (user segments)")
