import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 06_kpi_and_modeling.py  --  KPI Calculation & Modeling
 Zerve Hackathon 2026
=============================================================
Input : outputs/user_features_segmented.parquet
Output: outputs/06_kpi_*.html
        outputs/06_model_*.html
        outputs/model_rf.joblib

KPIs in a segment x cohort matrix, followed by an ML model
that predicts whether a user will become an Agent Builder/Runner
(= the definition of "success", now data-driven).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features
from analytics.viz import SEGMENT_COLORS, SEG_ORDER, write_html
warnings.filterwarnings("ignore")

INPUT_PATH = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

# =======================================================
# A) KPI MATRIX
# =======================================================
print("\n-- A) KPI Matrix --")

kpi_by_seg = feat.groupby("segment").agg(
    users                   = ("total_events", "count"),
    # Engagement
    avg_days_active         = ("days_active", "mean"),
    median_days_active      = ("days_active", "median"),
    avg_manual_runs         = ("manual_runs", "mean"),
    avg_agent_tools         = ("agent_tool_calls_total", "mean"),
    avg_agent_build         = ("agent_build_calls", "mean"),
    avg_conversations       = ("agent_conversations", "mean"),
    avg_canvases            = ("unique_canvases", "mean"),
    # Depth
    avg_tool_types          = ("agent_tool_types_used", "mean"),
    avg_unique_event_types  = ("unique_event_types", "mean"),
    # Retention proxy
    pct_multi_day           = ("days_active", lambda x: (x >= 3).mean() * 100),
    pct_week_plus           = ("days_active", lambda x: (x >= 7).mean() * 100),
    # Credit pressure
    pct_credit_exceeded     = ("had_credit_exceeded", "mean"),
    pct_addon_credits       = ("had_addon_credits", "mean"),
    # Agent adoption
    pct_ever_agent          = ("ever_used_agent", "mean"),
    pct_early_adopter       = ("adopted_agent_early", "mean"),
).round(2)

kpi_by_seg = kpi_by_seg.reindex([s for s in SEG_ORDER if s in kpi_by_seg.index])
kpi_by_seg["pct_credit_exceeded"] *= 100
kpi_by_seg["pct_addon_credits"]   *= 100
kpi_by_seg["pct_ever_agent"]      *= 100
kpi_by_seg["pct_early_adopter"]   *= 100
kpi_by_seg["pct_of_users"]         = kpi_by_seg["users"] / len(feat) * 100

print(kpi_by_seg[["users", "pct_of_users", "avg_days_active",
                    "avg_agent_tools", "pct_multi_day", "pct_credit_exceeded"]].to_string())

# KPI heatmap
display_kpis = {
    "Avg active days"    : "avg_days_active",
    "Avg manual run"   : "avg_manual_runs",
    "Avg agent tool"   : "avg_agent_tools",
    "Avg agent build"  : "avg_agent_build",
    "Avg canvas"       : "avg_canvases",
    "% 3+ day users"       : "pct_multi_day",
    "% 7+ day users"       : "pct_week_plus",
    "% credit limit"   : "pct_credit_exceeded",
    "% used agent"  : "pct_ever_agent",
    "% early adopter" : "pct_early_adopter",
}

kpi_heatmap = pd.DataFrame({
    label: kpi_by_seg[col]
    for label, col in display_kpis.items()
    if col in kpi_by_seg.columns
})

# Normalization by column
kpi_norm = kpi_heatmap.apply(lambda x: x / x.max() if x.max() > 0 else x, axis=0)

fig_kpi = px.imshow(
    kpi_norm.T,
    text_auto=False,
    color_continuous_scale="Blues",
    title="KPI Matrix -- Segment x Metric<br><sup>Normalized values (1.0 = best segment on given metric)</sup>",
    template="plotly_dark",
    aspect="auto",
    labels=dict(x="Segment", y="KPI", color="Normalized value"),
)

# Actual values as annotations
for i, kpi_label in enumerate(kpi_norm.columns):
    for j, seg in enumerate(kpi_norm.index):
        raw_val = kpi_heatmap.loc[seg, kpi_label] if kpi_label in kpi_heatmap.columns else 0
        fig_kpi.add_annotation(
            x=j, y=i,
            text=f"{raw_val:.1f}",
            showarrow=False,
            font=dict(size=10, color="white"),
        )

write_html(fig_kpi, f"{OUTPUT_DIR}/06_kpi_heatmap.html")
print(f"  Saved: {OUTPUT_DIR}/06_kpi_heatmap.html")

# KPIs by cohort as well
kpi_cohort = feat.groupby("signup_cohort").agg(
    users               = ("total_events", "count"),
    avg_days_active     = ("days_active", "mean"),
    avg_agent_tools     = ("agent_tool_calls_total", "mean"),
    pct_agent_builder   = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
    pct_agent_runner    = ("segment", lambda x: (x == "Agent Runner").mean() * 100),
    pct_power_user      = ("segment", lambda x: x.isin(["Agent Builder", "Agent Runner"]).mean() * 100),
    pct_ghost           = ("segment", lambda x: (x == "Ghost").mean() * 100),
    pct_ever_agent      = ("ever_used_agent", "mean"),
).round(2)
kpi_cohort["pct_ever_agent"] *= 100

print("\n-- KPIs by cohort --")
print(kpi_cohort[["users", "pct_power_user", "pct_ghost",
                    "pct_ever_agent", "avg_days_active"]].to_string())

fig_kpi2 = make_subplots(rows=1, cols=2,
    subplot_titles=["Power User % by cohort", "Ghost % by cohort"])
colors = ["#00b4d8", "#48cae4", "#90e0ef", "#0077b6"]
fig_kpi2.add_trace(go.Bar(x=kpi_cohort.index, y=kpi_cohort["pct_power_user"].round(1),
    marker_color=colors, text=kpi_cohort["pct_power_user"].round(1),
    texttemplate="%{text}%", showlegend=False), row=1, col=1)
fig_kpi2.add_trace(go.Bar(x=kpi_cohort.index, y=kpi_cohort["pct_ghost"].round(1),
    marker_color=["#ff6b6b"]*4, text=kpi_cohort["pct_ghost"].round(1),
    texttemplate="%{text}%", showlegend=False), row=1, col=2)
fig_kpi2.update_layout(
    title="KPI Trend by Cohort<br><sup>Is the platform improving? More power users, fewer ghosts?</sup>",
    template="plotly_dark", height=400,
)
write_html(fig_kpi2, f"{OUTPUT_DIR}/06_kpi_cohort_trend.html")
print(f"  Saved: {OUTPUT_DIR}/06_kpi_cohort_trend.html")

# =======================================================
# B) MODELING -- TWO LEVELS
#
# B1. FULL POPULATION: who becomes an Agent Builder?
#     (based on early signals, leakage-free)
#
# B2. NARROWED: only among users who used the agent
#     ki lesz Agent Builder vs Viewer/Manual Coder?
#     This is the more interesting question -- there is no trivial
#     "never opened it" baseline effect
# =======================================================
print("\n-- B) Modeling --")

feat["is_agent_builder"] = (feat["segment"] == "Agent Builder").astype(int)

# Early signals -- behavior measurable after the first day
BASE_EARLY_FEATURES = [
    "ttf_manual_run_min",
    "ttf_agent_tool_min",
    "ttf_agent_chat_min",
    "adopted_agent_early",
    "ever_used_agent",
    "ever_ran_manually",
    "signed_up",
    "skipped_onboarding_form",
    "submitted_onboarding",
    "completed_onboarding",
]

# New features from the first session and return behavior
NEW_EARLY_FEATURES = [
    "first_session_events",
    "first_session_event_types",
    "first_session_duration_min",
    "first_session_had_agent",
    "first_session_had_run",
    "had_second_session",
    "time_to_return_hours",
    "signup_hour",
    "signup_is_weekend",
]

# Referrer one-hot columns
ref_cols = [c for c in feat.columns if c.startswith("ref_")]

ALL_EARLY_FEATURES = BASE_EARLY_FEATURES + NEW_EARLY_FEATURES + ref_cols
X_cols = [c for c in ALL_EARLY_FEATURES if c in feat.columns]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42,
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                    class_weight="balanced", random_state=42)),
    ]),
}

# -------------------------------------------------------
# B1. FULL POPULATION
# -------------------------------------------------------
print("\n  B1. Full population (n=4,771)")
print(f"  Target: Agent Builder {feat['is_agent_builder'].sum():,} ({feat['is_agent_builder'].mean()*100:.1f}%)")

X_full = feat[X_cols].fillna(0)
y_full = feat["is_agent_builder"]

print(f"  Feature matrix: {X_full.shape}")
print("\n  Cross-validation (5-fold):")
b1_results = {}
for name, model in models.items():
    auc = cross_val_score(model, X_full, y_full, cv=cv, scoring="roc_auc", n_jobs=1)
    ap  = cross_val_score(model, X_full, y_full, cv=cv, scoring="average_precision", n_jobs=1)
    b1_results[name] = auc.mean()
    print(f"    {name:<25} AUC={auc.mean():.3f}+/-{auc.std():.3f}  AvgPrec={ap.mean():.3f}")

# -------------------------------------------------------
# B2. NARROWED -- only among users who used the agent
# -------------------------------------------------------
print("\n  B2. Narrowed: users who ever used the agent (n=649)")
print("  Question: among agent users, what distinguishes")
print("  Agent Builders from Viewers/Manual Coders?")

agent_users = feat[feat["ever_used_agent"] == 1].copy()
n_ab = (agent_users["segment"] == "Agent Builder").sum()
print(f"\n  Agent Builder : {n_ab:,} ({n_ab/len(agent_users)*100:.1f}%)")
print(f"  Other agent users: {len(agent_users)-n_ab:,} ({(1-n_ab/len(agent_users))*100:.1f}%)")

# Exclude ever_used_agent from early signals here (everyone is 1)
# and replace ttf_agent_* with relative timing
X_narrow_cols = [c for c in X_cols if c != "ever_used_agent"]
X_narrow = agent_users[X_narrow_cols].fillna(0)
y_narrow = (agent_users["segment"] == "Agent Builder").astype(int)

print(f"  Feature matrix: {X_narrow.shape}")
print("\n  Cross-validation (5-fold):")
b2_results = {}
for name, model in models.items():
    if len(y_narrow.unique()) < 2:
        print(f"    {name:<25} -- skip (only 1 class)")
        continue
    auc = cross_val_score(model, X_narrow, y_narrow, cv=cv, scoring="roc_auc", n_jobs=1)
    ap  = cross_val_score(model, X_narrow, y_narrow, cv=cv, scoring="average_precision", n_jobs=1)
    b2_results[name] = auc.mean()
    print(f"    {name:<25} AUC={auc.mean():.3f}+/-{auc.std():.3f}  AvgPrec={ap.mean():.3f}")

# -------------------------------------------------------
# Save final RF models
# -------------------------------------------------------
# B1 final model
rf_full = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=42, n_jobs=1)
rf_full.fit(X_full, y_full)
joblib.dump(rf_full, f"{OUTPUT_DIR}/model_rf_full.joblib")
joblib.dump(X_cols, f"{OUTPUT_DIR}/feature_names_full.joblib")

# B2 final model
rf_narrow = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=42, n_jobs=1)
rf_narrow.fit(X_narrow, y_narrow)
joblib.dump(rf_narrow, f"{OUTPUT_DIR}/model_rf_narrow.joblib")
joblib.dump(X_narrow_cols, f"{OUTPUT_DIR}/feature_names_narrow.joblib")

# -------------------------------------------------------
# Feature importance -- B2 (narrowed, more interesting)
# -------------------------------------------------------
fi = pd.DataFrame({
    "feature"   : X_narrow_cols,
    "importance": rf_narrow.feature_importances_,
}).sort_values("importance", ascending=False).head(15)

print("\n  -- Feature importance (narrowed model, among agent users) --")
FEATURE_NOTES = {
    "ttf_agent_tool_min"       : "The earlier they use it, the more likely success becomes",
    "adopted_agent_early"      : "Used the agent within the first hour",
    "ttf_agent_chat_min"       : "When they first opened the chat",
    "first_session_had_agent"  : "Whether there was an agent interaction in the first session",
    "first_session_events"     : "How active they were in the first session",
    "first_session_duration_min": "How long they stayed in the first session",
    "had_second_session"       : "Whether they came back for a second session at all",
    "time_to_return_hours"     : "The sooner they returned, the more engaged they were",
    "ever_ran_manually"        : "Whether they also ran code manually",
    "signup_hour"              : "Time of day influences engagement",
    "signup_is_weekend"        : "Weekend signup vs workday signup",
    "ttf_manual_run_min"       : "Time of first manual run",
    "first_session_event_types": "How much they explored the platform in the first session",
}
for _, row in fi.iterrows():
    if row["importance"] < 0.01:
        break
    note = FEATURE_NOTES.get(row["feature"], "")
    print(f"    {row['feature']:<35} {row['importance']:.3f}  {note}")

# Feature importance chart -- B2
fig_fi_narrow = px.bar(
    fi, x="importance", y="feature", orientation="h",
    title="What differentiates Agent Builders from other agent users?<br>"
          "<sup>Only among users who used the agent -- early signals -- leakage-free</sup>",
    color="importance", color_continuous_scale="Teal",
    template="plotly_dark",
    labels={"importance": "Importance", "feature": ""},
)
fig_fi_narrow.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
write_html(fig_fi_narrow, f"{OUTPUT_DIR}/06_feature_importance_narrow.html")
print(f"\n  Saved: {OUTPUT_DIR}/06_feature_importance_narrow.html")

# Feature importance chart -- B1 is
fi_full = pd.DataFrame({
    "feature"   : X_cols,
    "importance": rf_full.feature_importances_,
}).sort_values("importance", ascending=False).head(15)

fig_fi_full = px.bar(
    fi_full, x="importance", y="feature", orientation="h",
    title="What predicts becoming an Agent Builder? (full population)<br>"
          "<sup>Early signals -- 10+9 features -- leakage-free</sup>",
    color="importance", color_continuous_scale="Teal",
    template="plotly_dark",
    labels={"importance": "Importance", "feature": ""},
)
fig_fi_full.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
write_html(fig_fi_full, f"{OUTPUT_DIR}/06_feature_importance.html")
print(f"  Saved: {OUTPUT_DIR}/06_feature_importance.html")

# ROC curves -- both models
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_narrow, y_narrow, test_size=0.2, stratify=y_narrow, random_state=42)

fig_roc = make_subplots(rows=1, cols=2,
    subplot_titles=["B1: Full population", "B2: Agent users only"])

for name, model in models.items():
    # B1
    model.fit(X_train_f, y_train_f)
    prob_f = model.predict_proba(X_test_f)[:, 1] if hasattr(model, "predict_proba") \
             else model.decision_function(X_test_f)
    fpr_f, tpr_f, _ = roc_curve(y_test_f, prob_f)
    auc_f = roc_auc_score(y_test_f, prob_f)
    fig_roc.add_trace(go.Scatter(x=fpr_f, y=tpr_f, mode="lines",
        name=f"{name} ({auc_f:.3f})", showlegend=True), row=1, col=1)

    # B2
    from sklearn.base import clone
    model_clone = clone(model)
    try:
        model_clone.fit(X_train_n, y_train_n)
        prob_n = model_clone.predict_proba(X_test_n)[:, 1] if hasattr(model_clone, "predict_proba") \
                 else model_clone.decision_function(X_test_n)
        fpr_n, tpr_n, _ = roc_curve(y_test_n, prob_n)
        auc_n = roc_auc_score(y_test_n, prob_n)
        fig_roc.add_trace(go.Scatter(x=fpr_n, y=tpr_n, mode="lines",
            name=f"{name} ({auc_n:.3f})", showlegend=False), row=1, col=2)
    except Exception as e:
        print(f"    {name} B2 skip: {e}")

for col in [1, 2]:
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
        line=dict(dash="dash", color="grey"), showlegend=False), row=1, col=col)

fig_roc.update_layout(
    title="ROC Curves -- B1 (full) vs B2 (agent users only)<br>"
          "<sup>Based on early signals -- leakage-free</sup>",
    template="plotly_dark", height=420,
)
write_html(fig_roc, f"{OUTPUT_DIR}/06_roc_curves.html")
print(f"  Saved: {OUTPUT_DIR}/06_roc_curves.html")

print(f"\n  Models saved:")
print(f"    {OUTPUT_DIR}/model_rf_full.joblib    (full population)")
print(f"    {OUTPUT_DIR}/model_rf_narrow.joblib  (among agent users)")
print("\nKPI & Modeling complete.")

