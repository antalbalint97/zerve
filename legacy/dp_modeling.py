"""
=============================================================
 04_modeling.py  —  Predictive Modeling & Feature Importance
 Zerve Hackathon 2026: "What Drives Successful Usage?"
=============================================================
Input : outputs/user_features_labeled.parquet
Output: outputs/model_rf.joblib
        outputs/04_feature_importance.html
        outputs/04_model_comparison.html
        outputs/04_confusion_matrix.html
        outputs/04_shap_summary.html   (if shap available)
        outputs/model_metrics.csv

Models trained:
  1. Random Forest (main, for feature importance)
  2. Logistic Regression (interpretable baseline)
  3. Gradient Boosted Trees (performance benchmark)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

INPUT_PATH = "outputs/user_features_labeled.parquet"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────
print("Loading labeled features...")
feat = pd.read_parquet(INPUT_PATH)
print(f"  {len(feat):,} users  |  {feat.shape[1]} features")

# ── FEATURE SELECTION ────────────────────────────────────
# Drop meta/leakage columns
DROP_COLS = [
    "is_success", "ces_score",
    "c1_depth", "c2_retention", "c3_complexity",
    "c4_ai_adoption", "c5_reproducibility",  # criteria are part of target
    "first_seen", "last_seen",                # timestamps
    "signup_at", "first_run_at",
]
drop_existing = [c for c in DROP_COLS if c in feat.columns]

X = feat.drop(columns=drop_existing)
y = feat["is_success"]

# Keep only numeric columns
X = X.select_dtypes(include=[np.number]).fillna(0)

print(f"\n  Feature matrix: {X.shape}")
print(f"  Class balance: {y.mean():.1%} positive")

feature_names = X.columns.tolist()

# ── CROSS-VALIDATION SETUP ───────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
}

# ── TRAIN & EVALUATE ─────────────────────────────────────
results = {}
print("\n── Cross-validation (5-fold) ──")

for name, model in models.items():
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    ap_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    results[name] = {
        "ROC-AUC"  : auc_scores.mean(),
        "AUC std"  : auc_scores.std(),
        "Avg Prec" : ap_scores.mean(),
        "Accuracy" : acc_scores.mean(),
    }
    print(f"  {name:<25s}  AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}"
          f"  AvgPrec={ap_scores.mean():.3f}  Acc={acc_scores.mean():.3f}")

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv")

# ── FINAL RF MODEL (full dataset) ────────────────────────
print("\nTraining final Random Forest on full dataset...")
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)
joblib.dump(rf, f"{OUTPUT_DIR}/model_rf.joblib")
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.joblib")
print(f"  ✓ Model saved → {OUTPUT_DIR}/model_rf.joblib")

# ── FEATURE IMPORTANCE ───────────────────────────────────
fi = pd.DataFrame({
    "feature"   : feature_names,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).head(25)

print(f"\n── Top 15 features ──")
print(fi.head(15).to_string(index=False))

fig_fi = px.bar(
    fi,
    x="importance", y="feature",
    orientation="h",
    title="Feature Importance — Random Forest<br>"
          "<sup>Top 25 features predicting user success</sup>",
    color="importance",
    color_continuous_scale="Teal",
    template="plotly_dark",
    labels={"importance": "Importance", "feature": ""},
)
fig_fi.update_layout(
    yaxis=dict(autorange="reversed"),
    height=700,
    showlegend=False,
)
fig_fi.write_html(f"{OUTPUT_DIR}/04_feature_importance.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/04_feature_importance.html")

# ── MODEL COMPARISON CHART ───────────────────────────────
comp_df = metrics_df[["ROC-AUC", "Avg Prec", "Accuracy"]].reset_index()
comp_df.columns = ["Model", "ROC-AUC", "Avg Precision", "Accuracy"]
comp_melt = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

fig_comp = px.bar(
    comp_melt,
    x="Score", y="Model",
    color="Metric",
    barmode="group",
    orientation="h",
    title="Model Comparison (5-fold CV)",
    template="plotly_dark",
    color_discrete_sequence=["#00b4d8", "#90e0ef", "#caf0f8"],
    text_auto=".3f",
)
fig_comp.update_layout(height=400)
fig_comp.write_html(f"{OUTPUT_DIR}/04_model_comparison.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/04_model_comparison.html")

# ── ROC + PR CURVES ──────────────────────────────────────
# Re-train RF and LR on 80% for test curves
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

fig_roc = make_subplots(rows=1, cols=2,
                         subplot_titles=["ROC Curve", "Precision-Recall Curve"])

colors = {"Random Forest": "#00b4d8",
          "Gradient Boosting": "#90e0ef",
          "Logistic Regression": "#ff6b6b"}

for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)

    col = colors[name]
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                   line=dict(color=col), mode="lines"),
        row=1, col=1
    )
    fig_roc.add_trace(
        go.Scatter(x=rec, y=prec, name=f"{name} (AP={ap:.3f})",
                   line=dict(color=col), mode="lines", showlegend=False),
        row=1, col=2
    )

# Baseline diagonal
fig_roc.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], name="Random",
               line=dict(color="grey", dash="dash"), mode="lines"),
    row=1, col=1
)
fig_roc.update_xaxes(title_text="False Positive Rate", row=1, col=1)
fig_roc.update_yaxes(title_text="True Positive Rate",  row=1, col=1)
fig_roc.update_xaxes(title_text="Recall",     row=1, col=2)
fig_roc.update_yaxes(title_text="Precision",  row=1, col=2)
fig_roc.update_layout(template="plotly_dark", height=450,
                       title="ROC & Precision-Recall Curves (20% hold-out)")
fig_roc.write_html(f"{OUTPUT_DIR}/04_roc_pr_curves.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/04_roc_pr_curves.html")

# ── CONFUSION MATRIX ─────────────────────────────────────
rf_final = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_final.fit(X_train, y_train)
y_pred = rf_final.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=["Not Success", "Success"],
    y=["Not Success", "Success"],
    color_continuous_scale="Blues",
    title="Confusion Matrix — Random Forest",
    template="plotly_dark",
)
fig_cm.write_html(f"{OUTPUT_DIR}/04_confusion_matrix.html")
print(f"  ✓ Saved: {OUTPUT_DIR}/04_confusion_matrix.html")

print(f"\n── Classification Report (RF, 20% test) ──")
print(classification_report(y_test, y_pred, target_names=["Not Success", "Success"]))

# ── OPTIONAL: SHAP ───────────────────────────────────────
try:
    import shap
    print("\nComputing SHAP values (sample of 500)...")
    sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx]

    explainer  = shap.TreeExplainer(rf_final)
    shap_vals  = explainer.shap_values(X_sample)

    # For binary: shap_vals is list [class0, class1]
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    shap_df = pd.DataFrame(np.abs(sv).mean(axis=0),
                           index=feature_names,
                           columns=["mean_abs_shap"]).sort_values(
                               "mean_abs_shap", ascending=False).head(20)

    fig_shap = px.bar(
        shap_df.reset_index(),
        x="mean_abs_shap", y="index",
        orientation="h",
        title="SHAP Feature Importance (mean |SHAP|)",
        color="mean_abs_shap",
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    fig_shap.update_layout(yaxis=dict(autorange="reversed"), height=600)
    fig_shap.write_html(f"{OUTPUT_DIR}/04_shap_summary.html")
    print(f"  ✓ Saved: {OUTPUT_DIR}/04_shap_summary.html")

except ImportError:
    print("  (shap not installed — skipping SHAP analysis)")

print("\n✅  Modeling complete.")
