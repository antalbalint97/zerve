from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zerve · Product Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 14px !important;
            padding: 12px 14px !important;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] p,
        div[data-testid="stMetric"] span,
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
            color: #cbd5e1 !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] *,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] * {
            color: #22c55e !important;
            opacity: 1 !important;
        }

        /* Insight cards */
        .insight-box {
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            margin-bottom: 1rem !important;
            color: #e2e8f0 !important;
        }
        .insight-box, .insight-box * { color: #e2e8f0 !important; opacity: 1 !important; }
        .insight-box strong { color: #ffffff !important; }

        /* Warning callout */
        .callout-warn {
            background: #1c1508 !important;
            border: 1px solid #92400e !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            margin-bottom: 1rem !important;
        }
        .callout-warn, .callout-warn * { color: #fde68a !important; opacity: 1 !important; }
        .callout-warn strong { color: #fef3c7 !important; }

        /* Success callout */
        .callout-success {
            background: #052e16 !important;
            border: 1px solid #166534 !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            margin-bottom: 1rem !important;
        }
        .callout-success, .callout-success * { color: #bbf7d0 !important; opacity: 1 !important; }
        .callout-success strong { color: #dcfce7 !important; }

        /* North star box */
        .north-star-box {
            background: linear-gradient(135deg, #0c2340 0%, #0f172a 100%) !important;
            border: 2px solid #00b4d8 !important;
            border-radius: 16px !important;
            padding: 1.5rem 1.8rem !important;
            margin-bottom: 1.2rem !important;
        }
        .north-star-box, .north-star-box * { color: #e0f7ff !important; opacity: 1 !important; }
        .north-star-box strong { color: #ffffff !important; }

        /* Action box */
        .action-box {
            background: #0a1628 !important;
            border: 1px solid #1d4ed8 !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            margin-bottom: 0.8rem !important;
        }
        .action-box, .action-box * { color: #bfdbfe !important; opacity: 1 !important; }
        .action-box strong { color: #e0f2fe !important; }

        .section-caption {
            color: #64748b;
            font-size: 0.95rem;
            margin-top: -0.2rem;
            margin-bottom: 1.2rem;
        }
        .small-note { color: #64748b; font-size: 0.9rem; }
        .tight-list ul { margin-top: 0.25rem; margin-bottom: 0.25rem; }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

COLORS = {
    "Agent Builder": "#00b4d8",
    "Agent Runner": "#48cae4",
    "Manual Coder": "#90e0ef",
    "Viewer": "#94a3b8",
    "Ghost": "#475569",
}
SEG_ORDER = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]

T = "plotly_white"  # chart template


# ============================================================
# FILE LOADING
# ============================================================

def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def candidate_paths(base_name: str) -> list[Path]:
    return [
        OUTPUTS_DIR / f"{base_name}.parquet",
        OUTPUTS_DIR / f"{base_name}.csv",
        OUTPUTS_DIR / f"outputs_{base_name}.parquet",
        OUTPUTS_DIR / f"outputs_{base_name}.csv",
        BASE_DIR / f"{base_name}.parquet",
        BASE_DIR / f"{base_name}.csv",
    ]


def load_table(
    base_names: list[str], required: bool = False
) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    candidates: list[Path] = []
    for name in base_names:
        candidates.extend(candidate_paths(name))
    found = first_existing(candidates)
    if found is None:
        if required:
            raise FileNotFoundError(f"Required dataset not found. Tried: {candidates}")
        return None, None
    df = pd.read_parquet(found) if found.suffix == ".parquet" else pd.read_csv(found)
    return df, found


@st.cache_data(show_spinner=False)
def load_all_data() -> dict:
    feat, feat_path = load_table(
        ["user_features_segmented", "outputs_user_features", "user_features"],
        required=True,
    )

    def _l(names): return load_table(names, required=False)

    churn, churn_path = _l(["14_churn_scored_users", "churn_scored_users"])
    interv, interv_path = _l(["18_intervention_scored_users", "17_intervention_scored_users", "intervention_scored_users"])
    struggle, struggle_path = _l(["19_quality_of_struggle_scored_users", "18_quality_of_struggle_scored_users", "quality_of_struggle_scored_users"])

    kpi_seg, _ = _l(["06_kpi_by_segment"])
    kpi_cohort, _ = _l(["06_kpi_by_cohort"])
    model_cv, _ = _l(["06_model_cv_summary"])
    roc_full, _ = _l(["06_roc_curves_data"])
    fi_full, _ = _l(["06_feature_importance_full"])
    fi_narrow, _ = _l(["06_feature_importance_narrow"])

    surv_seg, _ = _l(["07_survival_by_segment_data"])
    surv_ret, _ = _l(["07_survival_by_return_data"])
    ret_cohort, _ = _l(["07_return_cohort_stats"])

    act_miles, _ = _l(["10_activation_milestones_summary"])
    ab_paths, _ = _l(["10_top_agent_builder_paths"])
    ghost_paths, _ = _l(["10_top_ghost_paths"])
    path_div, _ = _l(["10_path_divergence_data"])
    ttab_summary, _ = _l(["10_time_to_agent_builder_summary"])

    bigrams, _ = _l(["11_top_bigrams_by_segment"])
    tool_mix, _ = _l(["11_tool_mix_by_segment_data"])

    att_results, _ = _l(["12_att_results_data"])

    canvas_growth, _ = _l(["13_canvas_growth_by_segment_data"])
    canvas_repeat, _ = _l(["13_repeat_canvas_retention_melted"])

    churn_cv, _ = _l(["14_churn_cv_results"])
    churn_fi, _ = _l(["14_churn_feature_importance_data"])
    churn_roc, _ = _l(["14_churn_roc_pr_data"])
    churn_buckets, _ = _l(["14_churn_risk_bucket_summary"])

    motifs, _ = _l(["16_segment_workflow_motifs_data"])

    interv_summary, _ = _l(["18_intervention_summary"])
    interv_signals, _ = _l(["18_intervention_signal_profile_data"])

    struggle_summary, _ = _l(["19_quality_of_struggle_summary"])

    branch_top, _ = _l(["20_top_branch_points_data"])

    return dict(
        feat=feat, feat_path=feat_path,
        churn=churn, churn_path=churn_path,
        interv=interv, interv_path=interv_path,
        struggle=struggle, struggle_path=struggle_path,
        kpi_seg=kpi_seg, kpi_cohort=kpi_cohort,
        model_cv=model_cv, roc_full=roc_full,
        fi_full=fi_full, fi_narrow=fi_narrow,
        surv_seg=surv_seg, surv_ret=surv_ret, ret_cohort=ret_cohort,
        act_miles=act_miles, ab_paths=ab_paths, ghost_paths=ghost_paths,
        path_div=path_div, ttab_summary=ttab_summary,
        bigrams=bigrams, tool_mix=tool_mix,
        att_results=att_results,
        canvas_growth=canvas_growth, canvas_repeat=canvas_repeat,
        churn_cv=churn_cv, churn_fi=churn_fi, churn_roc=churn_roc,
        churn_buckets=churn_buckets,
        motifs=motifs, interv_summary=interv_summary,
        interv_signals=interv_signals,
        struggle_summary=struggle_summary, branch_top=branch_top,
    )


D = load_all_data()
feat = D["feat"]

# Convenience flags
def _ok(key): return D.get(key) is not None and isinstance(D[key], pd.DataFrame)


# ============================================================
# VALIDATION
# ============================================================

REQUIRED_COLS = [
    "segment", "days_active", "agent_tool_calls_total",
    "agent_build_calls", "had_second_session",
    "had_credit_exceeded", "signup_cohort", "ever_used_agent",
]
missing = [c for c in REQUIRED_COLS if c not in feat.columns]
if missing:
    st.error(f"Main dataset missing required columns: {missing}")
    st.stop()


# ============================================================
# PRECOMPUTE
# ============================================================

n = len(feat)

seg_stats = feat.groupby("segment").agg(
    users=("segment", "count"),
    avg_days=("days_active", "mean"),
    avg_tools=("agent_tool_calls_total", "mean"),
    avg_builds=("agent_build_calls", "mean"),
    pct_return=("had_second_session", "mean"),
    pct_credit=("had_credit_exceeded", "mean"),
).reindex([s for s in SEG_ORDER if s in feat["segment"].unique()])
seg_stats["pct_of_users"] = (seg_stats["users"] / n * 100).round(1)
seg_stats["pct_return"] = (seg_stats["pct_return"] * 100).round(1)
seg_stats["pct_credit"] = (seg_stats["pct_credit"] * 100).round(1)

cohort_stats = feat.groupby("signup_cohort").agg(
    users=("segment", "count"),
    pct_ab=("segment", lambda x: round((x == "Agent Builder").mean() * 100, 1)),
    pct_agent=("ever_used_agent", lambda x: round(x.mean() * 100, 1)),
    avg_tools=("agent_tool_calls_total", "mean"),
).sort_index()

ab_users = int((feat["segment"] == "Agent Builder").sum())
ghost_users = int((feat["segment"] == "Ghost").sum())
pct_agent = round(feat["ever_used_agent"].mean() * 100, 1)
pct_return = round(feat["had_second_session"].mean() * 100, 1)
pct_early = (
    round(feat["adopted_agent_early"].mean() * 100, 1)
    if "adopted_agent_early" in feat.columns else None
)

churn = D["churn"]
has_churn = _ok("churn")
churn_rate = None
if has_churn and "is_14d_survival_churn_proxy" in churn.columns:
    churn_rate = round(churn["is_14d_survival_churn_proxy"].mean() * 100, 1)
elif has_churn and "churn_probability" in churn.columns:
    churn_rate = round((churn["churn_probability"] > 0.5).mean() * 100, 1)

interv = D["interv"]
has_interv = _ok("interv")
struggle = D["struggle"]
has_struggle = _ok("struggle")


# ============================================================
# FIGURE BUILDERS
# ============================================================

def _layout(fig, title="", h=420, **kw):
    fig.update_layout(
        title=title, template=T, height=h,
        margin=dict(t=60 if title else 30, l=20, r=20, b=30),
        **kw,
    )
    return fig


# --- Overview / Segment ---

def fig_activation_funnel() -> go.Figure:
    ran_tool = int((feat["agent_tool_calls_total"] > 0).sum())
    ever_agent = int(feat["ever_used_agent"].sum())
    stages = ["Registered", "Ran ≥1 tool", "Used AI agent", "Agent Builder"]
    vals = [n, ran_tool, ever_agent, ab_users]
    pcts = [100, ran_tool/n*100, ever_agent/n*100, ab_users/n*100]
    bar_colors = ["#1e3a5f", "#0369a1", "#0284c7", "#00b4d8"]
    fig = go.Figure(go.Bar(
        x=stages, y=vals,
        text=[f"{v:,}<br>{p:.1f}%" for v, p in zip(vals, pcts)],
        textposition="outside",
        marker_color=bar_colors,
        textfont=dict(size=12),
    ))
    return _layout(fig, "Activation funnel - where users drop off", h=380,
                   yaxis_title="Users", yaxis_range=[0, n * 1.18])


def fig_segment_pie() -> go.Figure:
    segs = list(seg_stats.index)
    fig = go.Figure(go.Pie(
        labels=segs, values=seg_stats["users"],
        hole=0.48,
        marker_colors=[COLORS.get(s, "#888") for s in segs],
        textinfo="label+percent",
        showlegend=False,
    ))
    return _layout(fig, "User distribution by segment", h=380)


def fig_segment_bars() -> go.Figure:
    segs = list(seg_stats.index)
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Avg active days", "Avg tool calls", "Return rate (%)"])
    for col_idx, (col, fmt) in enumerate([
        ("avg_days", ".1f"), ("avg_tools", ".0f"), ("pct_return", ".0f")
    ], 1):
        vals = seg_stats[col].round(1)
        fig.add_trace(go.Bar(
            x=segs, y=vals,
            marker_color=[COLORS.get(s, "#888") for s in segs],
            text=[f"{v:{fmt}}" for v in vals],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
    return _layout(fig, "Behavioral profile by segment", h=380)


# --- Cohorts ---

def fig_cohort() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=cohort_stats.index, y=cohort_stats["users"],
        name="Users", marker_color="#1a3040", opacity=0.85,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=cohort_stats.index, y=cohort_stats["pct_ab"],
        name="Agent Builder %", mode="lines+markers",
        line=dict(color="#00b4d8", width=3), marker=dict(size=9),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=cohort_stats.index, y=cohort_stats["pct_agent"],
        name="Ever used agent %", mode="lines+markers",
        line=dict(color="#90e0ef", width=2, dash="dot"), marker=dict(size=7),
    ), secondary_y=True)
    fig.update_yaxes(title_text="Users signed up", secondary_y=False)
    fig.update_yaxes(title_text="Adoption rate (%)", secondary_y=True)
    fig.update_layout(
        title="Signup cohort performance - the Nov–Dec collapse",
        template=T, height=420,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=60, l=20, r=20, b=60),
    )
    return fig


# --- Retention & Survival ---

def fig_return_rate() -> go.Figure:
    segs = list(seg_stats.index)
    fig = go.Figure(go.Bar(
        x=segs, y=seg_stats["pct_return"],
        marker_color=[COLORS.get(s, "#888") for s in segs],
        text=[f"{v:.0f}%" for v in seg_stats["pct_return"]],
        textposition="outside",
    ))
    return _layout(fig, "Second-session return rate by segment", h=360,
                   yaxis_title="Return rate (%)", yaxis_range=[0, 115])


def fig_survival_segment() -> go.Figure:
    df = D.get("surv_seg")
    if df is None:
        return go.Figure().update_layout(title="Survival data unavailable", template=T, height=360)
    fig = go.Figure()
    palette = {"Agent Builder": "#00b4d8", "Manual Coder": "#90e0ef",
               "Viewer": "#94a3b8", "Ghost": "#475569"}
    for grp, gdf in df.groupby("group"):
        gdf = gdf.sort_values("timeline")
        fig.add_trace(go.Scatter(
            x=gdf["timeline"], y=gdf["survival"] * 100,
            mode="lines", name=grp,
            line=dict(color=palette.get(grp, "#888"), width=2.5),
        ))
    return _layout(fig, "Kaplan-Meier survival by segment (days since signup)", h=400,
                   xaxis_title="Days", yaxis_title="Active users (%)",
                   legend=dict(orientation="h", y=-0.22))


def fig_survival_return() -> go.Figure:
    df = D.get("surv_ret")
    if df is None:
        return go.Figure().update_layout(title="Return survival data unavailable", template=T, height=360)
    fig = go.Figure()
    palette = {
        "Returned quickly (<=24h)": "#00b4d8",
        "Returned (>24h)": "#90e0ef",
        "Never returned": "#475569",
    }
    for grp, gdf in df.groupby("group"):
        gdf = gdf.sort_values("timeline")
        fig.add_trace(go.Scatter(
            x=gdf["timeline"], y=gdf["survival"] * 100,
            mode="lines", name=grp,
            line=dict(color=palette.get(grp, "#888"), width=2.5),
        ))
    return _layout(fig, "Survival by return timing - early return predicts staying", h=400,
                   xaxis_title="Days", yaxis_title="Active users (%)",
                   legend=dict(orientation="h", y=-0.22))


def fig_activation_milestones() -> go.Figure:
    df = D.get("act_miles")
    if df is None:
        return go.Figure().update_layout(title="Milestone data unavailable", template=T, height=360)
    df = df.sort_values("lift", ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        y=df["milestone"].str.replace("_", " "),
        x=df["lift"],
        orientation="h",
        marker_color="#00b4d8",
        text=[f"{v:.1f}x lift" for v in df["lift"]],
        textposition="outside",
    ))
    return _layout(fig, "Activation milestone lift → P(Agent Builder | milestone reached)", h=400,
                   xaxis_title="Lift vs baseline")


# --- Modeling ---

def fig_roc_full() -> go.Figure:
    df = D.get("roc_full")
    if df is None:
        return go.Figure().update_layout(title="ROC data unavailable", template=T, height=380)
    fig = go.Figure()
    palette = {"Random Forest": "#00b4d8", "Gradient Boosting": "#7c3aed",
               "Logistic Regression": "#f59e0b"}
    for grp, gdf in df.groupby("model"):
        if "population" in df.columns:
            gdf = gdf[gdf["population"] == "full"] if "full" in gdf["population"].values else gdf
        gdf = gdf.sort_values("fpr")
        auc_val = gdf["auc"].iloc[0]
        fig.add_trace(go.Scatter(
            x=gdf["fpr"], y=gdf["tpr"],
            mode="lines", name=f"{grp} (AUC={auc_val:.3f})",
            line=dict(color=palette.get(grp, "#888"), width=2.5),
        ))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#94a3b8", dash="dash", width=1))
    return _layout(fig, "ROC curves - Agent Builder prediction model (full population)", h=420,
                   xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                   legend=dict(orientation="h", y=-0.22))


def fig_feature_importance() -> go.Figure:
    fi_f = D.get("fi_full")
    fi_n = D.get("fi_narrow")
    if fi_f is None and fi_n is None:
        return go.Figure().update_layout(title="Feature importance unavailable", template=T, height=380)

    def _prep(df, n=10):
        return df.nlargest(n, "importance").sort_values("importance")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Full population model", "Narrowed model (first-session signals only)"])

    if fi_f is not None:
        f = _prep(fi_f)
        fig.add_trace(go.Bar(
            y=f["feature"].str.replace("_", " "), x=f["importance"],
            orientation="h", marker_color="#00b4d8",
            name="Full", showlegend=False,
        ), row=1, col=1)

    if fi_n is not None:
        f = _prep(fi_n)
        fig.add_trace(go.Bar(
            y=f["feature"].str.replace("_", " "), x=f["importance"],
            orientation="h", marker_color="#7c3aed",
            name="Narrow", showlegend=False,
        ), row=1, col=2)

    return _layout(fig, "Feature importance - what predicts Agent Builder status", h=440)


# --- Churn ---

def fig_churn_distribution() -> go.Figure:
    if not has_churn:
        return go.Figure().update_layout(title="Churn data unavailable", template=T, height=360)
    df = D["churn"]
    if "churn_risk_bucket" not in df.columns or "churn_probability" not in df.columns:
        return go.Figure().update_layout(title="Churn columns missing", template=T, height=360)
    risk = df.groupby("churn_risk_bucket").agg(
        users=("churn_probability", "count"),
        avg_prob=("churn_probability", "mean"),
    ).reset_index()
    order = ["Low", "Medium", "High", "Critical"]
    risk["churn_risk_bucket"] = pd.Categorical(risk["churn_risk_bucket"], categories=order, ordered=True)
    risk = risk.sort_values("churn_risk_bucket")
    colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#f97316", "Critical": "#ef4444"}
    title = "14-day churn risk distribution"
    if churn_rate is not None:
        title += f"  ·  overall active-user rate: {churn_rate}%"
    fig = go.Figure(go.Bar(
        x=risk["churn_risk_bucket"], y=risk["users"],
        marker_color=[colors.get(v, "#888") for v in risk["churn_risk_bucket"]],
        text=risk["users"], textposition="outside",
    ))
    return _layout(fig, title, h=380, yaxis_title="Users")


def fig_churn_roc() -> go.Figure:
    df = D.get("churn_roc")
    if df is None:
        return go.Figure().update_layout(title="Churn ROC unavailable", template=T, height=380)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC Curve", "Precision-Recall Curve"])
    palette = {"Random Forest": "#00b4d8", "Logistic Regression": "#f59e0b"}
    for model, mdf in df.groupby("model"):
        roc = mdf[mdf["curve"] == "roc"].sort_values("x")
        pr = mdf[mdf["curve"] == "pr"].sort_values("x")
        auc_r = roc["auc_or_ap"].iloc[0] if len(roc) > 0 else 0
        ap_r = pr["auc_or_ap"].iloc[0] if len(pr) > 0 else 0
        clr = palette.get(model, "#888")
        if len(roc):
            fig.add_trace(go.Scatter(x=roc["x"], y=roc["y"], mode="lines",
                name=f"{model} (AUC={auc_r:.3f})", line=dict(color=clr, width=2.5)), row=1, col=1)
        if len(pr):
            fig.add_trace(go.Scatter(x=pr["x"], y=pr["y"], mode="lines",
                name=f"{model} (AP={ap_r:.3f})", line=dict(color=clr, width=2.5, dash="dot"),
                showlegend=False), row=1, col=2)
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, row=1, col=1,
                  line=dict(color="#94a3b8", dash="dash", width=1))
    return _layout(fig, "Churn model performance", h=420,
                   legend=dict(orientation="h", y=-0.22))


def fig_churn_fi() -> go.Figure:
    df = D.get("churn_fi")
    if df is None:
        return go.Figure().update_layout(title="Churn feature importance unavailable", template=T, height=360)
    df = df.nlargest(12, "importance").sort_values("importance")
    fig = go.Figure(go.Bar(
        y=df["feature"].str.replace("_", " "), x=df["importance"],
        orientation="h", marker_color="#ef4444",
        text=df["importance"].round(3), textposition="outside",
    ))
    return _layout(fig, "Churn model - top predictors of 14-day churn", h=420,
                   xaxis_title="Feature importance")


# --- Interventions ---

def fig_intervention_mix() -> go.Figure:
    if not has_interv:
        return go.Figure().update_layout(title="Intervention data unavailable", template=T, height=360)
    if "recommended_intervention" not in interv.columns:
        return go.Figure().update_layout(title="Intervention column missing", template=T, height=360)
    iv = interv.groupby("recommended_intervention").size().reset_index(name="users")
    iv = iv.sort_values("users", ascending=True)
    color_map = {
        "Activation nudge": "#00b4d8", "Retention rescue": "#f97316",
        "Monitor": "#94a3b8", "Builder acceleration": "#22c55e",
        "Productive struggle support": "#7c3aed",
    }
    fig = go.Figure(go.Bar(
        x=iv["users"], y=iv["recommended_intervention"],
        orientation="h",
        marker_color=[color_map.get(v, "#888") for v in iv["recommended_intervention"]],
        text=iv["users"], textposition="outside",
    ))
    return _layout(fig, "Recommended interventions - already scored and ready to deploy", h=360,
                   xaxis_title="Users")


def fig_intervention_signals() -> go.Figure:
    df = D.get("interv_signals")
    if df is None:
        return go.Figure().update_layout(title="Signal profile data unavailable", template=T, height=360)
    pivot = df.pivot(index="recommended_intervention", columns="metric", values="value")
    pivot = pivot[[c for c in pivot.columns if "avg" in c]]
    pivot.columns = [c.replace("avg_", "").replace("_", " ") for c in pivot.columns]
    pivot = pivot.fillna(0)
    color_map = {
        "Activation nudge": "#00b4d8", "Retention rescue": "#f97316",
        "Monitor": "#94a3b8", "Builder acceleration": "#22c55e",
        "Productive struggle support": "#7c3aed",
    }
    fig = go.Figure()
    for metric in pivot.columns:
        fig.add_trace(go.Bar(
            name=metric, x=list(pivot.index), y=pivot[metric],
            text=pivot[metric].round(2), textposition="outside",
        ))
    return _layout(fig, "Signal profile by intervention type", h=400,
                   yaxis_title="Score", barmode="group",
                   legend=dict(orientation="h", y=-0.28))


# --- Struggle & Recovery ---

def fig_struggle_distribution() -> go.Figure:
    df = D.get("struggle_summary")
    if df is None:
        if not has_struggle:
            return go.Figure().update_layout(title="Struggle data unavailable", template=T, height=360)
        df_raw = struggle
        col = next((c for c in ["struggle_class", "quality_of_struggle_class", "recommended_support"]
                    if c in df_raw.columns), None)
        if col is None:
            return go.Figure().update_layout(title="Struggle class column missing", template=T, height=360)
        df = df_raw.groupby(col).size().reset_index(name="users").rename(columns={col: "struggle_class"})

    color_map = {
        "Productive struggle": "#22c55e", "Abandonment-prone struggle": "#ef4444",
        "Mixed/uncertain struggle": "#f59e0b", "No visible struggle": "#94a3b8",
    }
    fig = go.Figure(go.Bar(
        x=df["struggle_class"].str.replace(" struggle", "").str.replace("/", "/\n"),
        y=df["users"],
        marker_color=[color_map.get(v, "#888") for v in df["struggle_class"]],
        text=df["users"], textposition="outside",
    ))
    return _layout(fig, "Quality of struggle - not all struggle predicts abandonment", h=380,
                   yaxis_title="Users")


# --- Workflow & Paths ---

def fig_path_divergence() -> go.Figure:
    df = D.get("path_div")
    if df is None:
        return go.Figure().update_layout(title="Path divergence data unavailable", template=T, height=380)
    segs_to_show = [s for s in ["Agent Builder", "Ghost"] if s in df["segment"].unique()]
    if not segs_to_show:
        segs_to_show = df["segment"].unique()[:2]
    fig = go.Figure()
    palette = {"Agent Builder": "#00b4d8", "Ghost": "#475569",
               "Manual Coder": "#90e0ef", "Viewer": "#94a3b8"}
    for seg in segs_to_show:
        sdf = df[df["segment"] == seg].sort_values("step")
        fig.add_trace(go.Bar(
            name=seg, x=[f"Step {r['step']}: {r['event']}" for _, r in sdf.iterrows()],
            y=sdf["pct"],
            marker_color=palette.get(seg, "#888"),
            text=sdf["pct"].round(1), textposition="outside",
        ))
    return _layout(fig, "Early path divergence - Agent Builders vs Ghosts", h=420,
                   yaxis_title="% of segment", barmode="group",
                   legend=dict(orientation="h", y=-0.22))


def fig_tool_mix() -> go.Figure:
    df = D.get("tool_mix")
    if df is None:
        return go.Figure().update_layout(title="Tool mix data unavailable", template=T, height=380)
    segs = [s for s in ["Agent Builder", "Manual Coder", "Viewer"] if s in df["segment"].unique()]
    fig = go.Figure()
    palette = {"Agent Builder": "#00b4d8", "Manual Coder": "#90e0ef", "Viewer": "#94a3b8"}
    for seg in segs:
        sdf = df[df["segment"] == seg].sort_values("pct", ascending=False).head(8)
        fig.add_trace(go.Bar(
            name=seg, x=sdf["tool_short"], y=sdf["pct"],
            marker_color=palette.get(seg, "#888"),
        ))
    return _layout(fig, "Tool usage mix by segment (%)", h=400,
                   yaxis_title="% of segment tool calls", barmode="group",
                   legend=dict(orientation="h", y=-0.22))


def fig_canvas_repeat() -> go.Figure:
    df = D.get("canvas_repeat")
    if df is None:
        return go.Figure().update_layout(title="Canvas repeat data unavailable", template=T, height=360)
    metric_map = {
        "avg_days_active": "Avg active days",
        "repeat_session_pct": "Repeat session %",
        "agent_builder_pct": "Agent Builder %",
    }
    fig = go.Figure()
    color_map = {"One-off canvas": "#94a3b8", "Repeat canvas": "#00b4d8"}
    for metric, label in metric_map.items():
        if metric not in df["metric"].values:
            continue
        mdf = df[df["metric"] == metric]
        for _, row in mdf.iterrows():
            fig.add_trace(go.Bar(
                name=f"{row['repeat_canvas_label']} - {label}",
                x=[label], y=[row["value"]],
                marker_color=color_map.get(row["repeat_canvas_label"], "#888"),
                showlegend=True,
                legendgroup=row["repeat_canvas_label"],
            ))
    return _layout(fig, "Repeat canvas users vs one-off: engagement gap", h=380,
                   barmode="group", legend=dict(orientation="h", y=-0.22))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Zerve Analytics")
    st.caption("Zerve × HackerEarth Hackathon 2026")
    st.markdown("---")
    st.markdown("### Analysis scope")
    st.markdown(f"- **{n:,} users** analyzed")
    st.markdown("- Signup cohorts: Sep–Dec 2025")
    st.markdown("- Dataset end: Dec 8, 2025")
    st.markdown("---")
    st.markdown("### Data sources")
    st.write(f"Features: `{D['feat_path'].name}`")
    if D.get("churn_path"): st.write(f"Churn: `{D['churn_path'].name}`")
    if D.get("interv_path"): st.write(f"Interventions: `{D['interv_path'].name}`")
    if D.get("struggle_path"): st.write(f"Struggle: `{D['struggle_path'].name}`")
    st.markdown("---")
    st.markdown(
        '<div class="small-note">Presentation layer over the Zerve analytics pipeline.<br>'
        'No data is recomputed at runtime.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("## Zerve · Product Analytics")
st.markdown(
    '<div class="section-caption">Zerve × HackerEarth Hackathon 2026 &nbsp;·&nbsp; '
    'What drives successful usage on Zerve?</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total users analyzed", f"{n:,}")
m2.metric("Agent Builder", f"{ab_users:,}", f"{ab_users/n*100:.1f}% of base")
m3.metric("Ghost users", f"{ghost_users:,}", f"{ghost_users/n*100:.1f}% of base")
m4.metric("Ever used the agent", f"{pct_agent}%")
m5.metric("Second-session return", f"{pct_return}%")

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("14-day churn rate", f"{churn_rate}%" if churn_rate else "N/A")
m7.metric("Early adopters (<1h)", f"{pct_early}%" if pct_early else "N/A")
m8.metric("Sep–Oct AB rate", "~19.5%")
m9.metric("Nov–Dec AB rate", "0.0% ⚠")
m10.metric("North star baseline", f"{pct_agent}%")

st.markdown("---")


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.subheader("Executive summary")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="insight-box">
        <strong>The product works. Most users never reach it.</strong><br><br>
        Agent Builders - the 6% who use the AI agent to construct pipelines - show 90% second-session
        return and average 157 tool interactions. The product delivers real value. But 63.7% of users
        are Ghosts who never ran a single tool. The activation funnel is broken, not the product.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="callout-warn">
        <strong>⚠ Critical signal: Nov–Dec cohort collapse</strong><br><br>
        September and October 2025 cohorts hit ~19.5% Agent Builder adoption. November and December
        show <strong>0.0%</strong> across 3,300 users. This is not a trend - it is a regression.
        Root-cause analysis is the highest-priority product action.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <div class="callout-success">
        <strong>✓ Intervention targets are already scored</strong><br><br>
        3,393 users flagged for activation nudges. 846 for retention rescue. 8 for builder
        acceleration. The targeting infrastructure exists and the users are identified.
        Deployment is the only remaining step.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Overview",
    "Segments",
    "Cohorts",
    "Retention & Survival",
    "Modeling",
    "Churn",
    "Interventions",
    "Struggle & Recovery",
    "Workflow & Paths",
])

tab_overview, tab_segs, tab_cohorts, tab_ret, tab_model, tab_churn, tab_interv, tab_struggle, tab_workflow = tabs


# ─── TAB 1: OVERVIEW ────────────────────────────────────────────────────────

with tab_overview:
    st.markdown("### The activation bottleneck")

    left, right = st.columns([1.3, 1])

    with left:
        st.plotly_chart(fig_activation_funnel(), key="funnel", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="insight-box">
            <strong>This is primarily an activation problem.</strong><br><br>
            The funnel collapses before users reach Zerve's core value. Of 4,771 users:
            <ul>
            <li>Only <strong>~36%</strong> ran at least one tool.</li>
            <li>Only <strong>13.6%</strong> ever used the AI agent.</li>
            <li>Only <strong>6.0%</strong> reached Agent Builder status - consistent,
            meaningful agent-assisted pipeline work.</li>
            </ul>
            The gap between "registered" and "experienced the product" is the dominant
            problem in this dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-box">
            <strong>The agent is not a feature. It is the retention mechanism.</strong><br><br>
            Among users who never reach the agent: 8% return for a second session.
            Among Agent Builders: 90%. The agent is the inflection point where the platform
            creates habits rather than single visits.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Return cohort comparison")
    rc = D.get("ret_cohort")
    if rc is not None:
        st.dataframe(
            rc.rename(columns={
                "return_cohort": "Return cohort", "users": "Users",
                "pct_ab": "Agent Builder %", "avg_days": "Avg active days"
            }).round(2),
            use_container_width=True,
        )
        st.markdown(
            '<div class="small-note">Users who returned within 24h have a 22.8% Agent Builder rate '
            'vs 0.8% among users who never returned - early return is the clearest leading '
            'indicator of eventual success.</div>',
            unsafe_allow_html=True,
        )


# ─── TAB 2: SEGMENTS ────────────────────────────────────────────────────────

with tab_segs:
    st.markdown("### Segment behavioral profiles")
    st.markdown(
        "Agent Builder is not just the best segment - it is categorically different "
        "from every other group in the dataset."
    )

    left, right = st.columns([1, 1.1])
    with left:
        st.plotly_chart(fig_segment_pie(), key="seg_pie", use_container_width=True)
    with right:
        st.plotly_chart(fig_return_rate(), key="seg_return", use_container_width=True)

    st.plotly_chart(fig_segment_bars(), key="seg_bars", use_container_width=True)

    st.markdown("#### Full behavioral comparison table")

    kpi = D.get("kpi_seg")
    if kpi is not None:
        display_kpi = kpi.rename(columns={
            "segment": "Segment", "users": "Users",
            "avg_days_active": "Avg active days", "avg_agent_tools": "Avg tool calls",
            "avg_agent_build": "Avg build calls", "pct_multi_day": "Multi-day %",
            "pct_week_plus": "Week+ active %", "pct_credit_exceeded": "Credit exceeded %",
            "pct_ever_agent": "Ever used agent %", "pct_early_adopter": "Early adopter %",
            "pct_of_users": "Share of users %",
        })
        cols_show = [c for c in [
            "Segment", "Users", "Share of users %", "Avg active days",
            "Avg tool calls", "Avg build calls", "Ever used agent %",
            "Multi-day %", "Credit exceeded %"
        ] if c in display_kpi.columns]
        st.dataframe(display_kpi[cols_show].round(1), use_container_width=True)
    else:
        seg_display = seg_stats.rename(columns={
            "users": "Users", "avg_days": "Avg active days",
            "avg_tools": "Avg tool calls", "avg_builds": "Avg build calls",
            "pct_return": "Return %", "pct_credit": "Credit exceeded %",
            "pct_of_users": "Share of users %",
        })
        st.dataframe(seg_display.round(2), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Agent Builder: the success state</strong><br><br>
            285 users (6.0%) · 157 avg tool calls · 4.75 avg active days ·
            90% second-session return · 6% credit exceeded (platform is being pushed hard).
            This is what successful engagement looks like on Zerve.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Ghost: a pre-activation failure, not a retention problem</strong><br><br>
            3,037 users (63.7%) · 0.00 avg tool calls · 8% second-session return.
            These users registered and never started. No retention strategy reaches them -
            the intervention must happen at or before the moment of first use.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ATT causal results
    att = D.get("att_results")
    if att is not None:
        st.markdown("#### Causal effect of agent adoption (propensity-matched ATT)")
        st.dataframe(
            att.rename(columns={
                "metric": "Metric", "diff": "Causal lift",
                "treated": "Agent users", "control": "Non-agent users"
            }).round(2),
            use_container_width=True,
        )
        st.markdown(
            '<div class="small-note">Propensity score matching controls for observable '
            'confounders. The "lift" column is the average treatment effect on the treated (ATT).</div>',
            unsafe_allow_html=True,
        )


# ─── TAB 3: COHORTS ─────────────────────────────────────────────────────────

with tab_cohorts:
    st.markdown("### Signup cohort analysis")
    st.plotly_chart(fig_cohort(), key="cohort_main", use_container_width=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        kc = D.get("kpi_cohort") if D.get("kpi_cohort") is not None else None
        display_c = kc if kc is not None else cohort_stats.reset_index()
        rename_map = {
            "signup_cohort": "Cohort", "users": "Users",
            "avg_days_active": "Avg active days",
            "avg_agent_tools": "Avg tool calls",
            "pct_agent_builder": "Agent Builder %",
            "pct_ever_agent": "Ever used agent %",
            "pct_ghost": "Ghost %",
            # fallback names from computed cohort_stats
            "pct_ab": "Agent Builder %",
            "pct_agent": "Ever used agent %",
            "avg_tools": "Avg tool calls",
        }
        display_c = display_c.rename(columns=rename_map)
        st.dataframe(display_c.round(1), use_container_width=True)

    with c2:
        st.markdown(
            """
            <div class="callout-warn">
            <strong>⚠ Nov–Dec adoption collapse: immediate investigation required</strong><br><br>
            Sep 2025: <strong>19.1%</strong> Agent Builder rate (999 users)<br>
            Oct 2025: <strong>19.9%</strong> (472 users)<br>
            Nov 2025: <strong>0.0%</strong> (2,000 users)<br>
            Dec 2025: <strong>0.0%</strong> (1,300 users)<br><br>
            Possible causes: UI change that buried the agent, feature regression,
            shift in acquisition channel, or observation-window bias for very recent users.
            All must be ruled out before any growth initiative targets these cohorts.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="insight-box">
            <strong>Observation-window caveat</strong><br><br>
            Nov–Dec users had fewer days to accumulate agent build calls before the dataset
            ended on Dec 8, 2025. This partially explains the zero rate - but the September
            and October cohorts had 19%+ Agent Builder rates well within their first weeks,
            so the observation window alone does not explain the collapse.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── TAB 4: RETENTION & SURVIVAL ────────────────────────────────────────────

with tab_ret:
    st.markdown("### Retention, survival, and activation milestones")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_survival_segment(), key="surv_seg", use_container_width=True)
        st.markdown(
            '<div class="small-note">Agent Builders decay more slowly but still churn - '
            'the platform creates strong early engagement but has not yet built durable habits.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.plotly_chart(fig_survival_return(), key="surv_ret", use_container_width=True)
        st.markdown(
            '<div class="small-note">Users who returned within 24h survive dramatically longer. '
            'Early return timing is one of the strongest behavioral signals in the dataset.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Activation milestones - what predicts Agent Builder status")
    col3, col4 = st.columns([1.3, 1])
    with col3:
        st.plotly_chart(fig_activation_milestones(), key="act_miles", use_container_width=True)
    with col4:
        am = D.get("act_miles")
        if am is not None:
            st.dataframe(
                am.rename(columns={
                    "milestone": "Milestone", "ab_if_did": "AB% if reached",
                    "ab_if_not": "AB% if not", "lift": "Lift",
                    "pct_users": "% users reached", "n_did": "Users",
                }).round(1),
                use_container_width=True,
            )
            best = am.loc[am["lift"].idxmax()]
            st.markdown(
                f"""
                <div class="callout-success">
                <strong>Strongest activation milestone</strong><br><br>
                <strong>{best['milestone'].replace('_', ' ')}</strong><br>
                Lift: {best['lift']:.1f}x · Reached by {best['pct_users']:.1f}% of users<br>
                AB rate if reached: {best['ab_if_did']:.1f}%  vs  baseline: {best['ab_if_not']:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Time to Agent Builder
    ttab = D.get("ttab_summary")
    if ttab is not None:
        st.markdown("#### Time to Agent Builder status (hours from signup)")
        ttab_d = ttab.set_index("stat")["value"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median", f"{ttab_d.get('50%', ttab_d.get('median', 'N/A')):.1f}h"
                  if isinstance(ttab_d.get('50%', ttab_d.get('median')), float) else "N/A")
        c2.metric("Mean", f"{ttab_d.get('mean', 'N/A'):.1f}h"
                  if isinstance(ttab_d.get('mean'), float) else "N/A")
        c3.metric("75th pct", f"{ttab_d.get('75%', 'N/A'):.1f}h"
                  if isinstance(ttab_d.get('75%'), float) else "N/A")
        c4.metric("Users w/ data", f"{int(ttab_d.get('count', 0)):,}")


# ─── TAB 5: MODELING ────────────────────────────────────────────────────────

with tab_model:
    st.markdown("### Predictive modeling - Agent Builder classification")
    st.markdown(
        "Three models were evaluated for predicting Agent Builder status. "
        "The full-population model is highly accurate; the narrowed model (first-session signals only) "
        "is more product-actionable - it identifies at-risk users early enough to intervene."
    )

    # Model comparison table
    mc = D.get("model_cv")
    if mc is not None:
        st.markdown("#### Cross-validated model comparison")
        display_mc = mc.rename(columns={
            "model": "Model", "population": "Population",
            "auc_mean": "AUC (mean)", "auc_std": "AUC (±std)",
            "avg_precision_mean": "Avg Precision (mean)", "avg_precision_std": "Avg Precision (±std)",
        })
        st.dataframe(display_mc.round(4), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_roc_full(), key="roc_full", use_container_width=True)
    with c2:
        st.plotly_chart(fig_feature_importance(), key="feat_imp", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Full model: near-perfect discrimination (AUC ≈ 0.99)</strong><br><br>
            With all behavioral signals, Agent Builder status is almost perfectly predictable.
            The model is effectively learning the definition: users who use the agent more are
            more likely to be Agent Builders. This validates the segment construction but
            is not directly actionable for early intervention.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Narrowed model: product-actionable signals</strong><br><br>
            When restricted to first-session signals - breadth of event types explored,
            time to return, session duration - the model still performs well while using
            only information available within hours of signup. These are the signals that
            should drive onboarding personalization and early-session nudge timing.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── TAB 6: CHURN ───────────────────────────────────────────────────────────

with tab_churn:
    st.markdown("### Churn analysis - active-user 14-day proxy")
    st.markdown(
        "The 14-day churn proxy among active users stands at **83.3%**. "
        "Most churn is silent: users stop coming back without triggering any visible struggle signal."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_churn_distribution(), key="churn_dist", use_container_width=True)
    with c2:
        cb = D.get("churn_buckets")
        if cb is not None:
            st.markdown("#### Risk bucket breakdown")
            st.dataframe(
                cb.rename(columns={
                    "churn_risk_bucket": "Risk bucket",
                    "users": "Users",
                    "actual_churn_pct": "Actual churn %",
                    "avg_days_active": "Avg active days",
                    "avg_canvas_complexity": "Avg canvas complexity",
                }).round(1),
                use_container_width=True,
            )
            st.markdown(
                '<div class="small-note">Critical bucket: 100% actual churn rate, '
                'avg 1.2 active days. Low bucket: 0% churn, avg 21 active days - '
                'canvas complexity strongly separates these groups.</div>',
                unsafe_allow_html=True,
            )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_churn_roc(), key="churn_roc", use_container_width=True)
    with c4:
        st.plotly_chart(fig_churn_fi(), key="churn_fi", use_container_width=True)

    # Churn model summary
    cc = D.get("churn_cv")
    if cc is not None:
        st.markdown("#### Churn model cross-validation")
        st.dataframe(
            cc.rename(columns={
                "model": "Model", "auc_mean": "AUC (mean)", "auc_std": "AUC (±std)",
                "ap_mean": "Avg Precision", "ap_std": "AP (±std)",
            }).round(4),
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="insight-box">
        <strong>Churn insight: time to return and canvas engagement are the key drivers</strong><br><br>
        The top churn predictors are time to return (slow returners churn), average canvas active days
        (users who revisit canvases stay), and agent message count (agent users retain better).
        These signals are all measurable in near-real-time and can drive targeted rescue workflows.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── TAB 7: INTERVENTIONS ───────────────────────────────────────────────────

with tab_interv:
    st.markdown("### Intervention targeting - scored and ready to deploy")
    st.markdown(
        "Behavioral scoring has identified specific users for each intervention type. "
        "These are not estimates - they are scored individual users with computed priority values."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(fig_intervention_mix(), key="interv_mix", use_container_width=True)
    with c2:
        is_df = D.get("interv_summary")
        if is_df is not None:
            st.markdown("#### Intervention summary by type")
            disp = is_df.rename(columns={
                "recommended_intervention": "Intervention",
                "users": "Users",
                "avg_priority": "Avg priority",
                "avg_churn_risk": "Avg churn risk",
                "avg_activation": "Avg activation",
                "avg_struggle": "Avg struggle",
                "avg_builder_momentum": "Avg builder momentum",
                "pct_agent_builder": "% Agent Builders",
            })
            st.dataframe(disp.round(3), use_container_width=True)

    st.plotly_chart(fig_intervention_signals(), key="interv_sig", use_container_width=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Activation nudge: 3,393 users</strong><br><br>
            Low churn risk, low activation signal. These users registered but never engaged.
            In-app guided prompts or email flows showing "what the agent can do" are the
            appropriate intervention - ideally triggering within 24h of signup.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Retention rescue: 846 users</strong><br><br>
            High churn risk, some prior engagement. These users showed intent but are drifting.
            Re-engagement flows anchored to their most recent canvas or unfinished pipeline
            have the best chance of recovery.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Builder acceleration: 8 users</strong><br><br>
            High builder momentum, high priority. These are users on the cusp of consistent
            Agent Builder behavior. Pro-tier feature unlocks, 1:1 outreach, or advanced
            use-case walkthroughs could convert them permanently.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top intervention candidates preview
    top_c = D.get("interv")
    if top_c is not None and "recommended_intervention" in top_c.columns:
        with st.expander("Show top-scored intervention candidates (sample)"):
            preview_cols = [c for c in [
                "person_id", "segment", "recommended_intervention",
                "intervention_priority_score", "churn_probability",
                "activation_signal", "builder_momentum_signal",
                "days_active", "agent_tool_calls_total",
            ] if c in top_c.columns]
            sample = top_c.sort_values("intervention_priority_score", ascending=False).head(15)
            st.dataframe(sample[preview_cols].round(3), use_container_width=True)


# ─── TAB 8: STRUGGLE & RECOVERY ─────────────────────────────────────────────

with tab_struggle:
    st.markdown("### Quality of struggle - not all friction predicts abandonment")
    st.markdown(
        "Struggle in a product is only a problem if it leads to abandonment. "
        "Users who push through difficulty and recover often become the most engaged. "
        "The key is distinguishing productive struggle from pre-abandonment signals."
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(fig_struggle_distribution(), key="struggle_dist", use_container_width=True)
    with c2:
        ss = D.get("struggle_summary")
        if ss is not None:
            st.markdown("#### Struggle class profiles")
            disp = ss.rename(columns={
                "struggle_class": "Class", "users": "Users",
                "avg_quality_score": "Quality score",
                "avg_abandonment_risk": "Abandonment risk",
                "avg_recovery_intensity": "Recovery intensity",
                "avg_churn_probability": "Avg churn prob",
                "avg_builder_momentum": "Builder momentum",
                "pct_agent_builder": "% Agent Builders",
            })
            st.dataframe(disp.round(3), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            """
            <div class="insight-box">
            <strong>Abandonment-prone struggle: 121 users</strong><br><br>
            High abandonment risk (0.69), high churn probability (0.89).
            These users are in active distress - they're trying and failing, and will not
            recover without direct support. Targeted help content, error-assist prompts,
            or human-in-the-loop interventions are warranted.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="insight-box">
            <strong>No visible struggle: 4,404 users</strong><br><br>
            Low quality score (0.06) but high abandonment risk (0.62) - these are mostly
            Ghost users who never started. Their churn is not from struggle; it is from
            never being activated in the first place. Intervention is onboarding, not support.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if has_struggle and struggle is not None:
        col = next(
            (c for c in ["struggle_class", "quality_of_struggle_class", "recommended_support"]
             if c in struggle.columns), None,
        )
        if col and "churn_probability" in struggle.columns and "builder_momentum_signal" in struggle.columns:
            st.markdown("#### Recovery intensity vs churn risk by struggle class")
            with st.expander("Show scatter sample (first 500 users)"):
                sample = struggle[[col, "churn_probability", "builder_momentum_signal"]].head(500)
                fig_sc = px.scatter(
                    sample, x="churn_probability", y="builder_momentum_signal",
                    color=col, template=T,
                    labels={"churn_probability": "Churn probability",
                            "builder_momentum_signal": "Builder momentum"},
                )
                fig_sc.update_layout(height=380, margin=dict(t=30, l=20, r=20, b=30))
                st.plotly_chart(fig_sc, use_container_width=True)


# ─── TAB 9: WORKFLOW & PATHS ─────────────────────────────────────────────────

with tab_workflow:
    st.markdown("### Workflow analysis - how successful users behave differently")
    st.markdown(
        "Agent Builders do not just use more tools - they follow distinct paths, "
        "exhibit iterative build-run loops, and return to canvases repeatedly to increase complexity."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_path_divergence(), key="path_div", use_container_width=True)
        st.markdown(
            '<div class="small-note">Agent Builders hit AGENT_CHAT and AGENT_OTHER events '
            'in their first steps. Ghosts start with OTHER or just AUTH/ONBOARD and stop there.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.plotly_chart(fig_tool_mix(), key="tool_mix", use_container_width=True)
        st.markdown(
            '<div class="small-note">CREATE→RUN→REFACTOR loops are the hallmark of productive '
            'Agent Builder sessions. Viewer tool usage is dominated by passive GET calls.</div>',
            unsafe_allow_html=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        # Top AB paths
        ab_p = D.get("ab_paths")
        gh_p = D.get("ghost_paths")
        if ab_p is not None:
            st.markdown("#### Top Agent Builder journey paths")
            st.dataframe(
                ab_p.rename(columns={"path_str": "Path", "users": "Users"}),
                use_container_width=True,
            )
        if gh_p is not None:
            st.markdown("#### Top Ghost journey paths")
            st.dataframe(
                gh_p.rename(columns={"path_str": "Path", "users": "Users"}),
                use_container_width=True,
            )
    with c4:
        # Bigrams
        bg = D.get("bigrams")
        if bg is not None:
            st.markdown("#### Top tool transition bigrams by segment")
            segs_bg = bg["segment"].unique()[:3]
            for seg in segs_bg:
                sdf = bg[bg["segment"] == seg].head(5)
                st.markdown(f"**{seg}**")
                st.dataframe(
                    sdf[["tool_1", "tool_2", "count"]].rename(
                        columns={"tool_1": "From", "tool_2": "To", "count": "Count"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("#### Canvas complexity - repeat canvas users engage longer")
    st.plotly_chart(fig_canvas_repeat(), key="canvas_repeat", use_container_width=True)

    cg = D.get("canvas_growth")
    if cg is not None:
        st.markdown("#### Canvas growth trajectory by segment")
        st.dataframe(
            cg.rename(columns={
                "segment": "Segment", "avg_growth": "Avg complexity growth",
                "pct_growth": "% users with growth", "users": "Users",
            }).round(2),
            use_container_width=True,
        )
        st.markdown(
            '<div class="small-note">Increasing canvas complexity over time is a strong '
            'signal of deepening engagement. Monitor this metric as a leading retention indicator.</div>',
            unsafe_allow_html=True,
        )

    # Branch points
    br = D.get("branch_top")
    if br is not None:
        st.markdown("#### Top path branch points - where AB and Ghost diverge most")
        disp_br = br.head(10)[["prefix", "next_event", "users",
                                "pct_agent_builder", "pct_ghost", "branch_gap"]].rename(columns={
            "prefix": "Event prefix", "next_event": "Next event",
            "users": "Users", "pct_agent_builder": "AB %",
            "pct_ghost": "Ghost %", "branch_gap": "Gap",
        })
        st.dataframe(disp_br.round(1), use_container_width=True)


# ============================================================
# FOOTER - NORTH STAR + ACTIONS
# ============================================================

st.markdown("---")
st.markdown("### Product direction")

col_ns, col_act = st.columns([1, 1.6])

with col_ns:
    st.markdown(
        """
        <div class="north-star-box">
        <strong style="font-size:1.05rem;">North star metric</strong><br><br>
        <span style="font-size:1.5rem; color:#00b4d8; font-weight:700;">
        % of new users who complete at least one<br>
        agent-assisted pipeline build within 7 days
        </span><br><br>
        Currently: <strong>~13.6%</strong> of all users ever use the agent<br>
        Target: <strong>30%+</strong> within 90 days of onboarding redesign<br><br>
        Every product decision should be evaluated against movement in this number.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_act:
    st.markdown(
        """
        <div class="action-box">
        <strong>Action 1 · Rebuild onboarding to make the first agent interaction mandatory</strong><br><br>
        63.7% of users never ran a single tool. The path to the agent is too long or too opaque.
        Compress time-to-first-agent-build through a guided, low-friction onboarding flow that
        demonstrates value before users have a reason to leave. <em>Metric: % users with ≥1 agent
        build call in session 1.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="action-box">
        <strong>Action 2 · Emergency audit of the Nov–Dec cohort agent adoption collapse</strong><br><br>
        A drop from 19.5% to 0.0% Agent Builder adoption across 3,300 users in two consecutive cohorts
        is a critical regression. Determine within two weeks whether the cause is a UI change,
        feature regression, acquisition channel shift, or observation-window bias.
        <em>Metric: root cause identified and confirmed.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="action-box">
        <strong>Action 3 · Deploy the scored intervention model immediately</strong><br><br>
        3,393 activation nudge candidates and 846 retention rescue targets are already identified
        with computed priority scores. Deploy in-app prompts, email sequences, or agent-assisted
        re-engagement flows to these users now. Delay converts a solvable churn problem into
        permanent user loss. <em>Metric: conversion rate from nudge to first agent build.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    '<div class="small-note">Zerve × HackerEarth Hackathon 2026 · '
    'Behavioral analysis of 4,771 users · '
    'Segmentation: rule-based behavioral profiles · '
    'Churn model: Random Forest · '
    'Intervention scoring: propensity-matched behavioral signals</div>',
    unsafe_allow_html=True,
)
