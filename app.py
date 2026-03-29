from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zerve Activation Lens",
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
            padding-bottom: 2rem;
            max-width: 1350px;
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 12px 14px;
        }
        .section-caption {
            color: #9ca3af;
            font-size: 0.95rem;
            margin-top: -0.2rem;
            margin-bottom: 1.2rem;
        }
        .small-note {
            color: #9ca3af;
            font-size: 0.9rem;
        }
        .insight-box {
            background: #0b1220;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .tight-list ul {
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
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
    "Viewer": "#555566",
    "Ghost": "#2d2d3a",
}
SEG_ORDER = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]


# ============================================================
# FILE LOADING
# ============================================================

def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def candidate_paths(base_name: str) -> list[Path]:
    return [
        OUTPUTS_DIR / f"{base_name}.parquet",
        OUTPUTS_DIR / f"{base_name}.csv",
        OUTPUTS_DIR / f"outputs_{base_name}.parquet",
        OUTPUTS_DIR / f"outputs_{base_name}.csv",
        BASE_DIR / f"{base_name}.parquet",
        BASE_DIR / f"{base_name}.csv",
        BASE_DIR / f"outputs_{base_name}.parquet",
        BASE_DIR / f"outputs_{base_name}.csv",
    ]


def load_table(base_names: list[str], required: bool = False) -> tuple[Optional[pd.DataFrame], Optional[Path]]:
    candidates: list[Path] = []
    for name in base_names:
        candidates.extend(candidate_paths(name))

    found = first_existing(candidates)
    if found is None:
        if required:
            searched = "\n".join(str(p) for p in candidates)
            raise FileNotFoundError(
                f"Required dataset not found. Tried:\n{searched}"
            )
        return None, None

    if found.suffix.lower() == ".parquet":
        df = pd.read_parquet(found)
    elif found.suffix.lower() == ".csv":
        df = pd.read_csv(found)
    else:
        if required:
            raise ValueError(f"Unsupported file type: {found}")
        return None, None

    return df, found


@st.cache_data(show_spinner=False)
def load_all_data():
    feat, feat_path = load_table(
        [
            "user_features_segmented",
            "outputs_user_features",
            "user_features",
        ],
        required=True,
    )

    churn, churn_path = load_table(
        [
            "14_churn_scored_users",
            "15_churn_scored_users",
            "churn_scored_users",
        ],
        required=False,
    )

    interv, interv_path = load_table(
        [
            "17_intervention_scoring",
            "17_intervention_scored_users",
            "18_intervention_scored_users",
            "intervention_scored_users",
        ],
        required=False,
    )

    struggle, struggle_path = load_table(
        [
            "18_quality_of_struggle",
            "18_quality_of_struggle_scored_users",
            "19_quality_of_struggle_scored_users",
            "quality_of_struggle_scored_users",
        ],
        required=False,
    )

    return {
        "feat": feat,
        "feat_path": feat_path,
        "churn": churn,
        "churn_path": churn_path,
        "interv": interv,
        "interv_path": interv_path,
        "struggle": struggle,
        "struggle_path": struggle_path,
    }


data = load_all_data()
feat = data["feat"]
churn = data["churn"]
interv = data["interv"]
struggle = data["struggle"]

has_churn = churn is not None and isinstance(churn, pd.DataFrame)
has_interv = interv is not None and isinstance(interv, pd.DataFrame)
has_struggle = struggle is not None and isinstance(struggle, pd.DataFrame)


# ============================================================
# VALIDATION
# ============================================================

required_feat_cols = [
    "segment",
    "days_active",
    "agent_tool_calls_total",
    "agent_build_calls",
    "had_second_session",
    "had_credit_exceeded",
    "signup_cohort",
    "ever_used_agent",
]

missing_cols = [c for c in required_feat_cols if c not in feat.columns]
if missing_cols:
    st.error(f"The main dataset is missing required columns: {missing_cols}")
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
)

seg_stats = seg_stats.reindex([s for s in SEG_ORDER if s in seg_stats.index])
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
    if "adopted_agent_early" in feat.columns
    else None
)

churn_rate = None
if has_churn and "is_14d_survival_churn_proxy" in churn.columns:
    churn_rate = round(churn["is_14d_survival_churn_proxy"].mean() * 100, 1)


# ============================================================
# FIGURES
# ============================================================

def build_segments_figure() -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["User distribution", "Average agent tool calls per user"],
    )

    fig.add_trace(
        go.Pie(
            labels=list(seg_stats.index),
            values=seg_stats["users"],
            hole=0.48,
            marker_colors=[COLORS.get(s, "#888888") for s in seg_stats.index],
            textinfo="label+percent",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=list(seg_stats.index),
            y=seg_stats["avg_tools"].round(1),
            marker_color=[COLORS.get(s, "#888888") for s in seg_stats.index],
            text=seg_stats["avg_tools"].round(1),
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Behavioral segmentation",
        template="plotly_dark",
        height=430,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    return fig


def build_cohort_figure() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=cohort_stats.index,
            y=cohort_stats["users"],
            name="Users",
            marker_color="#1a3040",
            opacity=0.9,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=cohort_stats.index,
            y=cohort_stats["pct_ab"],
            name="Agent Builder %",
            mode="lines+markers",
            line=dict(color="#00b4d8", width=3),
            marker=dict(size=9),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=cohort_stats.index,
            y=cohort_stats["pct_agent"],
            name="Ever used agent %",
            mode="lines+markers",
            line=dict(color="#90e0ef", width=2, dash="dot"),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Signup cohort performance",
        template="plotly_dark",
        height=430,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=70, l=20, r=20, b=20),
    )
    fig.update_yaxes(title_text="Users", secondary_y=False)
    fig.update_yaxes(title_text="Adoption rate (%)", secondary_y=True)
    return fig


def build_retention_figure() -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(seg_stats.index),
            y=seg_stats["pct_return"],
            marker_color=[COLORS.get(s, "#888888") for s in seg_stats.index],
            text=[f"{v:.0f}%" for v in seg_stats["pct_return"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Second-session return rate by segment",
        yaxis_title="Return rate (%)",
        yaxis_range=[0, 110],
        template="plotly_dark",
        height=380,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    return fig


def build_churn_figure() -> go.Figure:
    if not has_churn:
        fig = go.Figure()
        fig.update_layout(
            title="Churn data not available",
            template="plotly_dark",
            height=360,
        )
        return fig

    if "churn_risk_bucket" not in churn.columns or "churn_probability" not in churn.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Churn dataset loaded, but expected columns are missing",
            template="plotly_dark",
            height=360,
        )
        return fig

    risk = churn.groupby("churn_risk_bucket").agg(
        users=("churn_probability", "count"),
        avg_prob=("churn_probability", "mean"),
    ).reset_index()

    order = ["Low", "Medium", "High", "Critical"]
    risk["churn_risk_bucket"] = pd.Categorical(
        risk["churn_risk_bucket"],
        categories=order,
        ordered=True,
    )
    risk = risk.sort_values("churn_risk_bucket")

    risk_colors = {
        "Low": "#48cae4",
        "Medium": "#90e0ef",
        "High": "#f4a261",
        "Critical": "#e63946",
    }

    fig = go.Figure(
        go.Bar(
            x=risk["churn_risk_bucket"],
            y=risk["users"],
            marker_color=[risk_colors.get(v, "#888888") for v in risk["churn_risk_bucket"]],
            text=risk["users"],
            textposition="outside",
        )
    )

    title = "14-day churn risk distribution"
    if churn_rate is not None:
        title += f" · overall rate: {churn_rate}%"

    fig.update_layout(
        title=title,
        yaxis_title="Users",
        template="plotly_dark",
        height=380,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    return fig


def build_intervention_figure() -> go.Figure:
    if not has_interv:
        fig = go.Figure()
        fig.update_layout(
            title="Intervention data not available",
            template="plotly_dark",
            height=360,
        )
        return fig

    if "recommended_intervention" not in interv.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Intervention dataset loaded, but expected columns are missing",
            template="plotly_dark",
            height=360,
        )
        return fig

    iv = interv.groupby("recommended_intervention").size().reset_index(name="users")
    iv = iv.sort_values("users", ascending=True)

    color_map = {
        "Activation nudge": "#00b4d8",
        "Retention rescue": "#48cae4",
        "Monitor": "#555566",
        "Builder acceleration": "#90e0ef",
        "Productive struggle support": "#0077b6",
    }

    fig = go.Figure(
        go.Bar(
            x=iv["users"],
            y=iv["recommended_intervention"],
            orientation="h",
            marker_color=[color_map.get(v, "#888888") for v in iv["recommended_intervention"]],
            text=iv["users"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Intervention targeting",
        xaxis_title="Users",
        template="plotly_dark",
        height=380,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    return fig


def build_struggle_figure() -> go.Figure:
    if not has_struggle:
        fig = go.Figure()
        fig.update_layout(
            title="Quality-of-struggle data not available",
            template="plotly_dark",
            height=360,
        )
        return fig

    col = None
    for candidate in ["struggle_class", "quality_of_struggle_class", "recommended_support"]:
        if candidate in struggle.columns:
            col = candidate
            break

    if col is None:
        fig = go.Figure()
        fig.update_layout(
            title="Quality-of-struggle dataset loaded, but expected columns are missing",
            template="plotly_dark",
            height=360,
        )
        return fig

    summary = struggle.groupby(col).size().reset_index(name="users")
    summary = summary.sort_values("users", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=summary[col],
            y=summary["users"],
            text=summary["users"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Quality of struggle",
        xaxis_title="Class",
        yaxis_title="Users",
        template="plotly_dark",
        height=380,
        margin=dict(t=70, l=20, r=20, b=20),
    )
    return fig


segments_fig = build_segments_figure()
cohort_fig = build_cohort_figure()
retention_fig = build_retention_figure()
churn_fig = build_churn_figure()
intervention_fig = build_intervention_figure()
struggle_fig = build_struggle_figure()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Zerve Activation Lens")
    st.caption("Behavioral analysis dashboard")

    st.markdown("### Data sources")
    st.write(f"Main feature table: `{data['feat_path'].name}`")
    st.write(f"Churn: `{data['churn_path'].name}`" if data["churn_path"] else "Churn: not loaded")
    st.write(f"Intervention: `{data['interv_path'].name}`" if data["interv_path"] else "Intervention: not loaded")
    st.write(f"Struggle: `{data['struggle_path'].name}`" if data["struggle_path"] else "Struggle: not loaded")

    st.markdown("---")
    st.markdown("### Interpretation")
    st.markdown(
        """
        This app is a presentation layer over output tables
        generated from the Zerve project pipeline.
        """
    )


# ============================================================
# HEADER
# ============================================================

st.title("Zerve Activation Lens")
st.markdown(
    '<div class="section-caption">What drives successful usage on Zerve?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    This dashboard summarizes behavioral patterns across 4,771 Zerve users and focuses on
    activation, retention, churn risk, and intervention opportunity. It is designed as a
    polished presentation layer over precomputed Zerve outputs rather than a re-execution
    environment for the full modeling pipeline.
    """
)


# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total users", f"{n:,}")
m2.metric("Agent Builder", f"{ab_users:,}", f"{ab_users / n * 100:.1f}% of users")
m3.metric("Ghost users", f"{ghost_users:,}", f"{ghost_users / n * 100:.1f}% of users")
m4.metric("Ever used the agent", f"{pct_agent}%")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Second-session return", f"{pct_return}%")
m6.metric("Early adopters (<1h)", f"{pct_early}%" if pct_early is not None else "N/A")
m7.metric("14-day churn rate", f"{churn_rate}%" if churn_rate is not None else "N/A")
north_star_value = f"{pct_agent}%"
m8.metric("North star baseline", north_star_value)

st.markdown("---")


# ============================================================
# EXECUTIVE TAKEAWAYS
# ============================================================

st.subheader("Executive takeaways")

col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.markdown(
        """
        <div class="insight-box tight-list">
        <strong>Core finding</strong><br><br>
        The product appears highly valuable for a small minority of users who reach the
        agent-assisted workflow, but most users never get there. The primary problem is
        activation, not just downstream churn.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-box tight-list">
        <strong>North star metric</strong><br><br>
        Percentage of new users who complete at least one agent-assisted build within
        their first 7 days.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="insight-box tight-list">
        <strong>Recommended actions</strong>
        <ul>
            <li>Redesign onboarding to force earlier contact with the agent.</li>
            <li>Investigate the Nov–Dec cohort adoption collapse immediately.</li>
            <li>Deploy the scored intervention model for activation and retention rescue.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Segments",
        "Cohorts",
        "Retention",
        "Churn",
        "Interventions",
    ]
)

with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.plotly_chart(
            segments_fig,
            key="segments_fig_tab1",
            width="stretch",
        )

    with right:
        st.markdown("#### Summary")
        st.markdown(
            f"""
            - **{ab_users / n * 100:.1f}%** of users reached **Agent Builder** status.
            - **{ghost_users / n * 100:.1f}%** of users are **Ghosts** with near-zero meaningful engagement.
            - **{pct_agent}%** of all users ever used the AI agent.
            - **{pct_return}%** returned for a second session.
            """
        )

        st.markdown("#### Why Agent Builder is the success definition")
        st.markdown(
            """
            Agent Builders are the users who most clearly realize Zerve's core product value:
            they use the AI agent to construct or meaningfully transform workflows, and they
            show the strongest retention signature in the dataset.
            """
        )

with tab2:
    st.plotly_chart(
        segments_fig,
        key="segments_fig_tab2",
        width="stretch",
    )

    seg_table = seg_stats.copy()
    seg_table = seg_table.rename(
        columns={
            "users": "Users",
            "avg_days": "Avg active days",
            "avg_tools": "Avg agent tools",
            "avg_builds": "Avg build calls",
            "pct_return": "Second-session return %",
            "pct_credit": "Credit exceeded %",
            "pct_of_users": "Share of users %",
        }
    )
    st.dataframe(seg_table.round(2), width="stretch")

with tab3:
    st.plotly_chart(
        cohort_fig,
        key="cohort_fig_tab3",
        width="stretch",
    )

    st.markdown(
        """
        The cohort view highlights a potentially severe adoption breakdown in the most recent
        signup cohorts. Some of this may reflect shorter observation windows, but the magnitude
        is large enough to justify direct product investigation.
        """
    )

    cohort_table = cohort_stats.copy().rename(
        columns={
            "users": "Users",
            "pct_ab": "Agent Builder %",
            "pct_agent": "Ever used agent %",
            "avg_tools": "Avg agent tools",
        }
    )
    st.dataframe(cohort_table.round(2), width="stretch")

with tab4:
    left, right = st.columns([1.1, 1])

    with left:
        st.plotly_chart(
            retention_fig,
            key="retention_fig_tab4",
            width="stretch",
        )

    with right:
        st.markdown("#### Interpretation")
        st.markdown(
            """
            Retention is sharply segmented. The users who reach meaningful agent-assisted work
            return at substantially higher rates than users who stay in passive or low-engagement
            usage modes.
            """
        )

with tab5:
    st.plotly_chart(
        churn_fig,
        key="churn_fig_tab5",
        width="stretch",
    )

    if has_churn:
        st.markdown(
            """
            The churn model is used here as a risk-ranking layer over the active population.
            This helps prioritize users for rescue workflows instead of treating all engaged
            users as equally at risk.
            """
        )
    else:
        st.info("Add the churn-scored output file to enable this section.")

with tab6:
    left, right = st.columns([1.1, 1])

    with left:
        st.plotly_chart(
            intervention_fig,
            key="intervention_fig_tab6_left",
            width="stretch",
        )

    with right:
        st.plotly_chart(
            struggle_fig,
            key="struggle_fig_tab6_right",
            width="stretch",
        )

    if has_interv:
        st.markdown(
            """
            Intervention scoring is most useful when translated into immediate product actions:
            activation nudges for users who never meaningfully started, and retention rescue for
            users who showed intent but are slipping away.
            """
        )
    else:
        st.info("Add the intervention-scored output file to enable this section.")

st.markdown("---")
st.markdown(
    '<div class="small-note">Built as a polished presentation layer for outputs generated in the Zerve project pipeline.</div>',
    unsafe_allow_html=True,
)