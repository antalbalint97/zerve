import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.events import get_user_country, map_country_region
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.metrics import wilson_interval

warnings.filterwarnings("ignore")

DATA_PATH = "zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
CANVAS_COMPLEXITY_PATH = "outputs/canvas_complexity_features.parquet"
CHURN_PATH_CANDIDATES = [
    "outputs/14_churn_scored_users.parquet",
    "outputs/15_churn_scored_users.parquet",
]
MIN_USERS = 40


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    events = load_events(DATA_PATH)
    canvas = pd.read_parquet(CANVAS_COMPLEXITY_PATH)
    canvas.index = canvas.index.astype(str)
    churn_path = next((p for p in CHURN_PATH_CANDIDATES if Path(p).exists()), None)
    if churn_path is None:
        raise FileNotFoundError(
            f"None of the expected churn outputs exist: {CHURN_PATH_CANDIDATES}"
        )
    print(f"  Using churn file: {churn_path}")
    churn = pd.read_parquet(churn_path).set_index("person_id")
    churn_target_cols = ["churn_probability"]
    if "is_14d_survival_churn_proxy" in churn.columns:
        churn_target_cols.append("is_14d_survival_churn_proxy")
    elif "is_churn" in churn.columns:
        churn_target_cols.append("is_churn")

    countries = get_user_country(events).rename("country_code")
    geo = feat.join(canvas[["avg_canvas_complexity", "avg_canvas_growth"]], how="left").join(
        churn[churn_target_cols], how="left"
    ).join(countries, how="left")
    geo["country_code"] = geo["country_code"].fillna("Unknown")
    geo["region_group"] = geo["country_code"].map(map_country_region)
    geo["avg_canvas_complexity"] = geo["avg_canvas_complexity"].fillna(0)
    geo["avg_canvas_growth"] = geo["avg_canvas_growth"].fillna(0)
    geo["churn_probability"] = geo["churn_probability"].fillna(0)
    churn_col = "is_14d_survival_churn_proxy" if "is_14d_survival_churn_proxy" in geo.columns else "is_churn"
    geo[churn_col] = geo[churn_col].fillna(0)

    country_metrics = (
        geo.groupby("country_code")
        .agg(
            users=("segment", "count"),
            pct_agent_builder=("segment", lambda x: (x == "Agent Builder").mean() * 100),
            pct_ever_agent=("ever_used_agent", "mean"),
            pct_repeat_session=("had_second_session", "mean"),
            avg_canvas_complexity=("avg_canvas_complexity", "mean"),
            avg_canvas_growth=("avg_canvas_growth", "mean"),
            avg_credit_events=("credit_events", "mean"),
            avg_signup_hour=("signup_hour", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            actual_churn_pct=(churn_col, "mean"),
        )
        .reset_index()
    )
    country_metrics["pct_ever_agent"] *= 100
    country_metrics["pct_repeat_session"] *= 100
    country_metrics["actual_churn_pct"] *= 100
    country_metrics["region_group"] = country_metrics["country_code"].map(map_country_region)
    interval_df = country_metrics.apply(
        lambda row: pd.Series(
            wilson_interval(
                int(round(row["pct_agent_builder"] * row["users"] / 100)),
                int(row["users"]),
            ),
            index=["agent_builder_ci_low", "agent_builder_ci_high"],
        ),
        axis=1,
    )
    country_metrics[["agent_builder_ci_low", "agent_builder_ci_high"]] = interval_df * 100
    country_metrics = country_metrics.sort_values("users", ascending=False)
    country_metrics.to_csv(f"{OUTPUT_DIR}/17_country_metrics.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/17_country_metrics.csv")

    eligible_countries = country_metrics[country_metrics["users"] >= MIN_USERS].copy()
    top_countries = eligible_countries["country_code"].head(10).tolist()
    geo_top = geo[geo["country_code"].isin(top_countries)].copy()

    seg_mix = (
        geo_top.groupby(["country_code", "segment"])
        .size()
        .reset_index(name="users")
    )
    seg_mix_matrix = (
        seg_mix.pivot(index="country_code", columns="segment", values="users")
        .fillna(0)
    )
    segment_order = ["Agent Builder", "Manual Coder", "Viewer", "Ghost"]
    seg_mix_matrix = seg_mix_matrix.reindex(
        columns=[s for s in segment_order if s in seg_mix_matrix.columns],
        fill_value=0,
    )
    seg_colors = {
        "Agent Builder": "#00b4d8",
        "Manual Coder": "#90e0ef",
        "Viewer": "#555566",
        "Ghost": "#2d2d3a",
    }
    fig1 = go.Figure()
    for segment in seg_mix_matrix.columns:
        fig1.add_trace(go.Bar(
            x=seg_mix_matrix.index,
            y=seg_mix_matrix[segment],
            name=segment,
            marker_color=seg_colors.get(segment, "#888"),
        ))
    fig1.update_layout(
        barmode="stack",
        title="Country Segment Mix<br><sup>Where are the Agent Builders, Viewers, and Ghosts concentrated?</sup>",
        template="plotly_dark",
        xaxis_title="Country",
        yaxis_title="Users",
        legend_title_text="",
        height=420,
    )
    fig1.write_html(f"{OUTPUT_DIR}/17_country_segment_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/17_country_segment_mix.html")

    region_metrics = (
        geo[geo["country_code"] != "Unknown"].groupby("region_group")
        .agg(
            users=("segment", "count"),
            pct_ever_agent=("ever_used_agent", "mean"),
            pct_repeat_session=("had_second_session", "mean"),
            pct_onboarding_complete=("completed_onboarding", "mean"),
            pct_agent_builder=("segment", lambda x: (x == "Agent Builder").mean()),
        )
        .reset_index()
    )
    for col in ["pct_ever_agent", "pct_repeat_session", "pct_onboarding_complete", "pct_agent_builder"]:
        region_metrics[col] *= 100

    region_plot = region_metrics.melt(
        id_vars="region_group",
        value_vars=["pct_ever_agent", "pct_repeat_session", "pct_onboarding_complete", "pct_agent_builder"],
        var_name="metric",
        value_name="value",
    )
    metric_order = [
        "pct_ever_agent",
        "pct_repeat_session",
        "pct_onboarding_complete",
        "pct_agent_builder",
    ]
    metric_labels = {
        "pct_ever_agent": "Ever used agent %",
        "pct_repeat_session": "Repeat session %",
        "pct_onboarding_complete": "Onboarding complete %",
        "pct_agent_builder": "Agent Builder %",
    }
    metric_colors = {
        "pct_ever_agent": "#00b4d8",
        "pct_repeat_session": "#48cae4",
        "pct_onboarding_complete": "#90e0ef",
        "pct_agent_builder": "#ffd166",
    }
    fig2 = go.Figure()
    for metric in metric_order:
        sub = region_plot[region_plot["metric"] == metric]
        if len(sub) == 0:
            continue
        fig2.add_trace(go.Bar(
            x=sub["region_group"],
            y=sub["value"],
            name=metric_labels.get(metric, metric),
            marker_color=metric_colors.get(metric, "#888"),
        ))
    fig2.update_layout(
        barmode="group",
        title="Geo Onboarding and Activation Effectiveness<br><sup>India vs US vs EU vs Other</sup>",
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="%",
        legend_title_text="",
        height=420,
    )
    fig2.write_html(f"{OUTPUT_DIR}/17_geo_onboarding_effectiveness.html")
    print(f"  Saved: {OUTPUT_DIR}/17_geo_onboarding_effectiveness.html")

    heatmap_df = (
        eligible_countries
        .set_index("country_code")[["avg_canvas_complexity", "avg_churn_probability", "avg_credit_events", "pct_repeat_session"]]
    )
    heatmap_norm = heatmap_df.apply(lambda col: (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else col * 0, axis=0)
    fig3 = px.imshow(
        heatmap_norm.T,
        text_auto=".2f",
        color_continuous_scale="Tealgrn",
        title="Geo Churn / Complexity Heatmap<br><sup>Normalized country-level product signals</sup>",
        template="plotly_dark",
        labels={"x": "Country", "y": "Metric", "color": "Normalized"},
        aspect="auto",
    )
    fig3.write_html(f"{OUTPUT_DIR}/17_geo_churn_complexity_heatmap.html")
    print(f"  Saved: {OUTPUT_DIR}/17_geo_churn_complexity_heatmap.html")

    print("\n[OK] Geo location analysis complete.")

# Zerve: call main() directly (no __main__ guard)
main()
