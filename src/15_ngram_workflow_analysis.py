import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

import pandas as pd
import plotly.graph_objects as go

from analytics.events import (
    add_canvas_id,
    add_normalized_tool,
    canonical_event_name,
    reconstruct_sessions,
)
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.metrics import top_ngram_lift

warnings.filterwarnings("ignore")

DATA_PATH = "data/zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
CANVAS_COMPLEXITY_PATH = "outputs/canvas_complexity_features.parquet"
CHURN_PATH = "outputs/14_churn_scored_users.parquet"
CHURN_TARGET_COL = "is_14d_survival_churn_proxy"


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    canvas = pd.read_parquet(CANVAS_COMPLEXITY_PATH)
    canvas.index = canvas.index.astype(str)
    churn = pd.read_parquet(CHURN_PATH).set_index("person_id")

    events = reconstruct_sessions(add_normalized_tool(add_canvas_id(load_events(DATA_PATH))))
    events["unified_event"] = events.apply(canonical_event_name, axis=1)

    users = feat.join(canvas[["avg_canvas_complexity"]], how="left").join(churn[[CHURN_TARGET_COL]], how="left")
    users["avg_canvas_complexity"] = users["avg_canvas_complexity"].fillna(0)
    users[CHURN_TARGET_COL] = users[CHURN_TARGET_COL].fillna(0).astype(int)
    complexity_cut = users["avg_canvas_complexity"].median()
    users["complexity_group"] = users["avg_canvas_complexity"].apply(lambda x: "High complexity" if x >= complexity_cut else "Low complexity")

    user_seqs = (
        events.sort_values(["person_id", "timestamp"])
        .groupby("person_id")["unified_event"]
        .apply(list)
    )
    session_seqs = (
        events.sort_values(["person_id", "timestamp"])
        .groupby(["person_id", "derived_session_num"])["unified_event"]
        .apply(list)
    )
    tool_events = events[events["unified_event"].str.startswith("tool_")]
    tool_seqs = (
        tool_events.sort_values(["person_id", "timestamp"])
        .groupby("person_id")["unified_event"]
        .apply(list)
    )

    print(f"  User sequences: {len(user_seqs):,}")
    print(f"  Session sequences: {len(session_seqs):,}")
    print(f"  Tool-only sequences: {len(tool_seqs):,}")

    tables = []
    comparisons = [
        (
            "Agent Builder motifs",
            user_seqs[user_seqs.index.isin(users[users["segment"] == "Agent Builder"].index)].tolist(),
            user_seqs[user_seqs.index.isin(users[users["segment"] != "Agent Builder"].index)].tolist(),
            3,
        ),
        (
            "Ghost motifs",
            user_seqs[user_seqs.index.isin(users[users["segment"] == "Ghost"].index)].tolist(),
            user_seqs[user_seqs.index.isin(users[users["segment"] != "Ghost"].index)].tolist(),
            2,
        ),
        (
            "Churn motifs",
            session_seqs[session_seqs.index.get_level_values(0).isin(users[users[CHURN_TARGET_COL] == 1].index)].tolist(),
            session_seqs[session_seqs.index.get_level_values(0).isin(users[users[CHURN_TARGET_COL] == 0].index)].tolist(),
            2,
        ),
        (
            "High complexity tool motifs",
            tool_seqs[tool_seqs.index.isin(users[users["complexity_group"] == "High complexity"].index)].tolist(),
            tool_seqs[tool_seqs.index.isin(users[users["complexity_group"] == "Low complexity"].index)].tolist(),
            3,
        ),
    ]

    for label, focal, base, n in comparisons:
        table = top_ngram_lift(
            focal,
            base,
            n=n,
            min_count=5,
            min_baseline_count=3,
            top_k=20,
            smoothing_alpha=1.0,
        )
        if table.empty:
            continue
        table["comparison"] = label
        tables.append(table)
    ngram_tables = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    ngram_tables.to_csv(f"{OUTPUT_DIR}/16_ngram_tables.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/16_ngram_tables.csv")

    fig1_data = ngram_tables[
        (ngram_tables["comparison"] == "Agent Builder motifs")
        & (ngram_tables["publishable"] == 1)
    ].head(12).copy()
    fig1 = go.Figure()
    if not fig1_data.empty:
        fig1.add_trace(go.Bar(
            x=fig1_data["lift"],
            y=fig1_data["ngram"],
            orientation="h",
            marker=dict(
                color=fig1_data["focal_count"],
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="Count"),
            ),
            text=fig1_data["focal_count"],
            textposition="outside",
        ))
    fig1.update_layout(
        title="Event N-gram Lift: Agent Builder Workflow Fingerprints",
        template="plotly_dark",
        xaxis_title="Lift vs others",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=520,
        showlegend=False,
    )
    fig1.write_html(f"{OUTPUT_DIR}/16_event_ngram_lift.html")
    print(f"  Saved: {OUTPUT_DIR}/16_event_ngram_lift.html")

    motif_rows = []
    for segment in ["Agent Builder", "Manual Coder", "Viewer", "Ghost"]:
        focal_users = users[users["segment"] == segment].index
        base_users = users[users["segment"] != segment].index
        table = top_ngram_lift(
            user_seqs[user_seqs.index.isin(focal_users)].tolist(),
            user_seqs[user_seqs.index.isin(base_users)].tolist(),
            n=2,
            min_count=5,
            min_baseline_count=3,
            top_k=5,
            smoothing_alpha=1.0,
        )
        if table.empty:
            continue
        table = table[table["publishable"] == 1].head(3)
        if table.empty:
            continue
        table["segment"] = segment
        motif_rows.append(table)
    motif_table = pd.concat(motif_rows, ignore_index=True) if motif_rows else pd.DataFrame()
    fig2 = go.Figure()
    if not motif_table.empty:
        motif_order = motif_table["ngram"].drop_duplicates().tolist()
        palette = [
            "#00b4d8",
            "#48cae4",
            "#90e0ef",
            "#ffd166",
            "#ef476f",
            "#8338ec",
            "#06d6a0",
            "#f4a261",
        ]
        color_map = {motif: palette[i % len(palette)] for i, motif in enumerate(motif_order)}
        for motif in motif_order:
            sub = motif_table[motif_table["ngram"] == motif]
            fig2.add_trace(go.Bar(
                x=sub["segment"],
                y=sub["lift"],
                name=motif,
                marker_color=color_map[motif],
            ))
    fig2.update_layout(
        barmode="group",
        title="Top Segment Workflow Motifs<br><sup>Most over-indexing bigrams per segment</sup>",
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Lift",
        legend_title_text="Motif",
        height=420,
    )
    fig2.write_html(f"{OUTPUT_DIR}/16_segment_workflow_motifs.html")
    print(f"  Saved: {OUTPUT_DIR}/16_segment_workflow_motifs.html")

    churn_table = ngram_tables[
        (ngram_tables["comparison"] == "Churn motifs")
        & (ngram_tables["publishable"] == 1)
    ].head(12)
    fig3 = go.Figure()
    if not churn_table.empty:
        fig3.add_trace(go.Bar(
            x=churn_table["lift"],
            y=churn_table["ngram"],
            orientation="h",
            marker=dict(
                color=churn_table["focal_count"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Count"),
            ),
            text=churn_table["focal_count"],
            textposition="outside",
        ))
    fig3.update_layout(
        title="Churn vs Retained Workflow Motifs<br><sup>Session-bounded dead-end vs healthy patterns</sup>",
        template="plotly_dark",
        xaxis_title="Lift toward churn",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=520,
        showlegend=False,
    )
    fig3.write_html(f"{OUTPUT_DIR}/16_churn_vs_retained_motifs.html")
    print(f"  Saved: {OUTPUT_DIR}/16_churn_vs_retained_motifs.html")

    print("\n[OK] N-gram workflow analysis complete.")

# Zerve: call main() directly (no __main__ guard)
main()
