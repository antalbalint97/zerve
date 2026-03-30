import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from collections import Counter

import pandas as pd
import plotly.graph_objects as go

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import write_html

warnings.filterwarnings("ignore")

DATA_PATH = "data/zerve_events.csv"
FEAT_PATH = "outputs/user_features_segmented.parquet"
N_STEPS = 5
MIN_PREFIX_USERS = 15


def categorize_event(event: str) -> str:
    if event in {"sign_up", "sign_in"}:
        return "AUTH"
    if "onboarding" in event or "skip_onboarding" in event:
        return "ONBOARD"
    if event in {"run_block", "run_all_blocks"}:
        return "RUN"
    if event in {"agent_new_chat", "agent_start_from_prompt", "agent_message"}:
        return "AGENT_CHAT"
    if event == "agent_tool_call_create_block_tool":
        return "AGENT_BUILD"
    if event == "agent_tool_call_run_block_tool":
        return "AGENT_RUN"
    if event == "agent_tool_call_refactor_block_tool":
        return "AGENT_REFACTOR"
    if event.startswith("agent_tool_call"):
        return "AGENT_OTHER"
    if event in {"block_create", "block_delete", "block_resize"}:
        return "CANVAS_EDIT"
    if "fullscreen" in event:
        return "VIEW_OUTPUT"
    if "credits" in event:
        return "CREDITS"
    return "OTHER"


def deduped_path(events: list[str], n_steps: int = N_STEPS) -> list[str]:
    out = []
    for event in events:
        if not out or out[-1] != event:
            out.append(event)
    return out[:n_steps]


def next_step_table(path_df: pd.DataFrame, segment_a: str, segment_b: str) -> pd.DataFrame:
    rows = []
    for prefix_len in range(1, N_STEPS):
        prefix_col = f"prefix_{prefix_len}"
        next_col = f"step_{prefix_len + 1}"
        valid = path_df[path_df[prefix_col].notna() & path_df[next_col].notna()].copy()
        if valid.empty:
            continue
        for prefix, group in valid.groupby(prefix_col):
            if len(group) < MIN_PREFIX_USERS:
                continue
            for next_event, branch in group.groupby(next_col):
                a_mask = branch["segment"] == segment_a
                b_mask = branch["segment"] == segment_b
                rows.append(
                    {
                        "prefix_len": prefix_len,
                        "prefix": prefix,
                        "next_event": next_event,
                        "users": len(branch),
                        f"pct_{segment_a.lower().replace(' ', '_')}": a_mask.mean() * 100,
                        f"pct_{segment_b.lower().replace(' ', '_')}": b_mask.mean() * 100,
                        "overall_pct_ab": (branch["segment"] == "Agent Builder").mean() * 100,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    a_col = f"pct_{segment_a.lower().replace(' ', '_')}"
    b_col = f"pct_{segment_b.lower().replace(' ', '_')}"
    out["branch_gap"] = out[a_col] - out[b_col]
    return out.sort_values(["prefix_len", "branch_gap", "users"], ascending=[True, False, False])


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    events = load_events(DATA_PATH)
    events = events.merge(feat[["segment"]].reset_index(), on="person_id", how="left")
    events["event_cat"] = events["event"].map(categorize_event)

    user_paths = (
        events.sort_values(["person_id", "timestamp"])
        .groupby("person_id")["event_cat"]
        .apply(list)
        .apply(deduped_path)
        .rename("path")
        .to_frame()
        .join(feat[["segment"]], how="left")
    )
    user_paths["path_str"] = user_paths["path"].apply(lambda x: " -> ".join(x))

    for step in range(1, N_STEPS + 1):
        user_paths[f"step_{step}"] = user_paths["path"].apply(lambda x: x[step - 1] if len(x) >= step else None)
    for prefix_len in range(1, N_STEPS):
        user_paths[f"prefix_{prefix_len}"] = user_paths["path"].apply(
            lambda x, n=prefix_len: " -> ".join(x[:n]) if len(x) >= n else None
        )

    print(f"  User paths: {len(user_paths):,}")

    step_rows = []
    segments = ["Agent Builder", "Viewer", "Ghost"]
    for step in range(1, N_STEPS + 1):
        for segment in segments:
            values = user_paths[user_paths["segment"] == segment][f"step_{step}"].dropna()
            if len(values) == 0:
                continue
            counts = values.value_counts(normalize=True) * 100
            for event_name, pct in counts.head(5).items():
                step_rows.append(
                    {
                        "step": step,
                        "segment": segment,
                        "event": event_name,
                        "pct": pct,
                    }
                )
    step_df = pd.DataFrame(step_rows)
    step_df.to_csv(f"{OUTPUT_DIR}/20_branch_step_divergence.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/20_branch_step_divergence.csv")

    ab_vs_ghost = next_step_table(user_paths.reset_index(), "Agent Builder", "Ghost")
    ab_vs_viewer = next_step_table(user_paths.reset_index(), "Agent Builder", "Viewer")
    branch_summary = pd.concat(
        [
            ab_vs_ghost.assign(comparison="Agent Builder vs Ghost"),
            ab_vs_viewer.assign(comparison="Agent Builder vs Viewer"),
        ],
        ignore_index=True,
    )
    branch_summary.to_csv(f"{OUTPUT_DIR}/20_branching_summary.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/20_branching_summary.csv")

    fig1 = go.Figure()
    event_order = step_df["event"].drop_duplicates().tolist() if not step_df.empty else []
    palette = [
        "#00b4d8", "#48cae4", "#90e0ef", "#ffd166",
        "#ef476f", "#8338ec", "#06d6a0", "#f4a261",
    ]
    event_colors = {event: palette[i % len(palette)] for i, event in enumerate(event_order)}
    segment_offsets = {"Agent Builder": -0.25, "Viewer": 0.0, "Ghost": 0.25}
    for segment in segments:
        seg_df = step_df[step_df["segment"] == segment]
        for event_name in event_order:
            sub = seg_df[seg_df["event"] == event_name]
            if sub.empty:
                continue
            x_vals = [float(step) + segment_offsets.get(segment, 0.0) for step in sub["step"]]
            fig1.add_trace(go.Bar(
                x=x_vals,
                y=sub["pct"],
                name=f"{segment}: {event_name}",
                marker_color=event_colors.get(event_name, "#888"),
                width=0.22,
                offsetgroup=segment,
                legendgroup=event_name,
            ))
    fig1.update_layout(
        title="Early Path Step Mix By Segment<br><sup>Where Ghost, Viewer, and Builder journeys start to diverge</sup>",
        template="plotly_dark",
        xaxis_title="Path step",
        yaxis_title="Users %",
        height=720,
        barmode="group",
    )
    write_html(fig1, f"{OUTPUT_DIR}/20_path_step_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/20_path_step_mix.html")

    top_branches = branch_summary.sort_values("branch_gap", ascending=False).head(20)
    fig2 = go.Figure()
    if not top_branches.empty:
        branch_event_order = top_branches["next_event"].drop_duplicates().tolist()
        branch_event_colors = {
            event: palette[i % len(palette)]
            for i, event in enumerate(branch_event_order)
        }
        comparisons = top_branches["comparison"].drop_duplicates().tolist()
        comparison_offsets = {
            comp: (-0.18 if i == 0 else 0.18) for i, comp in enumerate(comparisons)
        }
        prefix_order = top_branches["prefix"].drop_duplicates().tolist()
        prefix_pos = {prefix: idx for idx, prefix in enumerate(prefix_order)}
        for comp in comparisons:
            comp_df = top_branches[top_branches["comparison"] == comp]
            for event_name in branch_event_order:
                sub = comp_df[comp_df["next_event"] == event_name]
                if sub.empty:
                    continue
                y_vals = [prefix_pos[prefix] + comparison_offsets.get(comp, 0.0) for prefix in sub["prefix"]]
                fig2.add_trace(go.Bar(
                    x=sub["branch_gap"],
                    y=y_vals,
                    orientation="h",
                    name=f"{comp}: {event_name}",
                    marker_color=branch_event_colors.get(event_name, "#888"),
                    width=0.28,
                ))
        fig2.update_yaxes(
            tickmode="array",
            tickvals=list(prefix_pos.values()),
            ticktext=list(prefix_pos.keys()),
            autorange="reversed",
        )
    fig2.update_layout(
        title="Top Early Branch Points<br><sup>Prefixes whose next step most strongly separates outcomes</sup>",
        template="plotly_dark",
        xaxis_title="Branch gap (pp)",
        yaxis_title="Prefix",
        height=620,
        barmode="group",
    )
    write_html(fig2, f"{OUTPUT_DIR}/20_top_branch_points.html")
    print(f"  Saved: {OUTPUT_DIR}/20_top_branch_points.html")

    top_prefix_rows = []
    for segment in segments:
        counts = user_paths[user_paths["segment"] == segment]["path_str"].value_counts().head(8)
        for prefix, n in counts.items():
            top_prefix_rows.append({"segment": segment, "path": prefix, "users": n})
    prefix_df = pd.DataFrame(top_prefix_rows)
    fig3 = go.Figure()
    seg_colors = {"Agent Builder": "#00b4d8", "Viewer": "#555566", "Ghost": "#2d2d3a"}
    if not prefix_df.empty:
        path_order = prefix_df["path"].drop_duplicates().tolist()
        path_pos = {path: idx for idx, path in enumerate(path_order)}
        for segment in segments:
            sub = prefix_df[prefix_df["segment"] == segment]
            if sub.empty:
                continue
            fig3.add_trace(go.Bar(
                x=sub["users"],
                y=[path_pos[path] for path in sub["path"]],
                orientation="h",
                name=segment,
                marker_color=seg_colors.get(segment, "#888"),
            ))
        fig3.update_yaxes(
            tickmode="array",
            tickvals=list(path_pos.values()),
            ticktext=list(path_pos.keys()),
            autorange="reversed",
        )
    fig3.update_layout(
        title="Top Early Paths By Segment<br><sup>Most common deduped path prefixes in the first few steps</sup>",
        template="plotly_dark",
        xaxis_title="Users",
        yaxis_title="Path",
        height=780,
        barmode="group",
        legend_title_text="",
    )
    write_html(fig3, f"{OUTPUT_DIR}/20_segment_path_prefixes.html")
    print(f"  Saved: {OUTPUT_DIR}/20_segment_path_prefixes.html")

    print("\nTop branch points:")
    if not top_branches.empty:
        print(
            top_branches[
                ["comparison", "prefix_len", "prefix", "next_event", "users", "branch_gap"]
            ]
            .round(2)
            .to_string(index=False)
        )

    print("\n[OK] Path branching model complete.")

# Zerve: call main() directly (no __main__ guard)
main()
