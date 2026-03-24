import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from collections import Counter

import pandas as pd
import plotly.express as px

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import write_html

warnings.filterwarnings("ignore")

DATA_PATH = "zerve_events.csv"
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

    fig1 = px.bar(
        step_df,
        x="step",
        y="pct",
        color="event",
        facet_row="segment",
        title="Early Path Step Mix By Segment<br><sup>Where Ghost, Viewer, and Builder journeys start to diverge</sup>",
        template="plotly_dark",
        labels={"step": "Path step", "pct": "Users %", "event": "Event"},
        height=720,
    )
    write_html(fig1, f"{OUTPUT_DIR}/20_path_step_mix.html")
    print(f"  Saved: {OUTPUT_DIR}/20_path_step_mix.html")

    top_branches = branch_summary.sort_values("branch_gap", ascending=False).head(20)
    fig2 = px.bar(
        top_branches,
        x="branch_gap",
        y="prefix",
        color="next_event",
        facet_col="comparison",
        orientation="h",
        title="Top Early Branch Points<br><sup>Prefixes whose next step most strongly separates outcomes</sup>",
        template="plotly_dark",
        labels={"branch_gap": "Branch gap (pp)", "prefix": "Prefix", "next_event": "Next event"},
        height=620,
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    write_html(fig2, f"{OUTPUT_DIR}/20_top_branch_points.html")
    print(f"  Saved: {OUTPUT_DIR}/20_top_branch_points.html")

    top_prefix_rows = []
    for segment in segments:
        counts = user_paths[user_paths["segment"] == segment]["path_str"].value_counts().head(8)
        for prefix, n in counts.items():
            top_prefix_rows.append({"segment": segment, "path": prefix, "users": n})
    prefix_df = pd.DataFrame(top_prefix_rows)
    fig3 = px.bar(
        prefix_df,
        x="users",
        y="path",
        color="segment",
        facet_row="segment",
        orientation="h",
        title="Top Early Paths By Segment<br><sup>Most common deduped path prefixes in the first few steps</sup>",
        template="plotly_dark",
        labels={"users": "Users", "path": "Path", "segment": ""},
        height=780,
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


if __name__ == "__main__":
    main()
