import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 02_feature_engineering.py  --  User-Level Feature Matrix
 Zerve Hackathon 2026
=============================================================
Input : zerve_events.csv
Output: outputs/user_features.parquet

Builds the user-level feature matrix from raw events.
One row = one unique person_id.
"""

import os

import numpy as np
import pandas as pd

from analytics.events import add_canvas_id, add_normalized_tool, reconstruct_sessions
from analytics.io import load_events

DATA_PATH = "zerve_events.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AGENT_TOOL_EVENTS = {
    "agent_tool_call_create_block_tool",
    "agent_tool_call_run_block_tool",
    "agent_tool_call_get_block_tool",
    "agent_tool_call_get_canvas_summary_tool",
    "agent_tool_call_refactor_block_tool",
    "agent_tool_call_finish_ticket_tool",
    "agent_tool_call_get_variable_preview_tool",
    "agent_tool_call_delete_block_tool",
    "agent_tool_call_create_edges_tool",
    "agent_tool_call_get_block_image_tool",
}
MANUAL_RUN_EVENTS = {"run_block", "run_all_blocks"}
AGENT_START_EVENTS = {"agent_new_chat", "agent_start_from_prompt", "agent_message"}
ONBOARDING_EVENTS = {
    "sign_up",
    "sign_in",
    "skip_onboarding_form",
    "submit_onboarding_form",
    "canvas_onboarding_tour_finished",
}
OUTPUT_EVENTS = {"fullscreen_open", "fullscreen_close", "fullscreen_preview_output"}
CREDIT_EVENTS = {
    "credits_used",
    "addon_credits_used",
    "credits_exceeded",
    "credits_below_1",
    "credits_below_2",
    "credits_below_3",
    "credits_below_4",
}


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    volume = df.groupby("person_id").agg(
        total_events=("event", "count"),
        unique_event_types=("event", "nunique"),
        days_active=("timestamp", lambda x: x.dt.normalize().nunique()),
        unique_sessions=("prop_$session_id", "nunique"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )
    volume["tenure_days"] = (
        (volume["last_seen"] - volume["first_seen"]).dt.total_seconds() / 86400
    ).clip(lower=0)
    volume["signup_cohort"] = volume["first_seen"].dt.strftime("%Y-%m")
    return volume


def build_canvas_features(df: pd.DataFrame, index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    canvas_df = add_canvas_id(df[df["prop_$pathname"].fillna("").str.contains("/canvas/", na=False)].copy())
    canvas_feats = (
        canvas_df.groupby("person_id").agg(unique_canvases=("canvas_id", "nunique"))
        .reindex(index, fill_value=0)
    )
    canvas_commitment = pd.DataFrame(index=index)
    if len(canvas_df) == 0:
        return canvas_feats, canvas_commitment

    canvas_event_counts = (
        canvas_df.groupby(["person_id", "canvas_id"]).size().rename("event_count").reset_index()
    )
    total_per_user = canvas_event_counts.groupby("person_id")["event_count"].sum()
    max_per_user = canvas_event_counts.groupby("person_id")["event_count"].max()
    canvas_active_days = (
        canvas_df.groupby(["person_id", "canvas_id"])["timestamp"]
        .agg(lambda x: x.dt.normalize().nunique())
        .rename("active_days")
        .reset_index()
    )
    canvas_groups = canvas_active_days.groupby("person_id")["active_days"]
    repeat_counts = canvas_groups.agg(lambda x: (x >= 2).sum())
    one_day_counts = canvas_groups.agg(lambda x: (x == 1).sum())
    total_canvases = canvas_groups.size()

    canvas_commitment = pd.DataFrame(
        {
            "primary_canvas_event_share": (max_per_user / total_per_user).replace([np.inf, -np.inf], 0),
            "repeat_canvas_count_raw": repeat_counts,
            "one_day_canvas_share": (one_day_counts / total_canvases).replace([np.inf, -np.inf], 0),
        }
    ).reindex(index, fill_value=0)
    return canvas_feats, canvas_commitment


def build_manual_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return (
        df[df["event"].isin(MANUAL_RUN_EVENTS)]
        .groupby("person_id")
        .agg(
            manual_runs=("event", "count"),
            manual_run_days=("timestamp", lambda x: x.dt.normalize().nunique()),
        )
        .reindex(index, fill_value=0)
    )


def build_agent_start_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return (
        df[df["event"].isin(AGENT_START_EVENTS)]
        .groupby("person_id")
        .agg(
            agent_conversations=("prop_message_id", "nunique"),
            agent_messages=("event", "count"),
            agent_start_days=("timestamp", lambda x: x.dt.normalize().nunique()),
        )
        .reindex(index, fill_value=0)
    )


def build_agent_tool_features(df: pd.DataFrame, index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tool_df = add_normalized_tool(df[df["event"].isin(AGENT_TOOL_EVENTS)].copy())
    tool_df["tool_name"] = tool_df["tool"]

    tool_summary = (
        tool_df.groupby("person_id")
        .agg(
            agent_tool_calls_total=("event", "count"),
            agent_tool_types_used=("tool_name", "nunique"),
            agent_tool_call_days=("timestamp", lambda x: x.dt.normalize().nunique()),
        )
        .reindex(index, fill_value=0)
    )

    tool_pivot = (
        tool_df.groupby(["person_id", "tool_name"]).size().unstack(fill_value=0).reindex(index, fill_value=0)
    )
    tool_pivot.columns = [f"tool_{c.replace(' ', '_').lower()}" for c in tool_pivot.columns]

    def tool_sum(names):
        cols = [c for c in tool_pivot.columns if any(name in c for name in names)]
        return tool_pivot[cols].sum(axis=1) if cols else pd.Series(0, index=tool_pivot.index)

    tool_groups = pd.DataFrame(
        {
            "agent_build_calls": tool_sum(["create_block", "refactor_block", "create_edges"]),
            "agent_run_calls": tool_sum(["run_block"]),
            "agent_inspect_calls": tool_sum(["get_block", "get_canvas", "get_variable", "get_block_image"]),
            "agent_finish_calls": tool_sum(["finish_ticket"]),
            "agent_delete_calls": tool_sum(["delete_block"]),
        },
        index=index,
    )

    tool_sequence_feats = pd.DataFrame(index=index)
    if len(tool_df) > 0:
        tool_seq = tool_df.sort_values(["person_id", "timestamp"]).copy()
        tool_seq["tool_short"] = tool_seq["tool_name"].fillna("").str.replace("_tool", "", regex=False)
        tool_seq["next_tool_short"] = tool_seq.groupby("person_id")["tool_short"].shift(-1)
        transitions = tool_seq.dropna(subset=["tool_short", "next_tool_short"]).copy()
        transitions["is_create_run"] = (
            (transitions["tool_short"] == "create_block") & (transitions["next_tool_short"] == "run_block")
        ).astype(int)
        transitions["is_run_refactor"] = (
            (transitions["tool_short"] == "run_block") & (transitions["next_tool_short"] == "refactor_block")
        ).astype(int)
        transitions["is_finish_summary"] = (
            (transitions["tool_short"] == "finish_ticket")
            & (transitions["next_tool_short"] == "get_canvas_summary")
        ).astype(int)
        tool_sequence_feats = (
            transitions.groupby("person_id")
            .agg(
                create_run_transitions=("is_create_run", "sum"),
                run_refactor_transitions=("is_run_refactor", "sum"),
                finish_summary_transitions=("is_finish_summary", "sum"),
                agent_tool_transitions_total=("is_create_run", "count"),
            )
            .reindex(index, fill_value=0)
        )
        denom = tool_sequence_feats["agent_tool_transitions_total"].replace(0, 1)
        tool_sequence_feats["create_run_alternation_rate"] = tool_sequence_feats["create_run_transitions"] / denom
        tool_sequence_feats["refactor_after_run_rate"] = tool_sequence_feats["run_refactor_transitions"] / denom
        tool_sequence_feats["finish_summary_rate"] = tool_sequence_feats["finish_summary_transitions"] / denom

    return tool_summary, tool_groups, tool_pivot.join(tool_sequence_feats, how="left").fillna(0)


def build_output_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    output_df = df[df["event"].isin(OUTPUT_EVENTS)].copy()
    grouped = output_df.groupby("person_id")["event"]
    out = pd.DataFrame(index=index)
    out["fullscreen_opens"] = grouped.agg(lambda x: (x == "fullscreen_open").sum()).reindex(index, fill_value=0)
    out["previewed_output"] = grouped.agg(lambda x: int("fullscreen_preview_output" in x.values)).reindex(index, fill_value=0)
    return out.fillna(0)


def build_onboarding_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    onb_df = df[df["event"].isin(ONBOARDING_EVENTS)].copy()
    grouped = onb_df.groupby("person_id")["event"]
    out = pd.DataFrame(index=index)
    out["signed_up"] = grouped.agg(lambda x: int("sign_up" in x.values)).reindex(index, fill_value=0)
    out["completed_onboarding"] = grouped.agg(
        lambda x: int("canvas_onboarding_tour_finished" in x.values)
    ).reindex(index, fill_value=0)
    out["skipped_onboarding_form"] = grouped.agg(
        lambda x: int("skip_onboarding_form" in x.values)
    ).reindex(index, fill_value=0)
    out["submitted_onboarding"] = grouped.agg(
        lambda x: int("submit_onboarding_form" in x.values)
    ).reindex(index, fill_value=0)
    out["sign_in_count"] = grouped.agg(lambda x: (x == "sign_in").sum()).reindex(index, fill_value=0)
    return out.fillna(0)


def build_time_to_first_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    signup_time = df[df["event"] == "sign_up"].groupby("person_id")["timestamp"].min()
    first_manual_run = df[df["event"].isin(MANUAL_RUN_EVENTS)].groupby("person_id")["timestamp"].min()
    first_agent_tool = df[df["event"].isin(AGENT_TOOL_EVENTS)].groupby("person_id")["timestamp"].min()
    first_agent_chat = df[df["event"].isin(AGENT_START_EVENTS)].groupby("person_id")["timestamp"].min()

    ttf = pd.DataFrame(
        {
            "signup_at": signup_time,
            "first_manual_run_at": first_manual_run,
            "first_agent_tool_at": first_agent_tool,
            "first_agent_chat_at": first_agent_chat,
        }
    ).reindex(index)

    def mins_since_signup(col: str) -> pd.Series:
        return ((ttf[col] - ttf["signup_at"]).dt.total_seconds() / 60).clip(0, 10080)

    out = pd.DataFrame(index=index)
    out["ttf_manual_run_min"] = mins_since_signup("first_manual_run_at")
    out["ttf_agent_tool_min"] = mins_since_signup("first_agent_tool_at")
    out["ttf_agent_chat_min"] = mins_since_signup("first_agent_chat_at")
    out["adopted_agent_early"] = (out["ttf_agent_tool_min"] <= 60).astype(int)
    out["ever_used_agent"] = ttf["first_agent_tool_at"].notna().astype(int)
    out["ever_ran_manually"] = ttf["first_manual_run_at"].notna().astype(int)
    return out


def build_credit_and_recency_features(df: pd.DataFrame, volume: pd.DataFrame, index: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    credit_df = df[df["event"].isin(CREDIT_EVENTS)].copy()
    grouped = credit_df.groupby("person_id")["event"]
    out = pd.DataFrame(index=index)
    out["credit_events"] = grouped.size().reindex(index, fill_value=0)
    out["had_credit_exceeded"] = grouped.agg(lambda x: int("credits_exceeded" in x.values)).reindex(index, fill_value=0)
    out["had_addon_credits"] = grouped.agg(lambda x: int("addon_credits_used" in x.values)).reindex(index, fill_value=0)
    dataset_end = df["timestamp"].max()
    recency = ((dataset_end - volume["last_seen"]).dt.total_seconds() / 86400).rename("recency_days")
    return out.fillna(0), recency


def build_first_session_features(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if "prop_$session_id" not in df.columns:
        return pd.DataFrame(index=index)

    first_session_id = (
        df.dropna(subset=["prop_$session_id"])
        .sort_values("timestamp")
        .groupby("person_id")["prop_$session_id"]
        .first()
        .rename("first_session_id")
    )
    tmp = df.dropna(subset=["prop_$session_id"]).merge(first_session_id.reset_index(), on="person_id")
    first_sess_df = tmp[tmp["prop_$session_id"] == tmp["first_session_id"]]
    return (
        first_sess_df.groupby("person_id")
        .agg(
            first_session_events=("event", "count"),
            first_session_event_types=("event", "nunique"),
            first_session_duration_min=("timestamp", lambda x: (x.max() - x.min()).total_seconds() / 60),
            first_session_had_agent=("event", lambda x: int(any(e in AGENT_TOOL_EVENTS | AGENT_START_EVENTS for e in x))),
            first_session_had_run=("event", lambda x: int(any(e in MANUAL_RUN_EVENTS for e in x))),
        )
        .reindex(index, fill_value=0)
    )


def build_return_and_session_structure_features(df: pd.DataFrame, index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "prop_$session_id" in df.columns:
        sess_times = (
            df.dropna(subset=["prop_$session_id"])
            .groupby(["person_id", "prop_$session_id"])["timestamp"]
            .min()
            .reset_index()
            .sort_values(["person_id", "timestamp"])
        )
        sess_times["sess_rank"] = sess_times.groupby("person_id").cumcount()
        first_sess_time = sess_times[sess_times["sess_rank"] == 0].set_index("person_id")["timestamp"]
        second_sess_time = sess_times[sess_times["sess_rank"] == 1].set_index("person_id")["timestamp"]
        return_gap = pd.DataFrame(
            {
                "had_second_session": second_sess_time.reindex(index).notna().astype(int),
                "time_to_return_hours": (
                    (second_sess_time.reindex(index) - first_sess_time.reindex(index)).dt.total_seconds() / 3600
                ).clip(0, 720).fillna(0),
            }
        )
    else:
        return_gap = pd.DataFrame(index=index)

    session_df = reconstruct_sessions(df[["person_id", "timestamp", "event"]].copy())
    session_structure = pd.DataFrame(index=index)
    if len(session_df) > 0:
        session_df["has_build_event"] = session_df["event"].isin(
            {"block_create", "agent_tool_call_create_block_tool", "agent_tool_call_refactor_block_tool"}
        ).astype(int)
        session_df["has_run_event"] = session_df["event"].isin(
            MANUAL_RUN_EVENTS | {"agent_tool_call_run_block_tool"}
        ).astype(int)
        productive = (
            session_df.groupby(["person_id", "derived_session_num"])
            .agg(
                has_build=("has_build_event", "max"),
                has_run=("has_run_event", "max"),
                event_count=("event", "count"),
            )
        )
        productive["is_productive_session"] = ((productive["has_build"] == 1) & (productive["has_run"] == 1)).astype(int)
        session_structure = (
            productive.groupby("person_id")
            .agg(
                productive_sessions=("is_productive_session", "sum"),
                total_sessions_derived=("is_productive_session", "count"),
                max_events_per_session=("event_count", "max"),
            )
            .reindex(index, fill_value=0)
        )
        session_structure["productive_session_share"] = (
            session_structure["productive_sessions"] / session_structure["total_sessions_derived"].replace(0, 1)
        )

    return return_gap, session_structure


def build_time_and_referrer_features(df: pd.DataFrame, index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    signup_events = df[df["event"] == "sign_up"].copy()
    if len(signup_events) > 0:
        signup_events["signup_hour"] = signup_events["timestamp"].dt.hour
        signup_events["signup_dow"] = signup_events["timestamp"].dt.dayofweek
        signup_events["signup_is_weekend"] = (signup_events["signup_dow"] >= 5).astype(int)
        time_feats = (
            signup_events.groupby("person_id")
            .agg(
                signup_hour=("signup_hour", "first"),
                signup_dow=("signup_dow", "first"),
                signup_is_weekend=("signup_is_weekend", "first"),
            )
            .reindex(index, fill_value=-1)
        )
    else:
        first_seen = df.groupby("person_id")["timestamp"].min()
        time_feats = pd.DataFrame(
            {
                "signup_hour": first_seen.dt.hour,
                "signup_dow": first_seen.dt.dayofweek,
                "signup_is_weekend": (first_seen.dt.dayofweek >= 5).astype(int),
            }
        ).reindex(index, fill_value=-1)

    referrer_col = "prop_$set_once.$initial_referring_domain"
    if referrer_col in df.columns:
        ref_df = df.dropna(subset=[referrer_col]).groupby("person_id")[referrer_col].first()

        def categorize_referrer(r):
            r = str(r).lower()
            if "google" in r:
                return "google"
            if "github" in r:
                return "github"
            if "linkedin" in r:
                return "linkedin"
            if "twitter" in r or "x.com" in r:
                return "twitter"
            if r in ("", "nan", "none", "$direct"):
                return "direct"
            return "other"

        ref_cat = ref_df.apply(categorize_referrer).rename("referrer_source")
        ref_dummies = pd.get_dummies(ref_cat, prefix="ref").reindex(index, fill_value=0)
    else:
        ref_dummies = pd.DataFrame(index=index)

    return time_feats, ref_dummies


def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  [1/9] Volume features...")
    volume = build_volume_features(df)
    index = volume.index

    print("  [2/9] Canvas features...")
    canvas_feats, canvas_commitment = build_canvas_features(df, index)

    print("  [3/9] Manual execution features...")
    manual_feats = build_manual_features(df, index)

    print("  [4/9] Agent start features...")
    agent_start_feats = build_agent_start_features(df, index)

    print("  [5/9] Agent tool features...")
    tool_summary, tool_groups, tool_features = build_agent_tool_features(df, index)

    print("  [6/9] Output features...")
    output_feats = build_output_features(df, index)

    print("  [7/9] Onboarding features...")
    onb_feats = build_onboarding_features(df, index)

    print("  [8/9] Time-to-first features...")
    ttf_feats = build_time_to_first_features(df, index)

    print("  [9/9] Credit + recency features...")
    credit_feats, recency = build_credit_and_recency_features(df, volume, index)

    print("  [10/12] First session features...")
    first_session_feats = build_first_session_features(df, index)

    print("  [11/12] Return pattern features...")
    return_gap, session_structure = build_return_and_session_structure_features(df, index)

    print("  [12/12] Time-of-day, referrer features...")
    time_feats, ref_dummies = build_time_and_referrer_features(df, index)

    print("\nMerging features...")
    features = (
        volume.join(canvas_feats)
        .join(manual_feats)
        .join(agent_start_feats)
        .join(tool_summary)
        .join(tool_groups)
        .join(tool_features)
        .join(output_feats)
        .join(onb_feats)
        .join(ttf_feats)
        .join(credit_feats)
        .join(recency)
        .join(first_session_feats)
        .join(return_gap)
        .join(session_structure)
        .join(time_feats)
        .join(canvas_commitment)
        .join(ref_dummies)
    )

    num_cols = features.select_dtypes(include=[np.number]).columns
    features[num_cols] = features[num_cols].fillna(0)
    bool_cols = features.select_dtypes(include=["bool"]).columns
    features[bool_cols] = features[bool_cols].astype(int)
    ref_cols = [c for c in features.columns if c.startswith("ref_")]
    for col in ref_cols:
        features[col] = features[col].astype(int)
    return features


def main() -> None:
    print("Loading data...")
    df = load_events(DATA_PATH)
    print(f"  {len(df):,} events  |  {df['person_id'].nunique():,} users")
    print(f"  Date range: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    features = assemble_features(df)

    print(f"\n-- Feature matrix: {features.shape} --")
    print(f"  Users   : {len(features):,}")
    print(f"  Features: {features.shape[1]}")
    print(
        f"\n  Agent-t valaha hasznalt   : {int(features['ever_used_agent'].sum()):,} "
        f"({features['ever_used_agent'].mean()*100:.1f}%)"
    )
    print(
        f"  Manualisan futtatott      : {int(features['ever_ran_manually'].sum()):,} "
        f"({features['ever_ran_manually'].mean()*100:.1f}%)"
    )
    print(
        f"  Volt masodik session      : {int(features['had_second_session'].sum()):,} "
        f"({features['had_second_session'].mean()*100:.1f}%)"
    )
    if "first_session_had_agent" in features.columns:
        print(
            f"  Elso sessionben agent     : {int(features['first_session_had_agent'].sum()):,} "
            f"({features['first_session_had_agent'].mean()*100:.1f}%)"
        )
    print("  Signup kohorszok:")
    for cohort, n in features["signup_cohort"].value_counts().sort_index().items():
        print(f"    {cohort}: {n:,} user")

    features.to_parquet(f"{OUTPUT_DIR}/user_features.parquet")
    features.to_csv(f"{OUTPUT_DIR}/user_features.csv")
    print(f"\n  Saved: {OUTPUT_DIR}/user_features.parquet")


if __name__ == "__main__":
    main()
