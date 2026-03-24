import re
from typing import Iterable

import numpy as np
import pandas as pd


CANVAS_ID_RE = re.compile(r"/canvas/([a-f0-9\-]{10,})")
SESSION_GAP_MIN = 30

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
BUILD_EVENTS = {
    "agent_tool_call_create_block_tool",
    "agent_tool_call_refactor_block_tool",
}
BUILD_TOOL_NAMES = {"create_block_tool", "refactor_block_tool"}
AGENT_START_EVENTS = {"agent_new_chat", "agent_start_from_prompt", "agent_message"}
MANUAL_RUN_EVENTS = {"run_block", "run_all_blocks", "run_from_block", "run_upto_block"}
STRUCTURAL_EVENTS = {
    "block_create",
    "block_delete",
    "block_copy",
    "block_paste",
    "block_rename",
    "block_resize",
    "edge_create",
    "edge_delete",
    "layer_create",
    "layer_delete",
    "layer_rename",
    "layer_duplicate",
    "canvas_create",
    "canvas_clone",
    "canvas_rename",
    "canvas_delete",
}
DEPENDENCY_EVENTS = {
    "requirements_build",
    "files_upload",
    "files_download",
    "files_update_lazy_load",
    "files_delete",
    "files_create_folder",
    "asset_add_to_canvas",
    "asset_create_connection",
}
OUTPUT_EVENTS = {
    "fullscreen_preview_output",
    "fullscreen_open",
    "block_output_download",
    "block_output_copy",
    "app_publish",
    "api_deploy",
    "hosted_apps_deploy",
}
ERROR_EVENTS = {"agent_open_error_assist", "credits_exceeded"}
ONBOARDING_EVENTS = {
    "sign_up",
    "skip_onboarding_form",
    "submit_onboarding_form",
    "canvas_onboarding_tour_finished",
}

EU_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
}


def normalize_tool_name(event: str, prop_tool_name: str | float | None) -> str | None:
    if isinstance(prop_tool_name, str) and prop_tool_name.strip():
        return prop_tool_name.strip()
    if isinstance(event, str) and event.startswith("agent_tool_call_"):
        return event.replace("agent_tool_call_", "", 1)
    return None


def add_normalized_tool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tool"] = [
        normalize_tool_name(event, tool)
        for event, tool in zip(out["event"], out.get("prop_tool_name", pd.Series(index=out.index)))
    ]
    return out


def extract_canvas_id(pathname: str | float | None) -> str | None:
    if not isinstance(pathname, str):
        return None
    match = CANVAS_ID_RE.search(pathname)
    return match.group(1) if match else None


def add_canvas_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["canvas_id"] = out.get("prop_$pathname", pd.Series(index=out.index)).map(extract_canvas_id)
    return out


def canonical_event_name(row: pd.Series) -> str:
    event = row.get("event")
    tool = row.get("tool")
    if isinstance(tool, str) and tool:
        short = tool.replace("_tool", "").replace(" ", "_").lower()
        if short != "coder_agent":
            return f"tool_{short}"
    if event in MANUAL_RUN_EVENTS:
        return "manual_run"
    if event in STRUCTURAL_EVENTS:
        return event
    if event in DEPENDENCY_EVENTS:
        return event
    if event in OUTPUT_EVENTS:
        return event
    if event in ONBOARDING_EVENTS:
        return event
    if event in ERROR_EVENTS:
        return event
    if event == "agent_new_chat" or event == "agent_start_from_prompt":
        return "agent_chat"
    if event == "agent_accept_suggestion":
        return "agent_accept_suggestion"
    if event == "credits_used" or event == "addon_credits_used":
        return "credit_use"
    if event == "canvas_open":
        return "canvas_open"
    return str(event)


def reconstruct_sessions(
    df: pd.DataFrame,
    person_col: str = "person_id",
    time_col: str = "timestamp",
    gap_min: int = SESSION_GAP_MIN,
    session_col: str = "derived_session_num",
) -> pd.DataFrame:
    out = df.sort_values([person_col, time_col]).copy()
    out["prev_time"] = out.groupby(person_col)[time_col].shift(1)
    out["gap_min"] = (
        (out[time_col] - out["prev_time"]).dt.total_seconds().div(60)
    )
    out["is_new_session"] = (
        out["prev_time"].isna() | (out["gap_min"] > gap_min)
    ).astype(int)
    out[session_col] = out.groupby(person_col)["is_new_session"].cumsum().astype(int)
    out = out.drop(columns=["prev_time", "gap_min", "is_new_session"])
    return out


def build_unified_session_id(df: pd.DataFrame) -> pd.Series:
    native = df.get("prop_$session_id")
    derived = df.get("derived_session_num")
    if native is not None:
        native = native.astype("string")
    if derived is None:
        derived = pd.Series(index=df.index, dtype="Int64")
    derived = derived.astype("Int64")
    session_id = []
    for idx in df.index:
        nval = native.loc[idx] if native is not None else pd.NA
        if pd.notna(nval) and str(nval).strip():
            session_id.append(f"web::{nval}")
        else:
            dval = derived.loc[idx]
            session_id.append(f"derived::{int(dval)}" if pd.notna(dval) else "derived::0")
    return pd.Series(session_id, index=df.index, dtype="string")


def get_user_country(df: pd.DataFrame) -> pd.Series:
    country_col = "prop_$geoip_country_code"
    if country_col not in df.columns:
        return pd.Series(index=pd.Index([], dtype=str), dtype="string")
    country = (
        df.dropna(subset=[country_col])
        .sort_values("timestamp")
        .groupby("person_id")[country_col]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .astype("string")
    )
    return country


def map_country_region(country_code: str | None) -> str:
    code = (country_code or "").upper()
    if code == "IN":
        return "India"
    if code == "US":
        return "US"
    if code in EU_COUNTRIES:
        return "EU"
    if not code:
        return "Unknown"
    return "Other"


def filter_supported_countries(values: Iterable[str], min_users: int) -> list[str]:
    counts = pd.Series(list(values)).value_counts()
    return counts[counts >= min_users].index.tolist()


def is_build_event(event: str | None, tool_name: str | None = None) -> bool:
    return (event in BUILD_EVENTS) or (tool_name in BUILD_TOOL_NAMES)
