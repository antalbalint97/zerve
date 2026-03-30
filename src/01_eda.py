import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 01_eda.py  --  Exploratory Data Analysis
 Zerve Hackathon 2026: "What Drives Successful Usage?"
=============================================================
Run this as the first block in your Zerve canvas.
Output: printed summaries + Plotly HTML charts saved to /outputs
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_raw_events
from analytics.viz import write_html

# -- CONFIG ------------------------------------------------
DATA_PATH   = "data/zerve_events.csv"
ensure_output_dir(OUTPUT_DIR)

# -- 1. LOAD -----------------------------------------------
print("Loading data...")
df = load_raw_events(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# -- 2. BASIC STATS ----------------------------------------
print("\n-- Dataset overview --")
print(f"  Date range  : {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
print(f"  Unique users (distinct_id) : {df['distinct_id'].nunique():,}")
print(f"  Unique users (person_id)   : {df['person_id'].nunique():,}")
print(f"  Unique events              : {df['event'].nunique():,}")
print(f"  Missing timestamp          : {df['timestamp'].isna().sum():,}")

# -- 3. EVENT FREQUENCY ------------------------------------
event_counts = df["event"].value_counts().reset_index()
event_counts.columns = ["event", "count"]

print(f"\n-- Top 30 events --")
print(event_counts.head(30).to_string(index=False))

fig_events = px.bar(
    event_counts.head(40),
    x="count", y="event",
    orientation="h",
    title="Top 40 Events by Frequency",
    color="count",
    color_continuous_scale="Teal",
    template="plotly_dark",
)
fig_events.update_layout(yaxis=dict(autorange="reversed"), height=900)
write_html(fig_events, f"{OUTPUT_DIR}/01_event_frequency.html")
print(f"  OK Saved: {OUTPUT_DIR}/01_event_frequency.html")

# -- 4. SOURCE SPLIT: web vs python SDK -------------------
if "prop_$lib" in df.columns:
    lib_split = df["prop_$lib"].value_counts()
    print(f"\n-- Event source (prop_$lib) --")
    print(lib_split.to_string())

# -- 5. ACTIVITY OVER TIME --------------------------------
df["hour"] = df["timestamp"].dt.floor("h")
hourly = df.groupby("hour").size().reset_index(name="events")

fig_time = px.line(
    hourly, x="hour", y="events",
    title="Event Volume Over Time",
    template="plotly_dark",
)
write_html(fig_time, f"{OUTPUT_DIR}/01_activity_timeline.html")
print(f"  OK Saved: {OUTPUT_DIR}/01_activity_timeline.html")

# -- 6. GEOGRAPHIC DISTRIBUTION ---------------------------
if "prop_$geoip_country_code" in df.columns:
    geo = (
        df.dropna(subset=["prop_$geoip_country_code"])
          .groupby("prop_$geoip_country_code")
          .agg(users=("distinct_id", "nunique"), events=("event", "count"))
          .reset_index()
          .sort_values("users", ascending=False)
    )
    print(f"\n-- Top 15 countries (by unique users) --")
    print(geo.head(15).to_string(index=False))

    fig_geo = px.choropleth(
        geo,
        locations="prop_$geoip_country_code",
        color="users",
        title="Unique Users by Country",
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    write_html(fig_geo, f"{OUTPUT_DIR}/01_geo_distribution.html")
    print(f"  OK Saved: {OUTPUT_DIR}/01_geo_distribution.html")

# -- 7. DEVICE / BROWSER SPLIT ----------------------------
if "prop_$device_type" in df.columns:
    device = df["prop_$device_type"].value_counts()
    print(f"\n-- Device types --\n{device.to_string()}")

# -- 8. USER JOURNEY FUNNEL -------------------------------
funnel_events = [
    "sign_up",
    "skip_onboarding_form",
    "canvas_onboarding_tour_finished",
    "block_create",
    "run_block",
    "agent_new_chat",
    "agent_accept_suggestion",
    "agent_block_created",
    "requirements_build",
    "fullscreen_preview_output",
]

funnel_data = []
for event in funnel_events:
    users = df.loc[df["event"] == event, "person_id"].nunique()
    funnel_data.append({"stage": event, "users": users})

funnel_df = pd.DataFrame(funnel_data)
print(f"\n-- User journey funnel --")
print(funnel_df.to_string(index=False))

fig_funnel = go.Figure(go.Funnel(
    y=funnel_df["stage"],
    x=funnel_df["users"],
    textinfo="value+percent initial",
    marker=dict(color=[
        "#00b4d8", "#0096c7", "#0077b6", "#023e8a",
        "#48cae4", "#90e0ef", "#ade8f4", "#caf0f8",
        "#03045e", "#023e8a",
    ]),
))
fig_funnel.update_layout(
    title="User Journey Funnel",
    template="plotly_dark",
    height=600,
)
write_html(fig_funnel, f"{OUTPUT_DIR}/01_user_funnel.html")
print(f"  OK Saved: {OUTPUT_DIR}/01_user_funnel.html")

# -- 9. SESSION DEPTH -------------------------------------
if "prop_$session_id" in df.columns:
    session_depth = (
        df.dropna(subset=["prop_$session_id"])
          .groupby("prop_$session_id")["event"]
          .count()
          .reset_index(name="events_per_session")
    )
    print(f"\n-- Session depth stats --")
    print(session_depth["events_per_session"].describe())

    session_depth_clipped = session_depth.copy()
    session_depth_clipped["events_per_session"] = session_depth_clipped["events_per_session"].clip(upper=100)

    fig_session = px.histogram(
        session_depth_clipped,
        x="events_per_session",
        nbins=50,
        title="Events per Session Distribution",
        template="plotly_dark",
        color_discrete_sequence=["#00b4d8"],
    )
    write_html(fig_session, f"{OUTPUT_DIR}/01_session_depth.html")
    print(f"  OK Saved: {OUTPUT_DIR}/01_session_depth.html")

print("\n[OK]  EDA complete. Check the 'outputs/' folder for charts.")
