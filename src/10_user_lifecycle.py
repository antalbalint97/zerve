import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 10_user_lifecycle.py  --  User Eletut & Activation Elemzes
 Zerve Hackathon 2026
=============================================================
Input : zerve_events.csv
        outputs/user_features_segmented.parquet
Output: outputs/10_*.html

HAROM ELEMZES:

A) Time-to-Agent-Builder
   -- Mikor eri el a user a 3. build callt?
   -- Mennyi ido telik el signup es az "Agent Builder" hatarko kozott?
   -- Ez magyarazza a nov/dec kohorsz 0%-os AB aranyt

B) Activation Milestone-ok
   -- Mi az a legelso esemeny ami utan vki biztosan Agent Builder lesz?
   -- "Point of no return" azonositasa
   -- Ha X esemeny megtortent az elso 48h-ban -> Y% valoszinuseggel AB

C) User Eletut Tipikus Utvonalak
   -- Mi a leggyakoribb esemeny sorrend signup utan?
   -- Agent Builder vs Ghost utvonal kulonbsege
   -- Mikor agaznak el az utvonalak?
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from analytics.events import AGENT_TOOL_EVENTS, BUILD_EVENTS, is_build_event
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import COHORT_COLORS, write_html
warnings.filterwarnings("ignore")

DATA_PATH  = "data/zerve_events.csv"
FEAT_PATH  = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(FEAT_PATH)
df = load_events(DATA_PATH)

# Szegmens info visszacsatolasa a raw df-be
df = df.merge(feat[["segment", "signup_cohort"]].reset_index(),
              on="person_id", how="left")

print(f"  {len(feat):,} users  |  {len(df):,} events")

# ============================================================
# A) TIME-TO-AGENT-BUILDER
# ============================================================
print("\n-- A) Time-to-Agent-Builder --")
print("  Mikor eri el a user a 3. build callt? (signup utan)")

# Signup ideje userenként -- sign_up event VAGY elso event (fallback)
signup_times_explicit = (
    df[df["event"] == "sign_up"]
    .groupby("person_id")["timestamp"].min()
)
first_seen_times = df.groupby("person_id")["timestamp"].min()

# Merge: ha van sign_up event, azt hasznaljuk, egyebkent az elso eventet
signup_times = signup_times_explicit.combine_first(first_seen_times).rename("signup_at")

# Build call-ok per user -- event nev VAGY prop_tool_name alapjan
build_df = df[
    [
        is_build_event(event, tool_name)
        for event, tool_name in zip(df["event"], df["prop_tool_name"].fillna(""))
    ]
].sort_values(["person_id", "timestamp"]).copy()

print(f"  Build eventek szama: {len(build_df):,}")

# 3. build call ideje userenként
third_build = (
    build_df.assign(build_rank=build_df.groupby("person_id").cumcount())
    .loc[lambda d: d["build_rank"] == 2]
    .set_index("person_id")["timestamp"]
    .rename("third_build_at")
)

# Time to 3rd build call (napokban)
# Index tipusok egységesítése
signup_times.index  = signup_times.index.astype(str)
third_build.index   = third_build.index.astype(str)

ttab = pd.DataFrame({
    "signup_at"     : signup_times,
    "third_build_at": third_build,
}).dropna()

ttab["days_to_ab"] = (
    (ttab["third_build_at"] - ttab["signup_at"])
    .dt.total_seconds() / 86400
).clip(0, 120)

print(f"\n  Userek akik elertek a 3. build callt: {len(ttab):,}")
print(f"\n  Time-to-Agent-Builder statisztikak (napokban):")
desc = ttab["days_to_ab"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
for stat, val in desc.items():
    print(f"    {stat:<10}: {val:.1f}")

# Hany % eri el 7, 14, 30 napon belul?
for days in [1, 3, 7, 14, 30, 60]:
    pct = (ttab["days_to_ab"] <= days).mean() * 100
    print(f"    <= {days:>2} nap : {pct:.1f}% eri el")

# Chart A1: Time-to-AB eloszlas
fig_a1 = go.Figure()
fig_a1.add_trace(go.Histogram(
    x=ttab["days_to_ab"],
    nbinsx=40,
    marker_color="#00b4d8",
    name="Userek",
))
# Vertikalis vonalak
for days, label, color in [(7, "1 het", "#ff6b6b"), (14, "2 het", "#ffa500"), (30, "1 ho", "#90e0ef")]:
    fig_a1.add_vline(x=days, line_dash="dash", line_color=color,
                     annotation_text=label, annotation_position="top right")

fig_a1.update_layout(
    title="Time-to-Agent-Builder: Mennyi ido kell a 3. build callhoz?<br>"
          "<sup>Ez magyarazza a nov/dec kohorsz 0%-os Agent Builder aranyt</sup>",
    xaxis_title="Napok a signup utan",
    yaxis_title="Felhasznalok szama",
    template="plotly_dark",
    height=420,
)
write_html(fig_a1, f"{OUTPUT_DIR}/10_time_to_agent_builder.html")
print(f"\n  Saved: {OUTPUT_DIR}/10_time_to_agent_builder.html")

# Chart A2: Kohorszonkent mennyi ideig tart?
ttab_cohort = ttab.join(feat[["signup_cohort"]], how="left")
ttab_cohort = ttab_cohort.dropna(subset=["signup_cohort"])

fig_a2 = go.Figure()
for cohort in sorted(ttab_cohort["signup_cohort"].unique()):
    sub = ttab_cohort[ttab_cohort["signup_cohort"] == cohort]["days_to_ab"]
    if len(sub) < 3:
        continue
    fig_a2.add_trace(go.Box(
        y=sub,
        name=f"{cohort} (n={len(sub)})",
        marker_color=COHORT_COLORS.get(cohort, "#888"),
        boxmean=True,
    ))

fig_a2.update_layout(
    title="Time-to-Agent-Builder kohorszonkent<br>"
          "<sup>A regi kohorszoknak tobb idejuk volt Agent Builderré valni</sup>",
    yaxis_title="Napok a signup utan",
    template="plotly_dark",
    height=420,
)
write_html(fig_a2, f"{OUTPUT_DIR}/10_time_to_ab_by_cohort.html")
print(f"  Saved: {OUTPUT_DIR}/10_time_to_ab_by_cohort.html")

# ============================================================
# B) ACTIVATION MILESTONE-OK
# ============================================================
print("\n-- B) Activation Milestone-ok --")
print("  Mi tortenik az elso 48 oraban ami megjósolja az AB-va valast?")

# Minden usernel: mik tortentek az elso 48 oraban?
signup_times_all = (
    df.groupby("person_id")["timestamp"].min()
    .rename("first_seen")
)

df_with_first = df.merge(signup_times_all.reset_index(), on="person_id")
df_with_first["hours_since_signup"] = (
    (df_with_first["timestamp"] - df_with_first["first_seen"])
    .dt.total_seconds() / 3600
)

first_48h = df_with_first[df_with_first["hours_since_signup"] <= 48]

# Milestone-ok: kulcsesemanyek az elso 48h-ban
MILESTONES = {
    "ran_code_48h"          : lambda g: g["event"].isin({"run_block", "run_all_blocks"}).any(),
    "opened_agent_48h"      : lambda g: g["event"].isin({"agent_new_chat", "agent_start_from_prompt"}).any(),
    "used_agent_tool_48h"   : lambda g: g["event"].isin(AGENT_TOOL_EVENTS).any(),
    "created_block_48h"     : lambda g: g["event"].isin({"agent_tool_call_create_block_tool"}).any(),
    "refactored_48h"        : lambda g: g["event"].isin({"agent_tool_call_refactor_block_tool"}).any(),
    "reached_3_build_48h"   : lambda g: g["event"].isin(BUILD_EVENTS).sum() >= 3,
    "fullscreen_48h"        : lambda g: g["event"].isin({"fullscreen_open", "fullscreen_preview_output"}).any(),
    "returned_within_48h"   : lambda g: g["hours_since_signup"].max() >= 12,
    "had_10plus_events_48h" : lambda g: len(g) >= 10,
    "had_5plus_event_types" : lambda g: g["event"].nunique() >= 5,
}

milestone_df = first_48h.groupby("person_id").apply(
    lambda g: pd.Series({k: int(v(g)) for k, v in MILESTONES.items()})
).reset_index()

# Merge szegmenssel
milestone_df = milestone_df.merge(
    feat[["segment"]].reset_index(), on="person_id", how="left"
)
milestone_df["is_ab"] = (milestone_df["segment"] == "Agent Builder").astype(int)

print(f"\n  Milestone elemzes (n={len(milestone_df):,} user):")
print(f"  {'Milestone':<30} {'AB ha IGEN':>12} {'AB ha NEM':>12} {'Lift':>8} {'% users'}")
print("  " + "-" * 75)

milestone_stats = []
for m in MILESTONES.keys():
    did     = milestone_df[milestone_df[m] == 1]
    did_not = milestone_df[milestone_df[m] == 0]
    ab_if_did     = did["is_ab"].mean() * 100     if len(did) > 0     else 0
    ab_if_not     = did_not["is_ab"].mean() * 100 if len(did_not) > 0 else 0
    lift          = ab_if_did / ab_if_not if ab_if_not > 0 else 999
    pct_users     = len(did) / len(milestone_df) * 100
    milestone_stats.append({
        "milestone"   : m,
        "ab_if_did"   : round(ab_if_did, 1),
        "ab_if_not"   : round(ab_if_not, 1),
        "lift"        : round(lift, 1),
        "pct_users"   : round(pct_users, 1),
        "n_did"       : len(did),
    })
    print(f"  {m:<30} {ab_if_did:>11.1f}%  {ab_if_not:>11.1f}%  {lift:>7.1f}x  {pct_users:.1f}%")

ms_df = pd.DataFrame(milestone_stats).sort_values("lift", ascending=False)

# Chart B1: Milestone lift chart
fig_b1 = go.Figure()
fig_b1.add_trace(go.Bar(
    x=ms_df["milestone"],
    y=ms_df["ab_if_did"],
    name="AB % ha megtortent",
    marker_color="#00b4d8",
    text=ms_df["ab_if_did"].round(1),
    texttemplate="%{text}%",
    textposition="outside",
))
fig_b1.add_trace(go.Bar(
    x=ms_df["milestone"],
    y=ms_df["ab_if_not"],
    name="AB % ha NEM tortent meg",
    marker_color="#555566",
    text=ms_df["ab_if_not"].round(1),
    texttemplate="%{text}%",
    textposition="outside",
))
fig_b1.update_layout(
    barmode="group",
    title="Activation Milestone-ok: Mi jósolja meg az Agent Builder-re valast?<br>"
          "<sup>Ha az elso 48 oraban megtortenik X, akkor Y% valoszinuseggel Agent Builder</sup>",
    yaxis_title="Agent Builder %",
    template="plotly_dark",
    height=480,
    xaxis_tickangle=20,
    legend=dict(orientation="h", y=-0.25),
)
write_html(fig_b1, f"{OUTPUT_DIR}/10_activation_milestones.html")
print(f"\n  Saved: {OUTPUT_DIR}/10_activation_milestones.html")

# Chart B2: Lift chart
ms_df_plot = ms_df.copy()
ms_df_plot["lift"] = ms_df_plot["lift"].clip(upper=50)
fig_b2 = px.bar(
    ms_df_plot,
    x="milestone", y="lift",
    color="pct_users",
    color_continuous_scale="Blues",
    title="Activation Lift: Hanyszor valoszinubb az AB-va valas ha X megtortenik?<br>"
          "<sup>Szin = hany % userenel tortenik meg az esemeny (ritka de eros vs gyakori de gyenge)</sup>",
    labels={"lift": "Lift (x-szeres)", "pct_users": "% users"},
    template="plotly_dark",
    height=440,
    text=ms_df_plot["lift"].round(1),
)
fig_b2.update_traces(texttemplate="%{text}x", textposition="outside")
fig_b2.update_layout(xaxis_tickangle=20)
write_html(fig_b2, f"{OUTPUT_DIR}/10_activation_lift.html")
print(f"  Saved: {OUTPUT_DIR}/10_activation_lift.html")

# ============================================================
# C) USER ELETUT TIPIKUS UTVONALAK
# ============================================================
print("\n-- C) User Eletut Tipikus Utvonalak --")

# Egyszerusitett event kategorizalas
def categorize_event(e):
    if e in {"sign_up", "sign_in"}:                     return "AUTH"
    if "onboarding" in e or "skip_onboarding" in e:     return "ONBOARD"
    if e in {"run_block", "run_all_blocks"}:             return "RUN"
    if e in {"agent_new_chat", "agent_start_from_prompt", "agent_message"}: return "AGENT_CHAT"
    if e == "agent_tool_call_create_block_tool":         return "AGENT_BUILD"
    if e == "agent_tool_call_run_block_tool":            return "AGENT_RUN"
    if e == "agent_tool_call_refactor_block_tool":       return "AGENT_REFACTOR"
    if e.startswith("agent_tool_call"):                  return "AGENT_OTHER"
    if e in {"block_create", "block_delete", "block_resize"}: return "CANVAS_EDIT"
    if "fullscreen" in e:                               return "VIEW_OUTPUT"
    if "credits" in e:                                  return "CREDITS"
    return "OTHER"

df["event_cat"] = df["event"].apply(categorize_event)

# Elso N esemeny per user (kategoriaként)
N_STEPS = 8

def get_first_n_steps(group, n=N_STEPS):
    sorted_events = group.sort_values("timestamp")["event_cat"].tolist()
    # Deduplikálás: egymás utani azonos kategoriakat osszevonjuk
    deduped = []
    for e in sorted_events:
        if not deduped or deduped[-1] != e:
            deduped.append(e)
    return deduped[:n]

print(f"  Elso {N_STEPS} esemeny kategoriak per user kiszamitasa...")
user_paths = (
    df.groupby("person_id")
    .apply(get_first_n_steps)
    .reset_index()
)
user_paths.columns = ["person_id", "path"]
user_paths["path_str"] = user_paths["path"].apply(lambda x: " -> ".join(x))
user_paths = user_paths.merge(feat[["segment"]].reset_index(), on="person_id", how="left")

# Top utvonalak Agent Builder-eknel
ab_paths = user_paths[user_paths["segment"] == "Agent Builder"]["path_str"]
ghost_paths = user_paths[user_paths["segment"] == "Ghost"]["path_str"]

print(f"\n  Top 10 Agent Builder utvonal:")
for path, n in ab_paths.value_counts().head(10).items():
    print(f"    {n:>4}x  {path}")

print(f"\n  Top 10 Ghost utvonal:")
for path, n in ghost_paths.value_counts().head(10).items():
    print(f"    {n:>4}x  {path}")

# Chart C1: Elso lepesek osszehasonlitasa
# Minden pozicioban (1-8) mi a legelterjedtebb esemeny AB vs Ghost
step_comparison = []
for seg, group in user_paths.groupby("segment"):
    if seg not in ["Agent Builder", "Ghost", "Manual Coder"]:
        continue
    for i in range(N_STEPS):
        step_events = group["path"].apply(
            lambda x: x[i] if len(x) > i else None
        ).dropna()
        if len(step_events) == 0:
            continue
        top_event = step_events.value_counts().index[0]
        top_pct   = step_events.value_counts().iloc[0] / len(step_events) * 100
        step_comparison.append({
            "segment": seg,
            "step"   : i + 1,
            "top_event": top_event,
            "pct"    : round(top_pct, 1),
        })

step_df = pd.DataFrame(step_comparison)
print(f"\n  Lepesenkenti leggyakoribb esemeny:")
for seg in ["Agent Builder", "Manual Coder", "Ghost"]:
    seg_steps = step_df[step_df["segment"] == seg].sort_values("step")
    steps_str = " -> ".join([
        f"{r['top_event']}({r['pct']:.0f}%)"
        for _, r in seg_steps.iterrows()
    ])
    print(f"  {seg:<20}: {steps_str}")

# Chart C1: Sankey-jellegű lépés összehasonlítás
fig_c1 = px.bar(
    step_df,
    x="step", y="pct",
    color="segment",
    facet_row="segment",
    text="top_event",
    title="Tipikus User Eletut Lepesenkent<br>"
          "<sup>Minden lepesnel a legelterjedtebb esemeny kategoria az adott szegmensben</sup>",
    labels={"step": "Lepesszam", "pct": "Felhasznalok %", "top_event": "Esemeny"},
    template="plotly_dark",
    color_discrete_map={
        "Agent Builder": "#00b4d8",
        "Manual Coder" : "#90e0ef",
        "Ghost"        : "#555566",
    },
    height=600,
)
fig_c1.update_traces(textposition="inside", textfont_size=9)
write_html(fig_c1, f"{OUTPUT_DIR}/10_user_path_steps.html")
print(f"\n  Saved: {OUTPUT_DIR}/10_user_path_steps.html")

# Chart C2: Mikor agaznak el az utvonalak?
# Az elso 3 lepesben hol ter el az AB vs Ghost
fig_c2 = make_subplots(
    rows=1, cols=3,
    subplot_titles=["1. Lepés", "2. Lepés", "3. Lepés"],
)
event_colors = {
    "AUTH": "#00b4d8", "ONBOARD": "#48cae4", "RUN": "#90e0ef",
    "AGENT_CHAT": "#0077b6", "AGENT_BUILD": "#023e8a", "AGENT_RUN": "#48cae4",
    "AGENT_REFACTOR": "#0096c7", "AGENT_OTHER": "#caf0f8",
    "CANVAS_EDIT": "#555566", "VIEW_OUTPUT": "#444455",
    "CREDITS": "#333344", "OTHER": "#222233",
}

for step_i in range(1, 4):
    step_data = []
    for seg in ["Agent Builder", "Ghost"]:
        paths_seg = user_paths[user_paths["segment"] == seg]["path"]
        events_at_step = paths_seg.apply(
            lambda x: x[step_i - 1] if len(x) >= step_i else "NONE"
        )
        vc = events_at_step.value_counts(normalize=True) * 100
        for event, pct in vc.head(5).items():
            step_data.append({
                "segment": seg, "event": event, "pct": round(pct, 1)
            })
    step_df_i = pd.DataFrame(step_data)
    for seg, color in [("Agent Builder", "#00b4d8"), ("Ghost", "#555566")]:
        sub = step_df_i[step_df_i["segment"] == seg]
        fig_c2.add_trace(go.Bar(
            name=f"{seg} (step {step_i})",
            x=sub["event"],
            y=sub["pct"],
            marker_color=color,
            showlegend=(step_i == 1),
            text=sub["pct"].round(1),
            texttemplate="%{text}%",
            textposition="outside",
        ), row=1, col=step_i)

fig_c2.update_layout(
    barmode="group",
    title="Hol agaznak el az utvonalak? Agent Builder vs Ghost<br>"
          "<sup>Az elso 3 lepesben mar latszik a kulonbseg</sup>",
    template="plotly_dark",
    height=480,
    legend=dict(orientation="h", y=-0.15),
)
write_html(fig_c2, f"{OUTPUT_DIR}/10_path_divergence.html")
print(f"  Saved: {OUTPUT_DIR}/10_path_divergence.html")

# ============================================================
# OSSZEFOGLALO
# ============================================================
print("\n-- Osszefoglalo --")
median_days = ttab["days_to_ab"].median()
pct_7d = (ttab["days_to_ab"] <= 7).mean() * 100
pct_30d = (ttab["days_to_ab"] <= 30).mean() * 100

print(f"\n  Time-to-Agent-Builder:")
print(f"    Median: {median_days:.1f} nap")
print(f"    {pct_7d:.1f}% eri el 7 napon belul")
print(f"    {pct_30d:.1f}% eri el 30 napon belul")
print(f"\n  --> Nov/Dec kohorsz magyarazata:")
print(f"    A dataset dec 8-an vegzodik.")
print(f"    Egy nov 1-en regisztralo usernek max 37 napja volt.")
print(f"    Mivel csak {pct_30d:.1f}% er el 30 napon belul Agent Builder szintet,")
print(f"    a nov kohorsz meg 'erlelodik' -- ez nem kudarc, hanem idohorizont.")

best_milestone = ms_df.iloc[0]
print(f"\n  Legerosebb activation milestone:")
print(f"    '{best_milestone['milestone']}'")
print(f"    AB arany ha megtortent: {best_milestone['ab_if_did']}%")
print(f"    AB arany ha NEM tortent meg: {best_milestone['ab_if_not']}%")
print(f"    Lift: {best_milestone['lift']}x")

print("\n[OK] User lifecycle & activation elemzes complete.")
