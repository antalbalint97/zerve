import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 11_tool_sequences_session_progression.py
 Zerve Hackathon 2026
=============================================================
Input : zerve_events.csv
        outputs/user_features_segmented.parquet
Output: outputs/11_*.html

KET ELEMZES:

A) Tool Sequence Patterns
   -- Milyen sorrendben hivjak az agent tool-okat?
   -- Vannak-e tipikus "workflow fingerprints"?
   -- Agent Builder vs Viewer tool szekvenciak

B) Session Progression
   -- Egyre komplexebb tool kombinaciokat hasznalnak-e?
   -- Skill progression: session 1 vs session 5 vs session 10
   -- A "depth before breadth" hipotezis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import os
import warnings
from analytics.events import AGENT_TOOL_EVENTS, add_normalized_tool
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import write_html
warnings.filterwarnings("ignore")

DATA_PATH  = "data/zerve_events.csv"
FEAT_PATH  = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(FEAT_PATH)
df = load_events(DATA_PATH)
df = df.merge(feat[["segment"]].reset_index(), on="person_id", how="left")

# Tool name normalizalas
df = add_normalized_tool(df)

# Csak agent tool eventek
tool_df = df[
    df["event"].isin(AGENT_TOOL_EVENTS) |
    df["prop_tool_name"].notna()
].copy()
tool_df = tool_df.dropna(subset=["tool"])
tool_df = tool_df[tool_df["tool"] != "Coder Agent"]  # worker type, nem tool

# Tool rovid nevek
TOOL_SHORT = {
    "create_block_tool"         : "CREATE",
    "run_block_tool"            : "RUN",
    "get_block_tool"            : "GET",
    "get_canvas_summary_tool"   : "SUMMARY",
    "refactor_block_tool"       : "REFACTOR",
    "finish_ticket_tool"        : "FINISH",
    "get_variable_preview_tool" : "PREVIEW",
    "delete_block_tool"         : "DELETE",
    "create_edges_tool"         : "EDGES",
}
tool_df["tool_short"] = tool_df["tool"].map(TOOL_SHORT).fillna(
    tool_df["tool"].str.replace("_tool", "").str.upper().str[:8]
)

print(f"  {len(tool_df):,} agent tool event  |  "
      f"{tool_df['person_id'].nunique():,} user hasznalt agentet")

# ============================================================
# A) TOOL SEQUENCE PATTERNS
# ============================================================
print("\n-- A) Tool Sequence Patterns --")

# Bigram es trigram elemzes -- milyen tool parok / triplet-ek fordulnak elo?
def get_ngrams(tokens, n):
    if not isinstance(tokens, list):
        tokens = list(tokens)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Userenként tool szekvencia
user_tool_seqs = (
    tool_df.sort_values(["person_id", "timestamp"])
    .groupby("person_id")["tool_short"]
    .apply(list)
)

# Bigrams szegmensenként
print("\n  Top bigram tool patok szegmensenként:")
for seg in ["Agent Builder", "Manual Coder", "Viewer"]:
    seg_users = feat[feat["segment"] == seg].index.astype(str)
    seg_seqs  = user_tool_seqs[user_tool_seqs.index.isin(seg_users)]
    if len(seg_seqs) == 0:
        continue
    bigrams = []
    for seq in seg_seqs:
        bigrams.extend(get_ngrams(seq, 2))
    top_bg = Counter(bigrams).most_common(8)
    print(f"\n  {seg} (n={len(seg_seqs)}):")
    for bg, n in top_bg:
        print(f"    {n:>5}x  {bg[0]} -> {bg[1]}")

# Trigrams Agent Buildernel
print("\n  Top Agent Builder trigram workflow-ok:")
ab_users = feat[feat["segment"] == "Agent Builder"].index.astype(str)
ab_seqs  = user_tool_seqs[user_tool_seqs.index.isin(ab_users)]
trigrams  = []
for seq in ab_seqs:
    trigrams.extend(get_ngrams(seq, 3))
top_tg = Counter(trigrams).most_common(10)
for tg, n in top_tg:
    print(f"    {n:>5}x  {tg[0]} -> {tg[1]} -> {tg[2]}")

# Chart A1: Bigram heatmap Agent Builder-eknel
bg_counter = Counter(bigrams) if len(ab_seqs) > 0 else Counter()
top_tools  = ["CREATE", "RUN", "GET", "REFACTOR", "FINISH", "SUMMARY", "PREVIEW", "DELETE"]
matrix     = pd.DataFrame(0, index=top_tools, columns=top_tools)
for (t1, t2), n in bg_counter.items():
    if t1 in top_tools and t2 in top_tools:
        matrix.loc[t1, t2] = n

# Normalizalas soronkent
matrix_pct = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100

fig_a1 = px.imshow(
    matrix_pct,
    text_auto=".0f",
    color_continuous_scale="Blues",
    title="Tool Transition Matrix -- Agent Builderek<br>"
          "<sup>Sor: elozi tool, Oszlop: koveto tool, Ertek: atmeneti valoszinuseg %</sup>",
    labels=dict(x="Koveto tool", y="Elozi tool", color="%"),
    template="plotly_dark",
    aspect="auto",
)
write_html(fig_a1, f"{OUTPUT_DIR}/11_tool_transition_matrix.html")
print(f"\n  Saved: {OUTPUT_DIR}/11_tool_transition_matrix.html")

# Chart A2: Tool usage osszetétele szegmensenként
tool_by_seg = (
    tool_df.groupby(["segment", "tool_short"])
    .size()
    .reset_index(name="count")
)
# Csak a fobb szegmensek es top tool-ok
top_tools_list = tool_df["tool_short"].value_counts().head(8).index.tolist()
tool_by_seg_top = tool_by_seg[
    tool_by_seg["segment"].isin(["Agent Builder", "Manual Coder", "Viewer"]) &
    tool_by_seg["tool_short"].isin(top_tools_list)
]
# Normalizalas szegmensen belul
tool_by_seg_top = tool_by_seg_top.copy()
seg_totals = tool_by_seg_top.groupby("segment")["count"].transform("sum")
tool_by_seg_top["pct"] = tool_by_seg_top["count"] / seg_totals * 100

fig_a2 = px.bar(
    tool_by_seg_top,
    x="tool_short", y="pct", color="segment",
    barmode="group",
    title="Tool Hasznalat Osszetétele Szegmensenként (%)<br>"
          "<sup>Agent Builder vs Manual Coder vs Viewer tool mix</sup>",
    template="plotly_dark",
    color_discrete_map={
        "Agent Builder": "#00b4d8",
        "Manual Coder" : "#90e0ef",
        "Viewer"       : "#555566",
    },
    labels={"tool_short": "Tool", "pct": "Arany %", "segment": ""},
    height=420,
)
write_html(fig_a2, f"{OUTPUT_DIR}/11_tool_mix_by_segment.html")
print(f"  Saved: {OUTPUT_DIR}/11_tool_mix_by_segment.html")

# ============================================================
# B) SESSION PROGRESSION
# ============================================================
print("\n-- B) Session Progression --")
print("  Egyre komplexebb tool kombinaciokat hasznalnak-e sessionrol sessionre?")

# ============================================================
print("\n-- B) Session Progression --")
print("  Idoablak-alapu session rekonstrukcio (30 perces gap = uj session)")
print("  (A backend agent tool eventeknél nincs prop_$session_id)")

SESSION_GAP_MIN = 30  # perc

# Userenként idorendbe rakjuk a tool eventeket
# majd 30 perces inaktivitas = uj session
tool_sorted = tool_df.sort_values(["person_id", "timestamp"]).copy()

# Gap kiszamitasa userenként
tool_sorted["prev_time"] = tool_sorted.groupby("person_id")["timestamp"].shift(1)
tool_sorted["gap_min"] = (
    (tool_sorted["timestamp"] - tool_sorted["prev_time"])
    .dt.total_seconds() / 60
)
# Uj session ha: elso event VAGY gap > 30 perc
tool_sorted["is_new_session"] = (
    tool_sorted["prev_time"].isna() |
    (tool_sorted["gap_min"] > SESSION_GAP_MIN)
).astype(int)

# Session sorszam userenként
tool_sorted["sess_num"] = tool_sorted.groupby("person_id")["is_new_session"].cumsum()

print(f"\n  Rekonstrualt sessionok szama: {tool_sorted.groupby(['person_id','sess_num']).ngroups:,}")
print(f"  Userek agent tool sessionokkal: {tool_sorted['person_id'].nunique():,}")

# Session metrikak
sess_metrics = (
    tool_sorted.groupby(["person_id", "sess_num"])
    .agg(
        tool_diversity = ("tool_short", "nunique"),
        build_calls    = ("tool_short", lambda x: x.isin(["CREATE", "REFACTOR"]).sum()),
        run_calls      = ("tool_short", lambda x: (x == "RUN").sum()),
        total_calls    = ("tool_short", "count"),
        has_finish     = ("tool_short", lambda x: int("FINISH" in x.values)),
        duration_min   = ("timestamp", lambda x: (x.max()-x.min()).total_seconds()/60),
    )
    .reset_index()
)
sess_metrics = sess_metrics.merge(
    feat[["segment"]].reset_index(), on="person_id", how="left"
)
sess_metrics["is_ab"] = (sess_metrics["segment"] == "Agent Builder").astype(int)

print(f"\n  Session metrikak (osszes): {len(sess_metrics):,} session")
print(f"  Agent Builder sessionok: {sess_metrics[sess_metrics['segment']=='Agent Builder']['sess_num'].count():,}")

# Progresszio: avg metrikak sessionkent (1-10. session)
prog = (
    sess_metrics[sess_metrics["sess_num"] <= 10]
    .groupby(["sess_num", "segment"])
    .agg(
        avg_diversity = ("tool_diversity", "mean"),
        avg_build     = ("build_calls", "mean"),
        avg_total     = ("total_calls", "mean"),
        avg_duration  = ("duration_min", "mean"),
        n_sessions    = ("total_calls", "count"),
    )
    .reset_index()
)
prog_ab = prog[prog["segment"] == "Agent Builder"]

print("\n  Agent Builder session progresszio:")
print(f"  {'Sess':>5} {'Diversity':>10} {'Build':>8} {'Total':>8} {'Duration':>10} {'N':>6}")
for _, row in prog_ab.iterrows():
    bar = "=" * int(row["avg_diversity"] * 2)
    print(f"  {int(row['sess_num']):>5} {row['avg_diversity']:>10.2f} "
          f"{row['avg_build']:>8.2f} {row['avg_total']:>8.1f} "
          f"{row['avg_duration']:>9.1f}m {int(row['n_sessions']):>6}  [{bar}]")

# Chart B1: Session progresszio -- tool diversity
fig_b1 = go.Figure()
for seg, color in [("Agent Builder", "#00b4d8"), ("Manual Coder", "#90e0ef"), ("Viewer", "#555566")]:
    sub = prog[prog["segment"] == seg].sort_values("sess_num")
    if len(sub) == 0:
        continue
    fig_b1.add_trace(go.Scatter(
        x=sub["sess_num"],
        y=sub["avg_diversity"],
        mode="lines+markers",
        name=seg,
        line=dict(color=color, width=3),
        marker=dict(size=8),
    ))
fig_b1.update_layout(
    title="Session Progresszio -- Tool Diversity<br>"
          "<sup>Egyre tobb fele tool-t hasznal-e sessionrol sessionre? (30 perces gap = uj session)</sup>",
    xaxis_title="Session sorszam (idoablak-alapu)",
    yaxis_title="Atlag egyedi tool tipusok szama",
    template="plotly_dark",
    height=420,
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)
fig_b1.write_html(f"{OUTPUT_DIR}/11_session_progression_diversity.html")
print(f"\n  Saved: {OUTPUT_DIR}/11_session_progression_diversity.html")

# Chart B2: Build calls sessionkent
fig_b2 = go.Figure()
for seg, color in [("Agent Builder", "#00b4d8"), ("Manual Coder", "#90e0ef")]:
    sub = prog[prog["segment"] == seg].sort_values("sess_num")
    if len(sub) == 0:
        continue
    fig_b2.add_trace(go.Bar(
        x=sub["sess_num"],
        y=sub["avg_build"].round(2),
        name=seg,
        marker_color=color,
    ))
fig_b2.update_layout(
    barmode="group",
    title="Build Call Novekedes Sessionkent<br>"
          "<sup>Agent Builder-ek egyre tobb blokkot epitenek sessionrol sessionre?</sup>",
    xaxis_title="Session sorszam",
    yaxis_title="Atlag build call / session",
    template="plotly_dark",
    height=400,
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)
fig_b2.write_html(f"{OUTPUT_DIR}/11_session_progression_builds.html")
print(f"  Saved: {OUTPUT_DIR}/11_session_progression_builds.html")

# Chart B3: Elso session vs kesobb -- tool mix valtozasa
tool_sorted["session_phase"] = tool_sorted["sess_num"].apply(
    lambda x: "1. session" if x == 1
    else "2-3. session" if x <= 3
    else "4+ session"
)
phase_tools = (
    tool_sorted[tool_sorted["segment"] == "Agent Builder"]
    .groupby(["session_phase", "tool_short"])
    .size()
    .reset_index(name="count")
)
phase_totals = phase_tools.groupby("session_phase")["count"].transform("sum")
phase_tools["pct"] = phase_tools["count"] / phase_totals * 100
top_tools_phase = phase_tools.groupby("tool_short")["count"].sum().nlargest(7).index
phase_tools_top = phase_tools[phase_tools["tool_short"].isin(top_tools_phase)]

fig_b3 = px.bar(
    phase_tools_top,
    x="tool_short", y="pct", color="session_phase",
    barmode="group",
    title="Agent Builder Tool Mix: Elso Session vs Kesobb<br>"
          "<sup>Hogyan valtozik a tool hasznalat ahogy a user tapasztalatot szerez?</sup>",
    template="plotly_dark",
    color_discrete_sequence=["#00b4d8", "#48cae4", "#90e0ef"],
    labels={"tool_short": "Tool", "pct": "Arany %", "session_phase": ""},
    height=420,
)
fig_b3.write_html(f"{OUTPUT_DIR}/11_tool_evolution_by_phase.html")
print(f"  Saved: {OUTPUT_DIR}/11_tool_evolution_by_phase.html")

# Chart B4: Depth score sessionkent
tool_sorted["depth_score"] = (
    (tool_sorted["tool_short"] == "CREATE").astype(int) +
    (tool_sorted["tool_short"] == "REFACTOR").astype(int) * 2 +
    (tool_sorted["tool_short"] == "FINISH").astype(int) * 3
)
depth_by_sess = (
    tool_sorted[tool_sorted["segment"] == "Agent Builder"]
    .groupby(["person_id", "sess_num"])["depth_score"]
    .sum()
    .reset_index()
    .groupby("sess_num")
    .agg(avg_depth=("depth_score", "mean"), n=("depth_score", "count"))
    .reset_index()
)
depth_by_sess = depth_by_sess[depth_by_sess["sess_num"] <= 10]

fig_b4 = px.line(
    depth_by_sess,
    x="sess_num", y="avg_depth",
    markers=True,
    title="Workflow Complexity Score Sessionkent -- Agent Builderek<br>"
          "<sup>CREATE=1pt, REFACTOR=2pt, FINISH=3pt -- egyre komplexebb workflow-kat epitenek?</sup>",
    template="plotly_dark",
    labels={"sess_num": "Session sorszam", "avg_depth": "Atlag complexity score"},
    height=380,
)
fig_b4.update_traces(line_color="#00b4d8", line_width=3, marker_size=10)
fig_b4.write_html(f"{OUTPUT_DIR}/11_workflow_complexity_progression.html")
print(f"  Saved: {OUTPUT_DIR}/11_workflow_complexity_progression.html")

# Chart B5: Session idotartam progresszio
fig_b5 = go.Figure()
sub = prog_ab.sort_values("sess_num")
fig_b5.add_trace(go.Scatter(
    x=sub["sess_num"], y=sub["avg_duration"].round(1),
    mode="lines+markers", name="Avg session hossz (perc)",
    line=dict(color="#00b4d8", width=3), marker=dict(size=10),
))
fig_b5.update_layout(
    title="Session Idotartam Novekedes -- Agent Builderek<br>"
          "<sup>Egyre hosszabb agent sessionokat tart-e a user?</sup>",
    xaxis_title="Session sorszam",
    yaxis_title="Atlag session hossz (perc)",
    template="plotly_dark",
    height=380,
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)
fig_b5.write_html(f"{OUTPUT_DIR}/11_session_duration_progression.html")
print(f"  Saved: {OUTPUT_DIR}/11_session_duration_progression.html")

# ============================================================
# OSSZEFOGLALO
# ============================================================
print("\n-- Osszefoglalo --")
print("\n  Tool sequence key findings:")
if len(top_tg) > 0:
    print(f"  Leggyakoribb Agent Builder trigram: "
          f"{top_tg[0][0][0]} -> {top_tg[0][0][1]} -> {top_tg[0][0][2]} "
          f"({top_tg[0][1]}x)")

print("\n  Session progresszio key findings:")
if len(prog_ab) >= 3:
    d1 = prog_ab[prog_ab["sess_num"] == 1]["avg_diversity"].values
    d3 = prog_ab[prog_ab["sess_num"] == 3]["avg_diversity"].values
    if len(d1) > 0 and len(d3) > 0:
        print(f"  Tool diversity: session 1 = {d1[0]:.2f}, session 3 = {d3[0]:.2f}")
        if d3[0] > d1[0]:
            print(f"  --> Agent Builderek egyre tobb tool-t hasznalnak "
                  f"(+{d3[0]-d1[0]:.2f} diversity noveledes)")
        else:
            print(f"  --> Tool diversity stabil -- specializalt, ismetlodo workflow")

    c1 = depth_by_sess[depth_by_sess["sess_num"] == 1]["avg_depth"].values
    c3 = depth_by_sess[depth_by_sess["sess_num"] == 3]["avg_depth"].values if len(depth_by_sess) >= 3 else []
    if len(c1) > 0 and len(c3) > 0:
        print(f"  Complexity score: session 1 = {c1[0]:.1f}, session 3 = {c3[0]:.1f}")
        if c3[0] > c1[0]:
            print(f"  --> Egyre komplexebb workflow-kat epitenek sessionrol sessionre!")

print("\n[OK] Tool sequences + Session progression complete.")
