import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
=============================================================
 12_credit_error_propensity.py
 Zerve Hackathon 2026
=============================================================
Input : zerve_events.csv
        outputs/user_features_segmented.parquet
Output: outputs/12_*.html

HAROM ELEMZES:

A) Credit Burn Rate
   -- Minel gyorsabban eget kreditet a user, annal elkotelezettebb?
   -- Kredit esemeny / aktiv nap mint engagement proxy
   -- Agent Builder vs Ghost kredit intenzitas

B) Error Assist Signal
   -- agent_open_error_assist: pozitiv vagy negativ jel?
   -- Aki hibaba fut es segitseget ker, az elkotelezettebb?
   -- "Productive struggle" hipotezis

C) Propensity Score Matching
   -- Kontrollaljuk a confounding-ot: signup forras, orszag, onboarding
   -- Az agent adoptacio CAUSALIS hatasa a retention-re?
   -- Agent user vs nem-agent user -- de azonos "hatterrel"
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from analytics.io import OUTPUT_DIR, ensure_output_dir, load_events, load_features
from analytics.viz import SEGMENT_COLORS, write_html
warnings.filterwarnings("ignore")

DATA_PATH  = "data/zerve_events.csv"
FEAT_PATH  = "outputs/user_features_segmented.parquet"
ensure_output_dir(OUTPUT_DIR)

print("Loading data...")
feat = load_features(FEAT_PATH)
df = load_events(DATA_PATH)
df = df.merge(feat[["segment", "signup_cohort"]].reset_index(),
              on="person_id", how="left")

print(f"  {len(feat):,} users  |  {len(df):,} events")

# ============================================================
# A) CREDIT BURN RATE
# ============================================================
print("\n-- A) Credit Burn Rate Elemzes --")

CREDIT_EVENTS = {"credits_used", "addon_credits_used"}

credit_df = df[df["event"].isin(CREDIT_EVENTS)].copy()
credit_counts = credit_df.groupby("person_id").size().rename("credit_events")

# Burn rate = kredit event / aktiv nap
burn_rate = pd.DataFrame({
    "credit_events": credit_counts,
    "days_active"  : feat["days_active"],
    "segment"      : feat["segment"],
    "signup_cohort": feat["signup_cohort"],
}).fillna({"credit_events": 0})

burn_rate["burn_rate"] = (
    burn_rate["credit_events"] / burn_rate["days_active"].replace(0, 1)
)

print("\n  Credit burn rate szegmensenként:")
br_stats = burn_rate.groupby("segment").agg(
    users        = ("burn_rate", "count"),
    avg_burn     = ("burn_rate", "mean"),
    median_burn  = ("burn_rate", "median"),
    pct_any_burn = ("credit_events", lambda x: (x > 0).mean() * 100),
).round(2)
print(br_stats.to_string())

# Chart A1: Burn rate eloszlas szegmensenként
fig_a1 = go.Figure()
for seg in ["Agent Builder", "Manual Coder", "Viewer"]:
    sub = burn_rate[
        (burn_rate["segment"] == seg) &
        (burn_rate["burn_rate"] > 0)
    ]["burn_rate"].clip(upper=200)
    if len(sub) < 3:
        continue
    fig_a1.add_trace(go.Box(
        y=sub,
        name=f"{seg} (n={len(sub)})",
        marker_color=SEGMENT_COLORS.get(seg, "#888"),
        boxmean=True,
    ))

fig_a1.update_layout(
    title="Credit Burn Rate Szegmensenként<br>"
          "<sup>Kredit esemenyek / aktiv nap -- csak aktiv kredit egetok</sup>",
    yaxis_title="Burn rate (kredit event / aktiv nap)",
    template="plotly_dark",
    height=420,
)
write_html(fig_a1, f"{OUTPUT_DIR}/12_credit_burn_rate.html")
print(f"\n  Saved: {OUTPUT_DIR}/12_credit_burn_rate.html")

# Chart A2: Burn rate es Agent Builder arany kapcsolata
burn_bins = [-0.1, 0, 5, 20, 50, 100, 999]
burn_labels = ["0 (nem eget)", "1-5", "6-20", "21-50", "51-100", "100+"]
burn_rate["burn_bucket"] = pd.cut(
    burn_rate["burn_rate"], bins=burn_bins, labels=burn_labels
)

burn_ab = burn_rate.groupby("burn_bucket", observed=True).agg(
    users  = ("segment", "count"),
    pct_ab = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
).reset_index()

print("\n  Burn rate bucket vs Agent Builder %:")
print(burn_ab.to_string(index=False))

fig_a2 = make_subplots(specs=[[{"secondary_y": True}]])
fig_a2.add_trace(go.Bar(
    x=burn_ab["burn_bucket"].astype(str),
    y=burn_ab["users"],
    name="Users", marker_color="#2d3a4a", opacity=0.8,
), secondary_y=False)
fig_a2.add_trace(go.Scatter(
    x=burn_ab["burn_bucket"].astype(str),
    y=burn_ab["pct_ab"],
    name="Agent Builder %", mode="lines+markers",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=10, symbol="diamond"),
), secondary_y=True)
fig_a2.update_layout(
    title="Credit Burn Rate vs Agent Builder Arany<br>"
          "<sup>Minel intenzivebben eget kreditet, annal valoszinubb Agent Builder?</sup>",
    template="plotly_dark", height=420,
)
fig_a2.update_yaxes(title_text="Users", secondary_y=False)
fig_a2.update_yaxes(title_text="Agent Builder %", secondary_y=True)
write_html(fig_a2, f"{OUTPUT_DIR}/12_burn_rate_vs_ab.html")
print(f"  Saved: {OUTPUT_DIR}/12_burn_rate_vs_ab.html")

# ============================================================
# B) ERROR ASSIST SIGNAL
# ============================================================
print("\n-- B) Error Assist Signal --")

error_users = df[df["event"] == "agent_open_error_assist"]["person_id"].unique()
feat["had_error_assist"] = feat.index.isin(error_users).astype(int)

n_error = feat["had_error_assist"].sum()
print(f"\n  Error assist hasznalok: {n_error:,} ({n_error/len(feat)*100:.1f}%)")

error_stats = feat.groupby("had_error_assist").agg(
    users     = ("segment", "count"),
    pct_ab    = ("segment", lambda x: (x == "Agent Builder").mean() * 100),
    pct_ghost = ("segment", lambda x: (x == "Ghost").mean() * 100),
    avg_days  = ("days_active", "mean"),
    avg_tools = ("agent_tool_calls_total", "mean"),
).round(2)
error_stats.index = ["Nem hasznalt", "Hasznalt error assist"]
print("\n  Error assist vs nem hasznalt:")
print(error_stats.to_string())

# Error assist timing -- mikor tortent az elso error assist?
first_error = (
    df[df["event"] == "agent_open_error_assist"]
    .groupby("person_id")["timestamp"].min()
    .rename("first_error_at")
)
first_seen = df.groupby("person_id")["timestamp"].min().rename("first_seen")

error_timing = pd.DataFrame({
    "first_seen"   : first_seen,
    "first_error_at": first_error,
}).dropna()
error_timing["hours_to_error"] = (
    (error_timing["first_error_at"] - error_timing["first_seen"])
    .dt.total_seconds() / 3600
).clip(0, 720)

print(f"\n  Elso error assist timing (ora):")
print(error_timing["hours_to_error"].describe().round(1).to_string())

# Chart B1: Error assist vs nem -- szegmens osszetétel
fig_b1 = go.Figure()
for seg, color in SEGMENT_COLORS.items():
    vals = feat.groupby("had_error_assist")["segment"].apply(
        lambda x: (x == seg).mean() * 100
    ).values
    if len(vals) < 2:
        continue
    fig_b1.add_trace(go.Bar(
        x=["Nem hasznalt error assist", "Hasznalt error assist"],
        y=vals,
        name=seg,
        marker_color=color,
    ))

fig_b1.update_layout(
    barmode="stack",
    title="Error Assist Signal: Pozitiv vagy Negativ Jel?<br>"
          "<sup>'Productive struggle' hipotezis -- aki hibaba fut es segitseget ker, elkotelezettebb?</sup>",
    yaxis_title="Arany %",
    template="plotly_dark",
    height=420,
)
write_html(fig_b1, f"{OUTPUT_DIR}/12_error_assist_signal.html")
print(f"\n  Saved: {OUTPUT_DIR}/12_error_assist_signal.html")

# ============================================================
# C) PROPENSITY SCORE MATCHING
# ============================================================
print("\n-- C) Propensity Score Matching --")
print("  Kerdes: az agent adoptacio CAUSALIS hatasa a retention-re?")
print("  Kontrollvaltozok: signup_cohort, orszag, onboarding, first_session")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Treatment: ever_used_agent
    # Outcome: is_agent_builder (retention proxy)
    # Confounders: signup_cohort, onboarding, first_session_events

    psm_feat = feat.copy()

    # Confounder feature-ok -- NEM tartalmazhat agent-specifikus dolgokat
    CONFOUNDERS = [
        "signed_up",
        "skipped_onboarding_form",
        "submitted_onboarding",
        "completed_onboarding",
        "signup_hour",
        "signup_is_weekend",
        "first_session_events",
        "first_session_duration_min",
    ]
    # Signup cohort one-hot
    cohort_dummies = pd.get_dummies(
        psm_feat["signup_cohort"], prefix="cohort"
    ).astype(int)
    psm_feat = psm_feat.join(cohort_dummies)
    CONFOUNDERS += cohort_dummies.columns.tolist()

    # Referrer
    ref_cols = [c for c in psm_feat.columns if c.startswith("ref_")]
    CONFOUNDERS += ref_cols

    X_conf = psm_feat[[c for c in CONFOUNDERS if c in psm_feat.columns]].fillna(0)
    T = psm_feat["ever_used_agent"]  # treatment
    Y = (psm_feat["segment"] == "Agent Builder").astype(int)  # outcome

    # Propensity score becslese
    ps_model = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    ps_model.fit(X_conf, T)
    psm_feat["propensity_score"] = ps_model.predict_proba(X_conf)[:, 1]

    print(f"\n  Propensity score eloszlas:")
    print(f"    Agent user-ek:     mean={psm_feat[T==1]['propensity_score'].mean():.3f}")
    print(f"    Nem agent user-ek: mean={psm_feat[T==0]['propensity_score'].mean():.3f}")

    # Nearest-neighbor matching (egyszerusitett -- caliper alapu)
    CALIPER = 0.05

    treated   = psm_feat[T == 1][["propensity_score"]].copy()
    untreated = psm_feat[T == 0][["propensity_score"]].copy()

    matched_pairs = []
    used_controls = set()

    for idx, row in treated.iterrows():
        ps = row["propensity_score"]
        # Legkozelebbi kontroll caliper-en belul
        candidates = untreated[
            (abs(untreated["propensity_score"] - ps) <= CALIPER) &
            (~untreated.index.isin(used_controls))
        ]
        if len(candidates) == 0:
            continue
        match_idx = (candidates["propensity_score"] - ps).abs().idxmin()
        matched_pairs.append((idx, match_idx))
        used_controls.add(match_idx)

    print(f"\n  Matched parok: {len(matched_pairs):,} "
          f"(a {T.sum():,} kezelt userbol)")

    if len(matched_pairs) > 10:
        treated_ids   = [p[0] for p in matched_pairs]
        control_ids   = [p[1] for p in matched_pairs]

        treated_df    = psm_feat.loc[treated_ids]
        control_df    = psm_feat.loc[control_ids]

        # ATT: Average Treatment Effect on the Treated
        att_ab   = treated_df["segment"].eq("Agent Builder").mean() - \
                   control_df["segment"].eq("Agent Builder").mean()
        att_days = treated_df["days_active"].mean() - control_df["days_active"].mean()
        att_ret  = treated_df["had_second_session"].mean() - control_df["had_second_session"].mean()

        print(f"\n  ATT (Average Treatment Effect on Treated):")
        print(f"    Agent Builder arany diff: +{att_ab*100:.1f}pp")
        print(f"    Aktiv napok diff:         +{att_days:.2f} nap")
        print(f"    Visszateres diff:         +{att_ret*100:.1f}pp")
        print(f"\n  Ellenorzo: matched csoportok ps egyenlosege:")
        print(f"    Kezelt PS:  {treated_df['propensity_score'].mean():.3f}")
        print(f"    Kontroll PS:{control_df['propensity_score'].mean():.3f}")

        # Chart C1: Propensity score eloszlas matching elott/utan
        fig_c1 = make_subplots(rows=1, cols=2,
            subplot_titles=["Matching ELOTT", "Matching UTAN"])

        for col, (tr_data, ct_data) in enumerate([
            (psm_feat[T==1]["propensity_score"],
             psm_feat[T==0]["propensity_score"]),
            (treated_df["propensity_score"],
             control_df["propensity_score"]),
        ], start=1):
            fig_c1.add_trace(go.Histogram(
                x=tr_data, name="Agent user", opacity=0.7,
                marker_color="#00b4d8", nbinsx=20,
                showlegend=(col==1),
            ), row=1, col=col)
            fig_c1.add_trace(go.Histogram(
                x=ct_data, name="Nem agent user", opacity=0.7,
                marker_color="#555566", nbinsx=20,
                showlegend=(col==1),
            ), row=1, col=col)

        fig_c1.update_layout(
            barmode="overlay",
            title="Propensity Score Matching -- Egyensulyositas<br>"
                  "<sup>Matching utan a ket csoport hasonlo hatterrel rendelkezik</sup>",
            template="plotly_dark",
            height=420,
        )
        write_html(fig_c1, f"{OUTPUT_DIR}/12_propensity_matching.html")
        print(f"\n  Saved: {OUTPUT_DIR}/12_propensity_matching.html")

        # Chart C2: ATT vizualizacio
        att_results = pd.DataFrame({
            "metrika" : ["Agent Builder %", "Aktiv napok", "Visszateres %"],
            "diff"    : [att_ab*100, att_days, att_ret*100],
            "kezelt"  : [
                treated_df["segment"].eq("Agent Builder").mean()*100,
                treated_df["days_active"].mean(),
                treated_df["had_second_session"].mean()*100,
            ],
            "kontroll": [
                control_df["segment"].eq("Agent Builder").mean()*100,
                control_df["days_active"].mean(),
                control_df["had_second_session"].mean()*100,
            ],
        })

        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Bar(
            x=att_results["metrika"], y=att_results["kezelt"],
            name="Agent user (kezelt)", marker_color="#00b4d8",
            text=att_results["kezelt"].round(1),
            texttemplate="%{text}",
            textposition="outside",
        ))
        fig_c2.add_trace(go.Bar(
            x=att_results["metrika"], y=att_results["kontroll"],
            name="Matched kontroll", marker_color="#555566",
            text=att_results["kontroll"].round(1),
            texttemplate="%{text}",
            textposition="outside",
        ))
        fig_c2.update_layout(
            barmode="group",
            title="Propensity Score Matching Eredmenye -- ATT<br>"
                  "<sup>Azonos hatterrel rendelkezo agent vs nem-agent userek osszehasonlitasa</sup>",
            template="plotly_dark",
            height=420,
        )
        write_html(fig_c2, f"{OUTPUT_DIR}/12_att_results.html")
        print(f"  Saved: {OUTPUT_DIR}/12_att_results.html")

    else:
        print("  Tul keves matched par -- caliper novelese szukseges")

except ImportError:
    print("  sklearn not found -- propensity score athagyva")

print("\n[OK] Credit burn rate + Error assist + Propensity score complete.")
