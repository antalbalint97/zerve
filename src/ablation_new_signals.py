import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analytics.io import OUTPUT_DIR, ensure_output_dir, load_features
from analytics.viz import write_html

warnings.filterwarnings("ignore")

FEAT_PATH = "outputs/user_features_segmented.parquet"


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> dict:
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    ap = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    return {
        "model": name,
        "auc_mean": auc.mean(),
        "auc_std": auc.std(),
        "ap_mean": ap.mean(),
        "ap_std": ap.std(),
    }


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Loading data...")
    feat = load_features(FEAT_PATH)
    feat["is_agent_builder"] = (feat["segment"] == "Agent Builder").astype(int)
    agent_users = feat[feat["ever_used_agent"] == 1].copy()
    y = (agent_users["segment"] == "Agent Builder").astype(int)

    baseline_features = [
        "ttf_manual_run_min",
        "ttf_agent_tool_min",
        "ttf_agent_chat_min",
        "adopted_agent_early",
        "signed_up",
        "skipped_onboarding_form",
        "submitted_onboarding",
        "completed_onboarding",
        "first_session_events",
        "first_session_event_types",
        "first_session_duration_min",
        "first_session_had_agent",
        "first_session_had_run",
        "had_second_session",
        "time_to_return_hours",
        "signup_hour",
        "signup_is_weekend",
    ]
    baseline_features += [col for col in agent_users.columns if col.startswith("ref_")]
    baseline_features = [col for col in baseline_features if col in agent_users.columns]

    feature_families = {
        "iteration_quality": [
            "create_run_transitions",
            "run_refactor_transitions",
            "finish_summary_transitions",
            "agent_tool_transitions_total",
            "create_run_alternation_rate",
            "refactor_after_run_rate",
            "finish_summary_rate",
        ],
        "session_structure": [
            "productive_sessions",
            "total_sessions_derived",
            "max_events_per_session",
            "productive_session_share",
        ],
        "canvas_commitment": [
            "primary_canvas_event_share",
            "repeat_canvas_count_raw",
            "one_day_canvas_share",
        ],
    }
    feature_families = {
        family: [col for col in cols if col in agent_users.columns]
        for family, cols in feature_families.items()
    }

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"  Agent-user modeling population: {len(agent_users):,}")
    print(f"  Baseline feature count: {len(baseline_features)}")

    rows = []
    for model_name, model in models.items():
        X_base = agent_users[baseline_features].fillna(0)
        result = evaluate_model(model_name, model, X_base, y, cv)
        result["feature_set"] = "baseline"
        rows.append(result)

        for family, columns in feature_families.items():
            if not columns:
                continue
            X = agent_users[baseline_features + columns].fillna(0)
            result = evaluate_model(model_name, model, X, y, cv)
            result["feature_set"] = f"baseline + {family}"
            rows.append(result)

        all_new = sorted({col for cols in feature_families.values() for col in cols})
        X_all = agent_users[baseline_features + all_new].fillna(0)
        result = evaluate_model(model_name, model, X_all, y, cv)
        result["feature_set"] = "baseline + all_new_signals"
        rows.append(result)

    results = pd.DataFrame(rows)
    baseline_auc = results[results["feature_set"] == "baseline"][["model", "auc_mean", "ap_mean"]].rename(
        columns={"auc_mean": "baseline_auc_mean", "ap_mean": "baseline_ap_mean"}
    )
    results = results.merge(baseline_auc, on="model", how="left")
    results["auc_delta_vs_baseline"] = results["auc_mean"] - results["baseline_auc_mean"]
    results["ap_delta_vs_baseline"] = results["ap_mean"] - results["baseline_ap_mean"]
    results.to_csv(f"{OUTPUT_DIR}/ablation_new_signals.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR}/ablation_new_signals.csv")
    print("\nAblation summary:")
    print(
        results[["model", "feature_set", "auc_mean", "auc_delta_vs_baseline", "ap_mean", "ap_delta_vs_baseline"]]
        .round(4)
        .to_string(index=False)
    )

    fig = px.bar(
        results,
        x="feature_set",
        y="auc_delta_vs_baseline",
        color="model",
        barmode="group",
        title="Ablation of New Raw-Data Signals<br><sup>Delta vs narrowed baseline Agent Builder model</sup>",
        template="plotly_dark",
        labels={"feature_set": "Feature set", "auc_delta_vs_baseline": "AUC delta vs baseline", "model": ""},
        text=results["auc_delta_vs_baseline"].round(4),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=20, height=420)
    write_html(fig, f"{OUTPUT_DIR}/ablation_new_signals.html")
    print(f"  Saved: {OUTPUT_DIR}/ablation_new_signals.html")

    print("\n[OK] New-signal ablation complete.")


if __name__ == "__main__":
    main()
