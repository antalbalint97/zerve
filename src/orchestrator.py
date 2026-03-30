"""
Legacy compatibility wrapper target for the standalone `orchestrator.py`.
Keep this file working so older references do not break.
"""

import subprocess
import sys
import json
import time
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent
ROOT       = SRC_DIR.parent
OUTPUT_DIR = ROOT / "outputs"
STATUS_FILE = OUTPUT_DIR / "pipeline_status.json"
LOG_FILE    = OUTPUT_DIR / "pipeline_log.txt"
OUTPUT_DIR.mkdir(exist_ok=True)

STEPS = [
    {
        "id"         : 1,
        "name"       : "Exploratory Data Analysis",
        "script"     : "01_eda.py",
        "emoji"      : "[01]",
        "description": "Load data, event frequencies, funnel, geo charts",
        "outputs"    : [
            "outputs/01_event_frequency.html",
            "outputs/01_activity_timeline.html",
            "outputs/01_user_funnel.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 2,
        "name"       : "Feature Engineering",
        "script"     : "02_feature_engineering.py",
        "emoji"      : "[02]",
        "description": "User-level features: agent tools, manual runs, canvas, onboarding, TTF",
        "outputs"    : [
            "outputs/user_features.parquet",
            "outputs/user_features.csv",
        ],
        "timeout_sec": 600,
    },
    {
        "id"         : 3,
        "name"       : "User Segmentation",
        "script"     : "03_user_segments.py",
        "emoji"      : "[03]",
        "description": "4 viselkedési szegmens: Agent Builder, Agent Runner, Manual Coder, Viewer/Ghost",
        "outputs"    : [
            "outputs/user_features_segmented.parquet",
            "outputs/03_segment_sizes.html",
            "outputs/03_segment_radar.html",
            "outputs/03_cohort_segment_heatmap.html",
        ],
        "timeout_sec": 120,
    },
    {
        "id"         : 4,
        "name"       : "Cohort Analysis",
        "script"     : "04_cohort_analysis.py",
        "emoji"      : "[04]",
        "description": "Signup hónap kohorsz + agent adoptáció kohorsz (külön elemzés)",
        "outputs"    : [
            "outputs/04_cohort_signup_overview.html",
            "outputs/04_cohort_segment_mix.html",
            "outputs/04_adoption_cohort_radar.html",
        ],
        "timeout_sec": 120,
    },
    {
        "id"         : 5,
        "name"       : "Lifecycle & Conversion",
        "script"     : "05_lifecycle_analysis.py",
        "emoji"      : "[05]",
        "description": "Életút elemzés, konverzió funnel, drop-off analízis",
        "outputs"    : [
            "outputs/05_global_funnel.html",
            "outputs/05_dropoff_analysis.html",
            "outputs/05_agent_adoption_timing.html",
        ],
        "timeout_sec": 180,
    },
    {
        "id"         : 6,
        "name"       : "KPI & Modelling",
        "script"     : "06_kpi_and_modeling.py",
        "emoji"      : "[06]",
        "description": "KPI matrix, B1 full + B2 narrowed early-signal model",
        "outputs"    : [
            "outputs/model_rf_full.joblib",
            "outputs/06_kpi_heatmap.html",
            "outputs/06_feature_importance_narrow.html",
            "outputs/06_roc_curves.html",
        ],
        "timeout_sec": 600,
    },
    {
        "id"         : 7,
        "name"       : "Signup Hour & Survival Analysis",
        "script"     : "07_signup_hour_survival.py",
        "emoji"      : "[07]",
        "description": "Signup-hour effects + Kaplan-Meier survival by segment/return/adoption",
        "outputs"    : [
            "outputs/07_signup_hour_ab_pct.html",
            "outputs/07_survival_by_segment.html",
            "outputs/07_survival_by_return.html",
            "outputs/07_return_threshold.html",
        ],
        "timeout_sec": 180,
    },
    {
        "id"         : 8,
        "name"       : "India Hypothesis & Success Definition",
        "script"     : "08_india_hypothesis_success_def.py",
        "emoji"      : "[08]",
        "description": "India hypothesis validation + Agent Builder definition argumentation",
        "outputs"    : [
            "outputs/08_country_hour_heatmap.html",
            "outputs/08_country_ab_comparison.html",
            "outputs/08_success_def_comparison.html",
        ],
        "timeout_sec": 180,
    },
    {
        "id"         : 9,
        "name"       : "Fleet Cohort Model",
        "script"     : "09_fleet_cohort_model.py",
        "emoji"      : "[09]",
        "description": "Parallel RF model by cohort -- Zerve Fleet pattern",
        "outputs"    : [
            "outputs/09_fleet_auc_by_cohort.html",
            "outputs/09_fleet_top_features.html",
            "outputs/09_fleet_feature_drift.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 10,
        "name"       : "User Lifecycle & Activation",
        "script"     : "10_user_lifecycle.py",
        "emoji"      : "[10]",
        "description": "Time-to-AB, activation milestones, user lifecycle paths",
        "outputs"    : [
            "outputs/10_time_to_agent_builder.html",
            "outputs/10_activation_milestones.html",
            "outputs/10_activation_lift.html",
            "outputs/10_path_divergence.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 11,
        "name"       : "Tool Sequences & Session Progression",
        "script"     : "11_tool_sequences_session_progression.py",
        "emoji"      : "[11]",
        "description": "Tool bigram/trigram workflow fingerprints, session complexity growth",
        "outputs"    : [
            "outputs/11_tool_transition_matrix.html",
            "outputs/11_session_progression_diversity.html",
            "outputs/11_workflow_complexity_progression.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 12,
        "name"       : "Credit Burn Rate + Error Assist + Propensity Score",
        "script"     : "12_credit_error_propensity.py",
        "emoji"      : "[12]",
        "description": "Credit intensity, error-assist signal, causal-effect estimation with PSM",
        "outputs"    : [
            "outputs/12_credit_burn_rate.html",
            "outputs/12_error_assist_signal.html",
            "outputs/12_propensity_matching.html",
            "outputs/12_att_results.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 13,
        "name"       : "Canvas Complexity Growth",
        "script"     : "13_canvas_complexity.py",
        "emoji"      : "[CC]",
        "description": "Canvas-level complexity growth and repeat-canvas retention signals",
        "outputs"    : [
            "outputs/13_canvas_complexity_distribution.html",
            "outputs/13_canvas_growth_by_segment.html",
            "outputs/13_repeat_canvas_retention.html",
            "outputs/canvas_complexity_features.parquet",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 14,
        "name"       : "Active User Churn Prediction",
        "script"     : "14_churn_prediction.py",
        "emoji"      : "[CH]",
        "description": "14-day churn model for active users with complexity-enriched features",
        "outputs"    : [
            "outputs/14_churn_roc_pr.html",
            "outputs/14_churn_feature_importance.html",
            "outputs/14_churn_risk_buckets.html",
            "outputs/14_churn_scored_users.parquet",
            "outputs/model_churn_rf.joblib",
        ],
        "timeout_sec": 420,
    },
    {
        "id"         : 15,
        "name"       : "N-gram Workflow Analysis",
        "script"     : "15_ngram_workflow_analysis.py",
        "emoji"      : "[NG]",
        "description": "Lifecycle and session motif analysis across segments, churn, and complexity",
        "outputs"    : [
            "outputs/15_event_ngram_lift.html",
            "outputs/15_segment_workflow_motifs.html",
            "outputs/15_churn_vs_retained_motifs.html",
            "outputs/15_ngram_tables.csv",
        ],
        "timeout_sec": 360,
    },
    {
        "id"         : 16,
        "name"       : "Geo Location Analysis",
        "script"     : "16_geo_location_analysis.py",
        "emoji"      : "[GEO]",
        "description": "Country and region level onboarding, churn, and complexity comparisons",
        "outputs"    : [
            "outputs/16_country_segment_mix.html",
            "outputs/16_geo_onboarding_effectiveness.html",
            "outputs/16_geo_churn_complexity_heatmap.html",
            "outputs/16_country_metrics.csv",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 17,
        "name"       : "Intervention Scoring",
        "script"     : "17_intervention_scoring.py",
        "emoji"      : "[INT]",
        "description": "Ranks users for activation nudge, struggle support, builder acceleration, and retention rescue",
        "outputs"    : [
            "outputs/17_intervention_scored_users.parquet",
            "outputs/17_intervention_summary.csv",
            "outputs/17_intervention_mix.html",
            "outputs/17_intervention_signal_profile.html",
            "outputs/17_top_intervention_candidates.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 18,
        "name"       : "Quality Of Struggle",
        "script"     : "18_quality_of_struggle.py",
        "emoji"      : "[STR]",
        "description": "Separates productive struggle from abandonment-prone friction after errors and credit pressure",
        "outputs"    : [
            "outputs/18_quality_of_struggle_scored_users.parquet",
            "outputs/18_quality_of_struggle_summary.csv",
            "outputs/18_quality_of_struggle_mix.html",
            "outputs/18_quality_of_struggle_signals.html",
            "outputs/18_recovery_vs_abandonment.html",
        ],
        "timeout_sec": 300,
    },
    {
        "id"         : 19,
        "name"       : "Path Branching Model",
        "script"     : "19_path_branching_model.py",
        "emoji"      : "[PTH]",
        "description": "Quantifies where Ghost, Viewer, and Builder paths diverge in the first few steps",
        "outputs"    : [
            "outputs/19_branch_step_divergence.csv",
            "outputs/19_branching_summary.csv",
            "outputs/19_path_step_mix.html",
            "outputs/19_top_branch_points.html",
            "outputs/19_segment_path_prefixes.html",
        ],
        "timeout_sec": 300,
    },
]

STEP_BY_ID = {step["id"]: step for step in STEPS}


def resolve_steps(selected_ids=None, from_step=1):
    if selected_ids:
        return [STEP_BY_ID[step_id] for step_id in selected_ids if step_id in STEP_BY_ID]
    return [step for step in STEPS if step["id"] >= from_step]


def build_status_steps(run_ids):
    status_steps = init_status(STEPS)["steps"]
    for step in status_steps:
        if step["id"] not in run_ids:
            step["status"] = "skipped"
    return status_steps


# ── STATUS HELPERS ────────────────────────────────────────

def init_status(steps):
    status = {
        "pipeline_id"   : datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "started_at"    : datetime.now(timezone.utc).isoformat(),
        "finished_at"   : None,
        "overall_status": "running",   # running | success | failed | partial
        "steps"         : [],
        "summary"       : {
            "total": len(steps),
            "done" : 0,
            "failed": 0,
            "skipped": 0,
        },
    }
    for s in steps:
        status["steps"].append({
            "id"          : s["id"],
            "name"        : s["name"],
            "emoji"       : s["emoji"],
            "description" : s["description"],
            "script"      : s["script"],
            "status"      : "pending",   # pending | running | success | failed | skipped
            "started_at"  : None,
            "finished_at" : None,
            "duration_sec": None,
            "exit_code"   : None,
            "output_tail" : "",          # last N lines of stdout
            "error_tail"  : "",
            "outputs_found": [],
        })
    return status


def write_status(status):
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def tail(text, n=20):
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:]) if lines else ""


# ── RUNNER ────────────────────────────────────────────────

def run_step(step_cfg, step_status, log_fh, dry_run=False):
    script      = step_cfg["script"]
    script_path = SRC_DIR / script

    if dry_run:
        print(f"  [DRY-RUN] Would run: python src/{script}")
        step_status["status"]      = "skipped"
        step_status["output_tail"] = "dry-run mode"
        return True

    # Import-only check (for API step)
    if step_cfg.get("import_only"):
        cmd = [sys.executable, "-c",
               f"import importlib.util, sys; "
               f"spec=importlib.util.spec_from_file_location('m','{script}'); "
               f"# import check only"]
        # Just try to compile the file
        try:
            with open(script_path) as f:
                source = f.read()
            compile(source, str(script_path), "exec")
            step_status["status"]      = "success"
            step_status["output_tail"] = "Syntax OK - import check passed"
            return True
        except SyntaxError as e:
            step_status["status"]     = "failed"
            step_status["error_tail"] = str(e)
            return False

    cmd = [sys.executable, str(script_path)]
    t0  = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=step_cfg["timeout_sec"],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Write to log
        log_fh.write(f"\n{'='*60}\n")
        log_fh.write(f"STEP {step_cfg['id']}: {script}\n")
        log_fh.write(f"Exit code: {proc.returncode} | Duration: {elapsed:.1f}s\n")
        log_fh.write("STDOUT:\n" + stdout)
        if stderr:
            log_fh.write("STDERR:\n" + stderr)
        log_fh.flush()

        step_status["exit_code"]    = proc.returncode
        step_status["duration_sec"] = round(elapsed, 2)
        step_status["output_tail"]  = tail(stdout, 25)
        step_status["error_tail"]   = tail(stderr, 10) if proc.returncode != 0 else ""

        # Check which output files exist
        step_status["outputs_found"] = [
            f for f in step_cfg.get("outputs", [])
            if (ROOT / f).exists()
        ]

        if proc.returncode == 0:
            step_status["status"] = "success"
            return True
        else:
            step_status["status"] = "failed"
            return False

    except subprocess.TimeoutExpired:
        step_status["status"]     = "failed"
        step_status["error_tail"] = f"TIMEOUT after {step_cfg['timeout_sec']}s"
        return False

    except FileNotFoundError:
        step_status["status"]     = "failed"
        step_status["error_tail"] = f"Script not found: {script_path}"
        return False


# ── MAIN ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zerve Hackathon Pipeline Orchestrator")
    parser.add_argument("--steps",     nargs="+", type=int, help="Run specific step IDs")
    parser.add_argument("--from-step", type=int, default=1, help="Start from step N")
    parser.add_argument("--dry-run",   action="store_true", help="Validate without running")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Abort pipeline on first failure (default: continue)")
    args = parser.parse_args()

    steps_to_run = resolve_steps(selected_ids=args.steps, from_step=args.from_step)

    print(f"\n{'='*60}")
    print(f"  ZERVE HACKATHON PIPELINE ORCHESTRATOR")
    print(f"  Steps: {[s['id'] for s in steps_to_run]}")
    print(f"  Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    status = init_status(steps_to_run)

    run_ids = {s["id"] for s in steps_to_run}
    status["steps"] = build_status_steps(run_ids)
    write_status(status)

    with open(LOG_FILE, "w") as log_fh:
        log_fh.write(f"Pipeline started: {status['started_at']}\n")

        for step_cfg in steps_to_run:
            step_idx = next(i for i, s in enumerate(status["steps"])
                            if s["id"] == step_cfg["id"])
            step_status = status["steps"][step_idx]

            print(f"{step_cfg['emoji']}  Step {step_cfg['id']}: {step_cfg['name']}")
            print(f"   {step_cfg['description']}")
            print(f"   Script: {step_cfg['script']}")

            step_status["status"]     = "running"
            step_status["started_at"] = datetime.now(timezone.utc).isoformat()
            write_status(status)

            success = run_step(step_cfg, step_status, log_fh, dry_run=args.dry_run)

            step_status["finished_at"] = datetime.now(timezone.utc).isoformat()

            if success:
                duration = step_status.get('duration_sec') or 0
                print(f"   [OK] Done in {duration:.1f}s")
                print(f"   Outputs found: {len(step_status['outputs_found'])}")
                status["summary"]["done"] += 1
            else:
                print(f"   [FAIL] FAILED")
                print(f"   {step_status['error_tail'][:200]}")
                status["summary"]["failed"] += 1
                write_status(status)
                if args.stop_on_failure:
                    print("\n[STOP] Stopping pipeline (--stop-on-failure)")
                    break

            write_status(status)
            print()

    # Finalize
    status["finished_at"]    = datetime.now(timezone.utc).isoformat()
    status["overall_status"] = (
        "success" if status["summary"]["failed"] == 0
        else "partial" if status["summary"]["done"] > 0
        else "failed"
    )
    write_status(status)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {status['overall_status'].upper()}")
    print(f"  Done: {status['summary']['done']}  |  Failed: {status['summary']['failed']}")
    print(f"  Status file: {STATUS_FILE}")
    print(f"  Log file:    {LOG_FILE}")
    print(f"{'='*60}\n")

    return 0 if status["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
