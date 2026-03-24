"""
=============================================================
 06_deployment_api.py  —  Zerve API Deployment
 Zerve Hackathon 2026: "What Drives Successful Usage?"
=============================================================
Deploy this as a Zerve App/API deployment.
Exposes a REST endpoint that scores user success probability
in real-time given a set of behavioural metrics.

Endpoints:
  GET  /health                   → service status
  POST /predict                  → score a single user
  POST /predict/batch            → score multiple users
  GET  /features                 → list expected features
  GET  /archetypes               → archetype definitions
  GET  /insights                 → key findings summary

Run locally:
  uvicorn 06_deployment_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import joblib
import os

# ── LOAD MODEL ───────────────────────────────────────────
MODEL_DIR   = "outputs"
MODEL_PATH  = os.path.join(MODEL_DIR, "model_rf.joblib")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_names.joblib")

try:
    model         = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEAT_PATH)
    MODEL_LOADED  = True
    print(f"✓ Model loaded ({len(feature_names)} features)")
except FileNotFoundError:
    MODEL_LOADED  = False
    model         = None
    feature_names = []
    print("⚠  Model not found — run 04_modeling.py first")

# ── APP ──────────────────────────────────────────────────
app = FastAPI(
    title="Zerve User Success Predictor",
    description=(
        "Predict whether a Zerve user will become a long-term successful user "
        "based on their early behavioural signals.\n\n"
        "**Success Definition**: Composite Engagement Score ≥ 2 out of 5 criteria\n"
        "- C1 Depth: ran code AND created agent blocks\n"
        "- C2 Retention: active ≥ 3 days\n"
        "- C3 Complexity: used requirements or previewed output\n"
        "- C4 AI Adoption: accepted ≥ 1 agent suggestion\n"
        "- C5 Reproducibility: ≥ 3 total code runs"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SCHEMAS ──────────────────────────────────────────────

class UserBehaviour(BaseModel):
    """Behavioural features for a single user."""
    # Volume
    total_events:          float = Field(0, ge=0, description="Total events fired")
    unique_event_types:    float = Field(0, ge=0, description="Distinct event types")
    unique_canvases:       float = Field(0, ge=0, description="Canvases visited")
    unique_sessions:       float = Field(0, ge=0, description="Distinct sessions")
    days_active:           float = Field(0, ge=0, description="Distinct active days")

    # Onboarding
    completed_onboarding_tour:  float = Field(0, ge=0, le=1)
    skipped_onboarding_form:    float = Field(0, ge=0, le=1)
    explored_quickstart:        float = Field(0, ge=0, le=1)
    onboarding_steps_completed: float = Field(0, ge=0)

    # Code execution
    run_block_count:      float = Field(0, ge=0)
    run_all_blocks_count: float = Field(0, ge=0)
    requirements_built:   float = Field(0, ge=0, le=1)
    total_code_runs:      float = Field(0, ge=0)

    # Canvas
    blocks_created:  float = Field(0, ge=0)
    layers_created:  float = Field(0, ge=0)
    canvas_edits:    float = Field(0, ge=0)
    canvas_opens:    float = Field(0, ge=0)

    # Agent / AI
    agent_chats_started:         float = Field(0, ge=0)
    agent_suggestions_accepted:  float = Field(0, ge=0)
    agent_blocks_created:        float = Field(0, ge=0)
    agent_blocks_run:            float = Field(0, ge=0)
    agent_workers_created:       float = Field(0, ge=0)
    agent_error_assist:          float = Field(0, ge=0)
    used_agent:                  float = Field(0, ge=0, le=1)
    agent_total_events:          float = Field(0, ge=0)

    # Output
    fullscreen_opens:  float = Field(0, ge=0)
    previewed_output:  float = Field(0, ge=0, le=1)

    # SDK / backend
    used_python_sdk: float = Field(0, ge=0, le=1)

    # Time-to-value
    time_to_first_run_min: float = Field(-1, description="-1 if never ran code")

    # Session depth
    avg_session_depth: float = Field(0, ge=0)
    max_session_depth: float = Field(0, ge=0)

    # Recency
    recency_days: float = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "total_events": 45,
                "days_active": 3,
                "total_code_runs": 8,
                "agent_total_events": 12,
                "agent_suggestions_accepted": 2,
                "time_to_first_run_min": 8.5,
                "completed_onboarding_tour": 1,
                "requirements_built": 1,
                "previewed_output": 1,
                "unique_canvases": 2,
                "avg_session_depth": 15.3,
            }
        }


class PredictionResponse(BaseModel):
    success_probability: float
    is_success_predicted: bool
    confidence: str
    archetype: str
    ces_score_estimate: int
    criteria_met: Dict[str, bool]
    recommendations: List[str]


class BatchRequest(BaseModel):
    users: List[UserBehaviour]
    user_ids: Optional[List[str]] = None


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    user_ids: Optional[List[str]] = None
    summary: Dict[str, Any]


# ── HELPERS ──────────────────────────────────────────────

def user_to_vector(u: UserBehaviour) -> np.ndarray:
    """Convert UserBehaviour to numpy array aligned with feature_names."""
    d = u.model_dump()
    vec = np.array([d.get(f, 0.0) for f in feature_names], dtype=float)
    return vec


def estimate_criteria(u: UserBehaviour) -> Dict[str, bool]:
    return {
        "C1_depth"          : u.total_code_runs >= 1 and u.agent_blocks_created >= 1,
        "C2_retention"      : u.days_active >= 3,
        "C3_complexity"     : u.requirements_built >= 1 or u.previewed_output >= 1,
        "C4_ai_adoption"    : u.agent_suggestions_accepted >= 1,
        "C5_reproducibility": u.total_code_runs >= 3,
    }


def assign_archetype(u: UserBehaviour, prob: float) -> str:
    if prob >= 0.65:
        return "🚀 Power User"
    elif prob >= 0.35:
        return "🔍 Explorer"
    else:
        if u.total_events <= 3:
            return "👻 Ghost User"
        return "🌱 Early Adopter"


def build_recommendations(criteria: Dict[str, bool], u: UserBehaviour) -> List[str]:
    recs = []
    if not criteria["C1_depth"]:
        if u.total_code_runs == 0:
            recs.append("▶  Run your first code block to unlock the full Zerve workflow.")
        else:
            recs.append("🤖  Try creating an AI agent block — power users combine code + agents.")
    if not criteria["C2_retention"]:
        recs.append("📅  Come back over multiple days; consistent use drives deeper mastery.")
    if not criteria["C3_complexity"]:
        recs.append("📦  Add a requirements.txt to your canvas or use the fullscreen preview.")
    if not criteria["C4_ai_adoption"]:
        recs.append("💡  Accept an AI suggestion — users who leverage the agent succeed more often.")
    if not criteria["C5_reproducibility"]:
        recs.append("🔄  Run your canvas multiple times to build reproducible pipelines.")
    if not recs:
        recs.append("🏆  You're on track! Explore advanced features like Fleet and self-hosting.")
    return recs[:3]  # top 3


def predict_one(u: UserBehaviour) -> PredictionResponse:
    if not MODEL_LOADED:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run 04_modeling.py first.")

    vec   = user_to_vector(u).reshape(1, -1)
    prob  = float(model.predict_proba(vec)[0, 1])
    pred  = prob >= 0.5
    crit  = estimate_criteria(u)
    ces   = sum(crit.values())

    if prob >= 0.75:
        confidence = "High"
    elif prob >= 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        success_probability  = round(prob, 4),
        is_success_predicted = pred,
        confidence           = confidence,
        archetype            = assign_archetype(u, prob),
        ces_score_estimate   = ces,
        criteria_met         = crit,
        recommendations      = build_recommendations(crit, u),
    )


# ── ROUTES ───────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status"       : "ok",
        "model_loaded" : MODEL_LOADED,
        "n_features"   : len(feature_names),
        "version"      : "1.0.0",
    }


@app.get("/features")
def get_features():
    return {
        "feature_count": len(feature_names),
        "features"      : feature_names,
    }


@app.get("/insights")
def get_insights():
    return {
        "title": "What Drives Successful Zerve Usage?",
        "key_findings": [
            {
                "insight": "The 15-Minute Rule",
                "description": (
                    "Users who run their first code block within 15 minutes of signing up "
                    "are significantly more likely to become successful long-term users. "
                    "The success rate drops sharply after 30 minutes."
                ),
                "actionable": "Reduce friction to first code execution in onboarding.",
            },
            {
                "insight": "The Agent Flywheel",
                "description": (
                    "AI agent interaction is the strongest predictor of success. "
                    "Users who engage with the agent run more code, create more complex "
                    "workflows, and return more often — a self-reinforcing flywheel."
                ),
                "actionable": "Surface the AI agent earlier and more prominently in the UI.",
            },
            {
                "insight": "Three User Archetypes",
                "description": (
                    "🚀 Power Users (agent + code + multi-day) — highest retention. "
                    "🔍 Explorers (active but not using advanced features) — growth opportunity. "
                    "👻 Ghost Users (signed up, rarely returned) — onboarding target."
                ),
                "actionable": "Tailor activation emails by archetype within first 48 hours.",
            },
            {
                "insight": "Onboarding Tour Completion Matters Less Than Expected",
                "description": (
                    "Completing the onboarding tour has a smaller effect on success than "
                    "actually running code or using the agent. The first real execution "
                    "is the true 'aha moment'."
                ),
                "actionable": "Redesign onboarding to end with a live code run, not a tour step.",
            },
        ],
        "success_definition": {
            "name": "Composite Engagement Score (CES)",
            "threshold": "≥ 2 out of 5 criteria",
            "criteria": {
                "C1": "Ran code AND created an agent block",
                "C2": "Active on ≥ 3 distinct days",
                "C3": "Used requirements.txt OR previewed fullscreen output",
                "C4": "Accepted ≥ 1 AI agent suggestion",
                "C5": "Total code runs ≥ 3",
            },
        },
    }


@app.get("/archetypes")
def get_archetypes():
    return {
        "archetypes": [
            {
                "name"           : "🚀 Power User",
                "probability_range": "≥ 0.65",
                "characteristics": [
                    "Uses AI agent extensively",
                    "Active across multiple days",
                    "Runs complex multi-block workflows",
                    "Builds and previews deployments",
                ],
            },
            {
                "name"           : "🔍 Explorer",
                "probability_range": "0.35 – 0.65",
                "characteristics": [
                    "Engages with core features",
                    "Starting to use agent capabilities",
                    "Growth opportunity with nudging",
                ],
            },
            {
                "name"           : "🌱 Early Adopter",
                "probability_range": "< 0.35 (with activity)",
                "characteristics": [
                    "Has some events but not deeply engaged",
                    "Hasn't discovered agent or complex features yet",
                ],
            },
            {
                "name"           : "👻 Ghost User",
                "probability_range": "< 0.35 (minimal activity)",
                "characteristics": [
                    "Signed up but barely returned",
                    "Likely did not reach 'aha moment'",
                    "Re-engagement email target",
                ],
            },
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(user: UserBehaviour):
    """Score a single user's likelihood of becoming a successful Zerve user."""
    return predict_one(user)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    """Score multiple users in a single call (max 1000)."""
    if len(req.users) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 users per batch.")

    predictions = [predict_one(u) for u in req.users]
    probs = [p.success_probability for p in predictions]

    return BatchResponse(
        predictions = predictions,
        user_ids    = req.user_ids,
        summary     = {
            "total_users"       : len(predictions),
            "predicted_success" : sum(p.is_success_predicted for p in predictions),
            "avg_probability"   : round(np.mean(probs), 4),
            "power_users"       : sum(1 for p in predictions if "Power" in p.archetype),
            "explorers"         : sum(1 for p in predictions if "Explorer" in p.archetype),
            "ghost_users"       : sum(1 for p in predictions if "Ghost" in p.archetype),
        },
    )


# ── ENTRY POINT ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
