# What Drives Successful Zerve Usage?
### Zerve × HackerEarth Hackathon 2026 — Written Summary

---

## 1. The Question

> **Which user behaviours and workflows are most predictive of long-term success on the Zerve platform?**

To answer this, I analysed Zerve's application event log — a rich dataset of user interactions including canvas creation, code execution, AI agent use, and onboarding steps.

---

## 2. Defining "Success"

Success on a computational notebook platform is multi-dimensional. Rather than using a single proxy (e.g. "upgraded account"), I defined a **Composite Engagement Score (CES)** across five criteria:

| # | Criterion | Rationale |
|---|-----------|-----------|
| C1 | **Depth** — ran code AND created an agent block | Combines two core value propositions |
| C2 | **Retention** — active on ≥ 3 distinct days | True habitual use, not a one-time visit |
| C3 | **Complexity** — used `requirements.txt` OR previewed output | Advanced features signal serious workflows |
| C4 | **AI Adoption** — accepted ≥ 1 agent suggestion | Engagement with Zerve's differentiator |
| C5 | **Reproducibility** — ≥ 3 total code runs | The canvas is being used as a reusable pipeline |

**A user is "successful" if they meet ≥ 2 of these 5 criteria.**

This threshold captures users who have genuinely discovered Zerve's value beyond a quick exploration, while being achievable enough to be meaningful for product decisions.

---

## 3. Methodology

### Data Preparation
- Parsed ~300 MB of PostHog event logs
- Normalised dual identity fields (`distinct_id` = browser, `person_id` = merged identity)
- Separated frontend events (`prop_$lib = web`) from backend SDK events (`posthog-python`)
- Engineered **35+ user-level features** aggregated from raw events

### Feature Engineering Categories
1. **Volume** — total events, unique sessions, days active
2. **Onboarding** — tour completion, form skipping, quickstart exploration
3. **Code Execution** — run counts, `requirements_build` usage
4. **Canvas Workflow** — block/layer creation, canvas edits
5. **AI Agent** — chats started, suggestions accepted, blocks created/run
6. **Output** — fullscreen preview, deployment interactions
7. **Time-to-Value** — minutes from sign-up to first code run
8. **SDK Usage** — backend Python API vs web-only usage

### Modelling
Three models were trained and evaluated via 5-fold stratified cross-validation:

| Model | ROC-AUC | Avg Precision |
|-------|---------|---------------|
| **Random Forest** | highest | highest |
| Gradient Boosting | comparable | comparable |
| Logistic Regression | interpretable baseline | — |

Random Forest was selected as the primary model for its superior feature importance interpretability and robustness to skewed distributions.

### User Clustering
K-Means (k=3) on normalised behavioural features revealed three natural user archetypes, validated against CES success rates.

---

## 4. Key Findings

### 🔑 Finding 1: The 15-Minute Rule
Users who execute their **first code block within 15 minutes** of signing up are substantially more likely to become successful long-term users. Success rates drop progressively as time-to-first-run increases — users who never run code in their first session rarely return.

**Implication:** The critical product intervention is reducing friction to that first execution. The onboarding tour should end with a *live code run*, not a UI walkthrough.

---

### 🔑 Finding 2: The Agent Flywheel
**AI agent interaction is the single strongest predictor of success.** But it's not merely correlated — it appears to be a flywheel: users who engage with the agent run *more* code, build *more complex* workflows, and return on *more days*. Each agent interaction reinforces deeper platform adoption.

Specifically:
- `agent_suggestions_accepted` is the top feature by importance
- `agent_blocks_created` drives `total_code_runs` (positive correlation)
- Users with 5+ agent events have dramatically higher success rates than zero-agent users

**Implication:** Surface the AI agent more prominently in the UI — it should be the default starting point for new users, not a discoverable side feature.

---

### 🔑 Finding 3: Three User Archetypes

| Archetype | Profile | Success Rate |
|-----------|---------|-------------|
| 🚀 **Power Users** | Agent + code + multi-day active | High |
| 🔍 **Explorers** | Active but haven't unlocked agent/advanced features | Medium |
| 👻 **Ghost Users** | Signed up, barely returned | Low |

Explorers are the highest-leverage segment: they're already engaged but one "aha moment" (typically accepting their first agent suggestion) away from becoming Power Users.

**Implication:** Targeted in-app prompts for Explorers at day 2–3 ("Try asking the agent to write this for you") could convert a significant share to Power Users.

---

### 🔑 Finding 4: Onboarding Completion ≠ Success
Completing the onboarding tour has a *smaller effect* on long-term success than expected. The first real code execution and first agent interaction are stronger predictors. Skipping the onboarding form is not necessarily a bad signal — determined users who skip directly to building often succeed.

**Implication:** Optimise onboarding for speed-to-execution, not completion-of-steps.

---

## 5. Zerve Deployment

The analysis is packaged as a **live Zerve API deployment**:

- **Endpoint**: `POST /predict` — given a user's behavioural metrics, returns:
  - Success probability (0–1)
  - Archetype classification (Power User / Explorer / Ghost)
  - CES score estimate
  - Personalised recommendations
- **Endpoint**: `GET /insights` — returns the key findings as structured JSON
- **Endpoint**: `POST /predict/batch` — batch scoring up to 1,000 users

This enables Zerve to integrate real-time user scoring into their product analytics pipeline.

---

## 6. Reproducibility

All analysis was performed within Zerve using a multi-block canvas:
1. `01_eda.py` — Exploratory data analysis
2. `02_feature_engineering.py` — User-level feature matrix
3. `03_success_definition.py` — CES labelling
4. `04_modeling.py` — ML training and evaluation
5. `05_visualization.py` — Interactive Plotly charts
6. `06_deployment_api.py` — FastAPI deployment

The `requirements.txt` ensures full environment reproducibility via Zerve's reusable environments.

---

*Built with Python 3.11 · pandas · scikit-learn · plotly · lifelines · FastAPI · Zerve*
