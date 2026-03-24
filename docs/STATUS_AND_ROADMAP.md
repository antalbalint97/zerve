# Status And Roadmap

## Where We Are

The project is now past the first working-pipeline stage and into the second stage: stabilization, correction, and extension.

### Completed
- full 1-17 analytical pipeline exists and runs
- late-phase roadmap items from the original technical document were implemented:
  - canvas complexity
  - active-user churn proxy modeling
  - n-gram workflow analysis
  - geo analysis
- main narrative and appendix were written:
  - `docs/CASE_NARRATIVE.md`
  - `docs/TECHNICAL_APPENDIX.md`
- the helper layer now exists and is used by a large portion of the pipeline:
  - `analytics/io.py`
  - `analytics/events.py`
  - `analytics/metrics.py`
  - `analytics/viz.py`
- the most important known analytical weak points were corrected:
  - step 10 third-build bug
  - step 14 growth interpretation
  - step 15 churn-proxy naming
  - step 16 smoothed n-gram lift
  - step 17 safer geo interpretation
- new raw-data signals were added into feature engineering:
  - iteration quality
  - session structure
  - canvas commitment
- the repo now contains a competition-oriented reproduction guide:
  - `docs/ZERVE_REPRODUCTION_GUIDE.md`

### Recently Completed
- `02_feature_engineering.py` was refactored into a function-based structure
- `06_kpi_and_modeling.py` and `12_credit_error_propensity.py` were migrated onto shared helpers
- `13_orchestrator.py` step resolution was cleaned up
- `10_user_lifecycle.py` now uses shared output handling
- first ablation script for the new signals was added:
  - `ablation_new_signals.py`
- first ablation run completed:
  - iteration-quality signals look extremely strong
  - this is promising, but should be treated as leakage-sensitive until reviewed carefully
- leakage review was documented:
  - `docs/LEAKAGE_REVIEW.md`
- phase 18 was implemented:
  - `18_intervention_scoring.py`
  - outputs now exist under `outputs/18_*`
- phase 19 was implemented:
  - `19_quality_of_struggle.py`
  - outputs now exist under `outputs/19_*`
- phase 20 was implemented:
  - `20_path_branching_model.py`
  - outputs now exist under `outputs/20_*`

## What Still Needs To Be Done

### 1. Architectural cleanup
- optional deeper helper-style cleanup of `10_user_lifecycle.py`
- optionally standardize remaining chart output and plotting conventions everywhere
- consider extracting reusable modeling helpers if repeated cross-validation logic keeps growing

### 2. Validate the new signals
- decide which new features are:
  - useful for modeling
  - useful for narrative only
  - too weak or noisy to keep emphasizing
- if useful, fold the strongest new signals into downstream modeling and reporting
- treat the iteration-quality family as reviewed but leakage-sensitive
- only promote iteration-quality features into predictive models after stronger time-bounding or target separation

### 3. Extend the analysis
The next agreed product-facing phases are now:

- `18_intervention_scoring.py`
  - implemented
  - identifies onboarding nudge, productive struggle support, builder acceleration, and retention rescue candidates
- `19_quality_of_struggle.py`
  - implemented
  - separates productive friction from abandonment-prone friction
- `20_path_branching_model.py`
  - implemented
  - quantifies where Ghost, Viewer, and Builder paths diverge early

### 4. Reproducibility on Zerve
- translate the local script pipeline into the actual Zerve canvas structure described in `docs/ZERVE_REPRODUCTION_GUIDE.md`
- build the submission-safe version first
- then add:
  - Fleet block
  - interpretation block
  - hosted app/dashboard block

### 5. Publication readiness
- add a compact publish-ready summary artifact
- align wording in all plots and docs with the final target definitions
- keep descriptive, predictive, and quasi-causal claims explicitly separated

## What Is Planned After The Current Refactor

### Immediate next move
1. inspect the `20_*` path-branching outputs and identify the cleanest early branch points for the product story
2. start translating the strongest pieces into the Zerve competition canvas
3. decide which reduced, submission-safe flow to prioritize first

### After that
1. start building the actual Zerve competition canvas from the reproduction guide
2. add a compact publishable summary artifact layer

## Strategic Direction

The repo is moving along three tracks at once:

- make the pipeline more modular and maintainable
- improve the quality and trustworthiness of the findings
- convert the project into a competition-ready, reproducible Zerve submission

That means the goal is no longer just "more charts". The goal is:

- a stable pipeline
- stronger product signals
- clearer intervention logic
- a reproducible Zerve-native project story
