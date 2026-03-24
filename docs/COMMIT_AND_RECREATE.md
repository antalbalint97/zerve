# Commit And Recreate

## Goal

Prepare the repo for a clean commit and a faithful rebuild on the Zerve platform under the hackathon rules.

## What We Have Now

The analysis is complete through phase 20.

The repo now includes:

- a standalone pipeline entrypoint: `orchestrator.py`
- numbered analysis scripts through `20_path_branching_model.py`
- shared helper modules under `analytics/`
- narrative and technical writeups under `docs/`
- intervention and struggle extensions
- a path branching model
- Zerve reproduction documentation

## Current Analytical Conclusion

The strongest current conclusion is:

1. most users are Ghosts or shallow explorers
2. the first 48 hours determine whether users become builders
3. fast return, build-run loops, and repeat-canvas depth are the strongest positive product signals
4. some struggle signals are healthy and engaged, but some are clearly abandonment-prone
5. the project now supports intervention logic, not just descriptive analysis

## What To Commit

Commit:

- all numbered phase scripts
- `orchestrator.py`
- helper modules
- docs
- tests
- optionally the generated output artifacts if you want a frozen reference snapshot

At minimum, the code + docs to preserve are:

- `orchestrator.py`
- `01_eda.py` through `20_path_branching_model.py`
- `analytics/io.py`
- `analytics/events.py`
- `analytics/metrics.py`
- `analytics/viz.py`
- `requirements.txt`
- `docs/CASE_NARRATIVE.md`
- `docs/TECHNICAL_APPENDIX.md`
- `docs/STATUS_AND_ROADMAP.md`
- `docs/NEXT_STEPS_IMPLEMENTATION.md`
- `docs/ZERVE_REPRODUCTION_GUIDE.md`
- `docs/LEAKAGE_REVIEW.md`

## What To Recreate On Zerve

Use `docs/ZERVE_REPRODUCTION_GUIDE.md` as the source of truth.

Recommended order:

1. build the submission-safe block set first
2. verify the core artifacts regenerate correctly
3. then add the extra presentation and Fleet layers

The key submission-safe outputs to verify are:

- user feature matrix
- segmented feature matrix
- canvas complexity features
- churn proxy scores
- geo metrics
- intervention summary
- quality-of-struggle summary
- branching summary

## Practical Pre-Commit Checklist

- confirm `python orchestrator.py --steps 18 19 20` still passes
- confirm `python -m unittest tests.test_analytics_helpers` passes
- confirm `docs/ZERVE_REPRODUCTION_GUIDE.md` reflects the current scripts and outputs
- confirm `README.md` reflects the current project, not the old 6-step layout
- confirm no doc still frames `13_orchestrator.py` as the primary entrypoint

## Zerve Submission Checklist

- use only `zerve_events.csv`
- use Zerve Python blocks for all analysis steps
- keep the final story reproducible end to end
- include at least one meaningful Fleet usage example
- clearly separate:
  - descriptive findings
  - predictive findings
  - quasi-causal findings
- clearly label:
  - success proxy = `Agent Builder`
  - churn target = `14-day survival-style churn proxy`
  - iteration-quality features = leakage-sensitive for prediction

## Recommended Final Focus

Do not keep expanding the local analysis indefinitely before the submission build.

The current best move is:

1. commit this stabilized state
2. rebuild the submission-safe version on Zerve
3. verify parity of the main outputs
4. then add polish layers only if time remains
