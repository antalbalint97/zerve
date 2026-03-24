# Leakage Review

## Purpose

Review the newly added raw-data signals before promoting them into downstream prediction or intervention logic.

This review focuses on the first ablation run from:

- `outputs/ablation_new_signals.csv`

## Key Result

The `iteration_quality` feature family produced an extremely large jump in the narrowed Agent Builder model.

Examples from the ablation run:

- Random Forest baseline AUC: `0.9077`
- Random Forest baseline + iteration_quality AUC: `0.9999`
- Logistic Regression baseline AUC: `0.8896`
- Logistic Regression baseline + iteration_quality AUC: `0.9988`

This is too strong to accept at face value without a leakage-sensitive review.

## Why This Is Suspicious

The strongest iteration-quality features are tightly tied to the operational success definition itself.

Examples:

- `create_run_transitions`
- `run_refactor_transitions`
- `finish_summary_transitions`
- `create_run_alternation_rate`
- `refactor_after_run_rate`
- `finish_summary_rate`

The current success proxy is `Agent Builder`, which is itself defined through heavy agent-building behavior. That means many of these new signals are not just "early product hints"; they may be near-direct behavioral expressions of the target.

## Evidence From Quick Correlation Checks

The following patterns were observed in the review pass:

- `create_run_alternation_rate` had very high correlation with the Agent Builder target
- transition-count features were also strongly correlated with `agent_build_calls`
- `agent_tool_transitions_total` was almost perfectly correlated with `agent_tool_calls_total`

Interpretation:

- some iteration-quality signals are likely valid product descriptors
- but several are too entangled with the existing success definition to be treated as independent early predictors

## Recommended Classification

### Safe enough for intervention and product diagnostics
- `productive_sessions`
- `productive_session_share`
- `repeat_canvas_count_raw`
- `primary_canvas_event_share`
- `one_day_canvas_share`
- credit/error struggle signals
- churn proxy risk signals

### Use with caution in predictive models
- `create_run_alternation_rate`
- `refactor_after_run_rate`
- `finish_summary_rate`

These may still be useful, but only if we explicitly position them as near-behavioral outcome markers rather than upstream early predictors.

### Avoid using as headline predictive evidence without stronger controls
- `create_run_transitions`
- `run_refactor_transitions`
- `finish_summary_transitions`
- `agent_tool_transitions_total`

These are too close to workflow volume and the operational success label to be trusted as clean early signals.

## Policy Going Forward

### For product interventions
Use the iteration-quality family as:

- descriptive workflow health indicators
- builder-loop quality signals
- support or onboarding routing inputs

### For prediction
Keep iteration-quality features out of the core narrowed Builder prediction baseline until one of the following is done:

1. time-bound them to a much earlier window
2. predict a later downstream outcome that is not mechanically close to them
3. formally separate target-definition features from candidate predictor families

## Decision

For phase 18 intervention scoring:

- use lower-leakage activation, repeat-canvas, struggle, and churn-proxy signals as the main backbone
- do not rely on the iteration-quality family as the main score driver

The iteration-quality features should stay in the repo, but as reviewed diagnostics rather than automatically trusted predictors.
