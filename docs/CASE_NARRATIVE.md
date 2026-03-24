# Zerve Case Narrative

## The Question

Which user behaviors most reliably predict successful usage on Zerve, and what should the team do differently because of that?

This narrative is based on the latest successful full pipeline run recorded in [pipeline_log.txt](c:/Users/Balint/Documents/Pet_projects/zerve/outputs/pipeline_log.txt). The older technical document was useful as roadmap context, but the current pipeline outputs are treated as the source of truth.

## Product Context And Dataset Snapshot

The analysis covers Zerve product telemetry from **September 1, 2025 to December 8, 2025**.

- `409,287` raw events
- `408,919` cleaned events in feature engineering
- `141` event types
- `4,771` modeled users
- `79` engineered features in the current user feature matrix

The data combines web activity with backend agent-tool telemetry. That matters because some of the most valuable user behaviors, like agent-assisted block creation, refactoring, and run loops, happen outside the browser-only event stream.

The current rule-based segmentation remains the clearest high-level summary of Zerve's product reality:

| Segment | Users | Share | Avg days active | Avg agent tools | Avg build calls |
|---|---:|---:|---:|---:|---:|
| Agent Builder | 285 | 6.0% | 4.75 | 156.99 | 39.15 |
| Manual Coder | 161 | 3.4% | 5.59 | 125.20 | 0.02 |
| Viewer | 1,288 | 27.0% | 1.69 | 26.07 | 0.01 |
| Ghost | 3,037 | 63.7% | 1.05 | 0.00 | 0.00 |

The central product problem is immediately visible: the challenge is not fine-tuning already engaged builders. It is moving users out of the very large Ghost and shallow Viewer population before they disappear.

## The Core Narrative

### 1. Zerve's biggest problem is shallow activation, not late retention

The funnel narrows almost immediately:

- `1,922` users signed up
- `523` ever ran code manually
- `649` ever used an agent tool
- `292` reached agent-building behavior
- `117` reached `5+` active days

This means Zerve's main failure mode is early stall-out. Most users never reach a meaningful creation loop.

That same story shows up in return behavior:

- `3,458` users never came back for a second session
- among users returning within `6h`, `25.6%` became Agent Builders
- among users returning later than `6h`, only `5.9%` became Agent Builders
- among users returning within `24h`, `22.8%` became Agent Builders
- among users who never returned, only `0.8%` became Agent Builders

The product is not mostly losing users late. It is losing them before they ever become active builders.

### 2. Success on Zerve is builder behavior, not mere AI exposure

The operational success proxy is still `Agent Builder`, defined around meaningful agent-assisted build activity rather than passive chat exposure.

That definition continues to hold up because it is:

- behaviorally distinct
- stable under threshold sensitivity checks
- strongly tied to return and multi-day usage
- clearly connected to monetizable pressure such as credit usage

The key interpretation is simple: on Zerve, "used AI once" is not success. Entering a build-run-refactor loop is.

### 3. First-session depth matters more than onboarding compliance

The strongest predictive signals are not classic onboarding-completion metrics. They are depth and intent signals from the first real interaction window.

Inside the narrowed model on agent users, the most important features were:

| Feature | Importance |
|---|---:|
| `first_session_event_types` | 0.211 |
| `time_to_return_hours` | 0.130 |
| `first_session_duration_min` | 0.120 |
| `first_session_events` | 0.101 |
| `had_second_session` | 0.061 |

This is one of the cleanest findings in the project. The stronger users are not the ones who simply finish onboarding. They are the ones who explore more broadly, stay longer, do more in the first session, and return quickly.

### 4. The first 48 hours are the decisive activation window

The activation milestone analysis is still the strongest behavioral section of the entire case.

| Milestone | Agent Builder rate if yes | Agent Builder rate if no | Lift |
|---|---:|---:|---:|
| `reached_3_build_48h` | 45.3% | 0.8% | 56.2x |
| `used_agent_tool_48h` | 43.6% | 0.8% | 53.8x |
| `created_block_48h` | 44.4% | 0.8% | 53.4x |
| `had_10plus_events_48h` | 24.1% | 0.5% | 52.0x |
| `refactored_48h` | 46.9% | 1.1% | 42.5x |
| `had_5plus_event_types` | 18.8% | 0.5% | 35.1x |

The message here is not just "get people to the agent." It is:

- get them into real construction
- get them iterating early
- get them doing enough meaningful work fast enough that a return session becomes likely

### 5. Agent-assisted building looks like the central product flywheel

The predictive and quasi-causal sections point in the same direction.

Main model results:

| Model | Population | AUC |
|---|---|---:|
| Random Forest | Full population | 0.990 |
| Random Forest | Agent users only | 0.906 |
| Random Forest churn proxy | Active-user slice | 0.880 |

The full-population model is very strong because so many users never enter meaningful creation at all. The more interesting result is the `0.906` AUC model inside already-engaged agent users. Even there, quick return and first-session depth separate future builders from everyone else.

The propensity-score matching results remain one of the strongest directional findings in the repo:

- `+46.4pp` Agent Builder share
- `+1.79` active days
- `+17.4pp` return rate

after matching agent users to similar non-agent users on observable characteristics.

The careful interpretation is that agent-assisted building is not just correlated with success. It appears to materially help create it.

### 6. Successful users behave iteratively, not linearly

Zerve's strongest users do not move through a neat one-way funnel. They enter an iterative working loop.

The most common Agent Builder motifs include:

- `CREATE -> RUN`
- `RUN -> CREATE`
- `RUN -> REFACTOR -> RUN`
- `FINISH -> SUMMARY -> GET`

The top Agent Builder trigram in the session/tool analysis is still `RUN -> CREATE -> RUN` with `4,700` occurrences.

This is the real product story: Zerve works when users stop behaving like tour takers and start behaving like active builders refining a live canvas.

### 7. Step 10 is now corrected, and it strengthens the case

Earlier, the time-to-Agent-Builder section was inconsistent. In the current run, that weak point is fixed.

It now shows:

- `625` users reached a third build call
- median time-to-Agent-Builder is `0.0` days
- `86.4%` reach it within `1` day
- `92.8%` reach it within `7` days
- `97.8%` reach it within `30` days

That sharpens the overall conclusion: once users truly convert into builder behavior, they usually do it quickly.

### 8. Productive struggle is often a value signal, not just friction

The credit and error analysis remains one of the most useful business sections.

Credit pressure:

- Agent Builders: `43.86%` any burn
- Ghosts: `0.86%`
- the `21-50` burn bucket has the highest Agent Builder share at `48.84%`

Error assist:

- only `95` users used error assist
- `47.37%` of them were Agent Builders
- they averaged `9.60` active days and `361.27` tool calls

This is an important product nuance. Some pain signals are really signs of engaged work. They should often trigger support, education, or upgrade logic, not just warnings.

### 9. Repeat-canvas behavior is one of the strongest markers of durable usage

Canvas revisitation remains one of the clearest separators in the whole analysis.

| Group | Users | Avg days active | Second-session rate | Agent Builder share | Avg canvas complexity |
|---|---:|---:|---:|---:|---:|
| One-off canvas | 4,505 | 1.20 | 23.3% | 3.9% | 0.73 |
| Repeat canvas | 266 | 8.34 | 99.6% | 40.6% | 15.65 |

This is not a subtle difference. Returning to the same canvas is a marker of real work.

The churn proxy model supports the same story. Important later-stage drivers include:

- `time_to_return_hours`
- `avg_canvas_active_days`
- `repeat_canvas_count`
- `repeat_canvas_users`
- `agent_messages`
- `ttf_agent_chat_min`

Once a user comes back to the same canvas and deepens it, they look much closer to durable value creation.

### 10. The new intervention layer makes the analysis operational

Phase 18 turns the analysis into a targeting system.

Current intervention mix:

| Intervention | Users | Avg priority | Avg churn risk | Builder share |
|---|---:|---:|---:|---:|
| Activation nudge | 3,393 | 0.044 | 0.010 | 0.85% |
| Retention rescue | 846 | 0.417 | 0.875 | 21.04% |
| Monitor | 518 | 0.183 | 0.169 | 13.71% |
| Builder acceleration | 8 | 0.444 | 0.628 | 62.5% |
| Productive struggle support | 6 | 0.429 | 0.217 | 33.3% |

This adds a concrete product translation layer:

- most users need activation help
- a meaningful middle slice needs retention rescue
- a very small but valuable slice deserves builder acceleration or struggle support

### 11. Productive struggle and abandonment-prone struggle are meaningfully different

Phase 19 adds an important refinement: not all struggling users are the same.

| Struggle class | Users | Avg churn prob | Avg recovery intensity | Agent Builder share |
|---|---:|---:|---:|---:|
| Productive struggle | 15 | 0.113 | 0.579 | 40.0% |
| Abandonment-prone struggle | 121 | 0.891 | 0.358 | 33.88% |
| Mixed/uncertain struggle | 231 | 0.154 | 0.367 | 17.32% |
| No visible struggle | 4,404 | 0.164 | 0.000 | 4.50% |

This is very actionable. The same high-friction surface can contain both valuable builders who need help and users at serious risk of dropping out.

### 12. Early path branching now completes the story

Phase 20 adds a path-branching layer to show where trajectories diverge.

The strongest publishable branch points are not the very low-support `100%` gaps. They are the higher-support ones, for example:

- `AUTH -> ONBOARD -> AGENT_CHAT -> OTHER -> AGENT_OTHER`
- `CREDITS -> OTHER -> CREDITS -> AGENT_OTHER -> AGENT_BUILD`
- `AUTH -> ONBOARD -> AGENT_CHAT -> CREDITS -> OTHER`
- `OTHER -> AUTH -> ONBOARD -> AGENT_CHAT` often leading toward Ghost/Viewer dead ends when not followed by build-oriented actions

So the story is now richer than "builders do more." It is: builder and non-builder paths branch early, and the next action after chat/onboarding/credits often determines whether a user starts building or stalls.

### 13. Geography matters, but it is not the main explanation

Geo remains secondary to activation mechanics.

Countries above a meaningful threshold:

- `IN`: `2,025` users, `6.27%` Agent Builders, `27.56%` repeat session
- `US`: `838` users, `4.30%` Agent Builders, `21.36%` repeat session
- `IE`: `220` users, `8.18%` Agent Builders, `41.36%` repeat session
- `FR`: `88` users, `13.64%` Agent Builders, `48.86%` repeat session

The conclusion is not that one country explains the product. The conclusion is:

1. core activation behavior matters more than geography
2. geo can refine targeting after activation is fixed
3. `Unknown` geo remains large enough that country stories must stay cautious

## What Changed Since The Original Technical Document

The original roadmap items are no longer future work. They are implemented:

- canvas complexity
- churn proxy modeling
- n-gram workflow analysis
- geo analysis
- intervention scoring
- quality-of-struggle analysis
- path branching analysis

A few important updates from the older framing:

- the project is now a full multi-phase pipeline through step 20
- the earlier step-10 inconsistency is corrected in the current run
- the current feature base has `79` engineered features
- churn is explicitly a `14-day survival-style churn proxy`
- the analysis now supports product intervention strategy, not just descriptive insight

## Recommended Product Actions

### 1. Redesign onboarding around a build-run moment

Do not optimize mainly for tour completion. Optimize for the first real build or run action.

### 2. Intervene before the 6-hour and 24-hour return windows expire

Return speed is too predictive to ignore. Re-entry prompts, context restoration, and reminder flows should be targeted before those windows close.

### 3. Treat credit pain and error assist as support and monetization moments

These often identify engaged users, not just broken experiences. Use them for help, recovery, and upgrade prompts.

### 4. Detect repeat-canvas behavior early and accelerate it

Users returning to the same canvas are dramatically closer to durable value. Save context well, surface next steps, and reduce re-entry friction there.

### 5. Use lifecycle messaging by behavioral trajectory

The most useful behavioral groups are:

- Ghost
- Viewer
- Builder-in-progress
- Agent Builder
- productive struggle
- abandonment-prone struggle

That is a better targeting basis than geography alone.

## Why This Is A Strong Zerve Case

This project now does more than describe user behavior.

It:

- builds a reusable end-to-end pipeline
- combines descriptive, predictive, and quasi-causal analysis
- adds workflow, canvas, churn, geo, intervention, struggle, and branching layers
- turns behavioral analytics into product decisions
- is organized to be reproducible on Zerve under the competition rules

The clearest final narrative is:

> Zerve succeeds when users quickly enter an iterative agent-assisted build loop, return soon, revisit the same canvas, and keep working through real, sometimes messy workflows. The biggest opportunity is to help more users cross that activation threshold before they disappear into the Ghost majority.
