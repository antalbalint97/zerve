# Zerve × HackerEarth 2026 — 5-Minute Presentation Script

> **Delivery notes:** ~750 words, ~5 min at a steady pace. Dashboard is live on screen throughout.
> Cues in *[brackets]* indicate what to show or click — do not read aloud.

---

## OPENING — 0:00–0:30

"Let me ask a simple question: what does success actually look like on a platform like Zerve?

Zerve is an AI-powered data science notebook. The agent writes and runs code on your behalf. That's the promise. Our question for this hackathon was: of the 4,771 real users we analyzed — how many of them actually experienced that promise?

The answer is 285. Six percent.

That's not a retention problem. That's an activation problem. And that's what this analysis is about."

*[Open dashboard — Overview tab, activation funnel visible]*

---

## THE FUNNEL — 0:30–1:15

"Look at this funnel. We start with 4,771 registered users. Roughly a third ran at least one tool. About 650 ever used the AI agent. And only 285 used it consistently enough to be classified as Agent Builders.

Each step of this funnel is a dropout point. Most of the loss happens right at the top — users sign up and do nothing. Sixty-three-point-seven percent of the entire user base are what we call Ghosts. Zero tool interactions. Zero.

This is not people trying and failing. This is people never starting."

*[Switch to Segments tab — show behavioral comparison table]*

---

## THE SEGMENTS — 1:15–2:00

"When we look at the segments side by side, the picture becomes very clear.

Agent Builders average 157 tool interactions and 4.75 active days per user. Ninety percent return for a second session. They're pushing the platform hard — six percent hit credit limits.

Everyone else? Manual Coders come closest, but they barely use the agent. Viewers return at 52%. Ghosts at 8%.

This is not a marginal difference. It's structural. The Agent Builder segment is categorically different from every other user in the dataset — and that difference is entirely explained by one thing: they reached the agent.

The agent is not a feature. It is the retention mechanism."

*[Switch to Cohorts tab]*

---

## THE COHORT COLLAPSE — 2:00–2:40

"Now here's the most urgent signal in the entire dataset.

September 2025 cohort: 19.1% Agent Builder rate. October: 19.9%. These are healthy numbers. Then November: zero. December: zero. Across 3,300 users — two thirds of the analyzed base — not a single Agent Builder.

We need to be careful here. November and December users had fewer days to accumulate behavior before the dataset ended. Time-horizon bias is real. But September and October cohorts had 19-plus percent within their first few weeks. The observation window alone does not explain a complete collapse.

This warrants immediate investigation. UI change, feature regression, acquisition channel shift — something happened. Until we know what, no growth initiative targeting these cohorts will be effective."

*[Switch to Retention & Survival tab — show survival curves]*

---

## RETENTION & THE MODEL — 2:40–3:30

"On the retention side — look at these survival curves. Agent Builders decay slower. Users who return within 24 hours survive dramatically longer than those who come back late, and far longer than those who never return.

Early return timing is the single strongest behavioral signal in the dataset. And the activation milestone analysis shows that users who reach three agent build calls within 48 hours have a 45% chance of becoming Agent Builders — versus 0.8% for those who don't. That's a 56x lift.

The modeling layer confirms this."

*[Switch to Modeling tab — show ROC curve and feature importance]*

"Three models were trained to predict Agent Builder status. All three hit AUC above 0.99 on the full population. The more interesting result is the narrowed model — trained only on first-session signals. It still performs well, and it tells us something actionable: the breadth of events a user explores in their first session, how quickly they return, how long that session lasts — these are detectable within hours of signup. That's the intervention window."

*[Switch to Churn and Interventions tabs]*

---

## CHURN & INTERVENTIONS — 3:30–4:15

"Churn among active users sits at 83.3%. Even users who started are not staying. And most of it is silent — they don't struggle visibly, they just stop coming back.

But here's what makes this actionable: the behavioral scoring is already done.

3,393 users have been flagged for activation nudges. 846 for retention rescue. 8 for builder acceleration — these are users on the cusp of consistent Agent Builder behavior. The targeting infrastructure exists. These users are identified. The only thing missing is deployment.

Every day without deploying this scoring is permanent user loss."

---

## CLOSE — 4:15–5:00

"So — three actions, one north star.

**Action one:** Redesign onboarding to make the first agent interaction mandatory. Not optional. Not a tooltip. A guided build, in session one, before users have a reason to leave.

**Action two:** Emergency audit of the November–December cohort collapse. Root cause identified within two weeks.

**Action three:** Deploy the scored intervention model now. The work is done. The users are waiting.

And the north star metric is simple: percentage of new users who complete at least one agent-assisted build within their first seven days. Currently at 13.6%. The target is 30 percent within 90 days of an onboarding redesign.

The product already works. The challenge is getting users there. Thank you."

---

*Analysis: 4,771 users · Sep–Dec 2025 cohorts · Random Forest churn model · Propensity-matched intervention scoring*
