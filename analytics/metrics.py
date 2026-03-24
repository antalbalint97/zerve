from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


COMPLEXITY_WEIGHTS = {
    "structural_actions": 1.0,
    "agent_build_actions": 1.8,
    "execution_actions": 1.2,
    "dependency_actions": 1.3,
    "output_actions": 0.8,
    "active_days": 0.6,
}


def compute_canvas_complexity_score(row: pd.Series | dict) -> float:
    return float(sum(float(row.get(k, 0)) * w for k, w in COMPLEXITY_WEIGHTS.items()))


def split_first_vs_later_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "canvas_id",
                "first_day_score",
                "later_avg_day_score",
                "later_max_day_score",
                "later_active_days",
                "growth_delta",
            ]
        )
    tmp = df.copy()
    tmp["event_day"] = tmp["timestamp"].dt.normalize()
    tmp["first_canvas_day"] = tmp.groupby(["person_id", "canvas_id"])["event_day"].transform("min")
    tmp["phase"] = np.where(tmp["event_day"] == tmp["first_canvas_day"], "first", "later")
    daily = (
        tmp.groupby(["person_id", "canvas_id", "event_day", "phase"])
        .agg(
            structural_actions=("is_structural", "sum"),
            execution_actions=("is_execution", "sum"),
            dependency_actions=("is_dependency", "sum"),
            output_actions=("is_output", "sum"),
            agent_build_actions=("is_agent_build", "sum"),
            active_days=("event_day", "nunique"),
        )
        .reset_index()
    )
    daily["day_score"] = daily.apply(compute_canvas_complexity_score, axis=1)

    first_days = (
        daily[daily["phase"] == "first"]
        .groupby(["person_id", "canvas_id"])["day_score"]
        .max()
        .rename("first_day_score")
    )
    later_days = daily[daily["phase"] == "later"].copy()
    later_summary = (
        later_days.groupby(["person_id", "canvas_id"])
        .agg(
            later_avg_day_score=("day_score", "mean"),
            later_max_day_score=("day_score", "max"),
            later_active_days=("event_day", "nunique"),
        )
    )
    out = first_days.to_frame().join(later_summary, how="outer").reset_index()
    out[["first_day_score", "later_avg_day_score", "later_max_day_score", "later_active_days"]] = out[
        ["first_day_score", "later_avg_day_score", "later_max_day_score", "later_active_days"]
    ].fillna(0)
    out["growth_delta"] = np.where(
        out["later_active_days"] > 0,
        out["later_max_day_score"] - out["first_day_score"],
        0.0,
    )
    return out


def label_churn(
    activity: pd.DataFrame,
    horizon_days: int = 14,
    person_col: str = "person_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    if activity.empty:
        return pd.DataFrame(columns=[person_col, "last_active_at", "observation_end", "days_until_end", "is_censored", "is_churn"])
    last_active = activity.groupby(person_col)[time_col].max().rename("last_active_at")
    observation_end = activity[time_col].max()
    out = last_active.to_frame()
    out["observation_end"] = observation_end
    out["days_until_end"] = (observation_end - out["last_active_at"]).dt.total_seconds() / 86400
    out["is_censored"] = (out["days_until_end"] < horizon_days).astype(int)
    out["is_churn"] = ((out["days_until_end"] >= horizon_days) & (out["is_censored"] == 0)).astype(int)
    return out.reset_index()


def generate_ngrams(tokens: Iterable[str], n: int) -> list[tuple[str, ...]]:
    seq = [str(token) for token in tokens if token is not None and str(token) != ""]
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_counts(sequences: Iterable[Iterable[str]], n: int) -> Counter:
    counter: Counter = Counter()
    for seq in sequences:
        counter.update(generate_ngrams(seq, n))
    return counter


def top_ngram_lift(
    focal_sequences: Iterable[Iterable[str]],
    baseline_sequences: Iterable[Iterable[str]],
    n: int,
    min_count: int = 10,
    min_baseline_count: int = 0,
    top_k: int = 20,
    smoothing_alpha: float = 1.0,
) -> pd.DataFrame:
    focal = ngram_counts(focal_sequences, n)
    base = ngram_counts(baseline_sequences, n)
    vocab = set(focal) | set(base)
    vocab_size = max(len(vocab), 1)
    focal_total = max(sum(focal.values()), 1)
    base_total = max(sum(base.values()), 1)
    rows = []
    for gram, count in focal.items():
        if count < min_count:
            continue
        base_count = base.get(gram, 0)
        focal_support = count / focal_total
        baseline_support = base_count / base_total
        focal_smoothed = (count + smoothing_alpha) / (focal_total + smoothing_alpha * vocab_size)
        baseline_smoothed = (base_count + smoothing_alpha) / (base_total + smoothing_alpha * vocab_size)
        rows.append(
            {
                "ngram": " -> ".join(gram),
                "n": n,
                "focal_count": count,
                "baseline_count": base_count,
                "focal_support": focal_support,
                "baseline_support": baseline_support,
                "lift": focal_smoothed / baseline_smoothed,
                "publishable": int(base_count >= min_baseline_count),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["lift", "focal_count"], ascending=[False, False]).head(top_k)


def summarize_risk(probabilities: pd.Series) -> pd.Series:
    bins = [-0.01, 0.2, 0.5, 0.8, 1.0]
    labels = ["Low", "Medium", "High", "Critical"]
    return pd.cut(probabilities.clip(0, 1), bins=bins, labels=labels)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
