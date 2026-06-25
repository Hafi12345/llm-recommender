"""
recommender.py
Weighted multi-criteria scoring engine for the LLM recommender.

There's no labeled "correct model" dataset to train a classifier on -- this
is a ranking problem, not a supervised learning one. So instead of training
a model, we calibrate a transparent, explainable scoring function:

Each model is scored on 4 normalized criteria (0-1, higher = better):
  - performance : benchmark composite (higher is better)
  - price       : blended cost per million tokens (LOWER raw price is better,
                   so this is inverted after normalization)
  - privacy     : data-handling / open-weight score (higher is better)
  - speed       : throughput / latency score (higher is better)

The user supplies relative IMPORTANCE weights for each criterion (any
positive numbers -- they're normalized to sum to 1 internally), and every
model gets a single 0-100 "match score" used to rank the results.

This is a Weighted Sum Model (WSM). For a future upgrade, TOPSIS
(distance from an ideal/worst hypothetical model) is a natural extension
that's more robust when criteria scales differ wildly -- left as a clean
extension point since the normalization step below is already isolated.
"""
from dataclasses import dataclass
from typing import Dict
import pandas as pd


CRITERIA = ("performance", "price", "privacy", "speed")


@dataclass
class Weights:
    performance: float = 0.25
    price: float = 0.25
    privacy: float = 0.25
    speed: float = 0.25

    def normalized(self) -> Dict[str, float]:
        total = self.performance + self.price + self.privacy + self.speed
        if total <= 0:
            raise ValueError("At least one weight must be greater than 0.")
        return {
            "performance": self.performance / total,
            "price": self.price / total,
            "privacy": self.privacy / total,
            "speed": self.speed / total,
        }


def _minmax(series: pd.Series) -> pd.Series:
    """Scales a series to [0, 1]. If every value is identical, returns all 1s
    (no model should be unfairly penalized when a criterion has no spread)."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def score_models(df: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    """Returns a copy of df with per-criterion normalized scores and a final
    weighted `match_score` (0-100), sorted best match first."""
    w = weights.normalized()
    out = df.copy()

    out["norm_performance"] = _minmax(out["performance_score"])
    out["norm_price"] = 1 - _minmax(out["blended_price_per_mtok"])  # cheaper -> higher score
    out["norm_privacy"] = _minmax(out["privacy_score"])
    out["norm_speed"] = _minmax(out["speed_score"])

    out["match_score"] = 100 * (
        w["performance"] * out["norm_performance"]
        + w["price"] * out["norm_price"]
        + w["privacy"] * out["norm_privacy"]
        + w["speed"] * out["norm_speed"]
    )

    return out.sort_values("match_score", ascending=False).reset_index(drop=True)


def recommend(df: pd.DataFrame, weights: Weights, top_n: int = 5) -> pd.DataFrame:
    """Top-N recommended models for the given weights, as a slim result frame."""
    scored = score_models(df, weights)
    cols = [
        "model_name", "provider", "match_score", "performance_score",
        "blended_price_per_mtok", "privacy_score", "speed_score",
        "open_weight", "context_window_k", "notes",
    ]
    return scored[cols].head(top_n)
