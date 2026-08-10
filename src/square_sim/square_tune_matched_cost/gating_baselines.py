from __future__ import annotations

import numpy as np
import pandas as pd


def top_fraction_mask(scores: pd.Series, rate: float) -> pd.Series:
    if len(scores) == 0:
        return pd.Series([], dtype=bool)
    k = round(len(scores) * rate)
    if k <= 0:
        return pd.Series([False] * len(scores), index=scores.index)
    order = scores.sort_values(ascending=False).index[:k]
    mask = pd.Series([False] * len(scores), index=scores.index)
    mask.loc[order] = True
    return mask


def random_gating_mask(frame: pd.DataFrame, rate: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    scores = pd.Series(rng.random(len(frame)), index=frame.index)
    return top_fraction_mask(scores, rate)


def uncertainty_gating_mask(frame: pd.DataFrame, rate: float) -> pd.Series:
    return top_fraction_mask(frame["uncertainty"].astype(float), rate)


def retrieval_confidence_gating_mask(frame: pd.DataFrame, rate: float) -> pd.Series:
    score = (1.0 - frame["retrieval_confidence"].astype(float)) + 0.35 * frame["retrieval_conflict"].astype(float)
    return top_fraction_mask(score, rate)


def entropy_or_margin_gating_mask(frame: pd.DataFrame, rate: float) -> pd.Series:
    score = 1.0 - (frame["uncertainty"].astype(float) - 0.5).abs() * 2.0
    return top_fraction_mask(score.clip(0, 1), rate)


def adaptive_compute_mask(frame: pd.DataFrame, rate: float = 0.20) -> pd.Series:
    score = (
        0.42 * frame["uncertainty"].astype(float)
        + 0.28 * frame["retrieval_conflict"].astype(float)
        + 0.20 * frame["hallucination_labels_optional"].astype(float)
        + 0.10 * (1.0 - frame["retrieval_confidence"].astype(float))
    )
    return top_fraction_mask(score.clip(0, 1), rate)
