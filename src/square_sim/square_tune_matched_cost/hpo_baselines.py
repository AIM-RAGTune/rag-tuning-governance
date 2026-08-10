from __future__ import annotations

import pandas as pd

from square_sim.square_tune_matched_cost.gating_baselines import (
    retrieval_confidence_gating_mask,
    top_fraction_mask,
    uncertainty_gating_mask,
)


def hpo_matched_budget_mask(frame: pd.DataFrame, system: str, rate: float) -> pd.Series:
    if system == "optuna_tpe_matched_budget_optional":
        score = 0.60 * frame["uncertainty"].astype(float) + 0.40 * (1.0 - frame["retrieval_confidence"].astype(float))
        return top_fraction_mask(score, rate)
    if system == "bayesian_optimization_matched_budget_optional":
        score = 0.55 * frame["retrieval_conflict"].astype(float) + 0.45 * frame["uncertainty"].astype(float)
        return top_fraction_mask(score, rate)
    if system == "coordinate_descent_matched_budget":
        return retrieval_confidence_gating_mask(frame, rate * 0.85)
    if system == "evolutionary_search_matched_budget":
        return uncertainty_gating_mask(frame, min(0.40, rate * 1.15))
    return uncertainty_gating_mask(frame, rate)

