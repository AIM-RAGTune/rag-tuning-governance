from __future__ import annotations

from collections.abc import Mapping


def cost_adjusted_utility(
    *,
    quality: float,
    cost: float,
    latency: float,
    regression: float,
    weights: Mapping[str, float],
) -> float:
    return float(
        weights.get("quality", 1.0) * quality
        - weights.get("cost", 0.25) * cost
        - weights.get("latency", 0.10) * latency
        - weights.get("regression", 0.50) * regression
    )


def budget_deviation_pct(observed: float, target: float) -> float:
    if target <= 0:
        return 0.0 if observed <= 0 else 100.0
    return float(abs(observed - target) / target * 100.0)
