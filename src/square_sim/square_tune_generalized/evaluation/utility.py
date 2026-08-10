from __future__ import annotations


def cost_adjusted_utility(final_utility: float, compute_cost_proxy: float, regression_count: float = 0.0) -> float:
    return float(final_utility - 0.28 * compute_cost_proxy - 0.035 * regression_count)


def experiments_to_threshold(final_utility: float, threshold: float = 0.75) -> int:
    if final_utility >= threshold:
        return 4
    return int(4 + (threshold - final_utility) * 30)
