from __future__ import annotations


def utility_from_error(error: float, cost: float = 1.0, penalty: float = 0.0) -> dict[str, float]:
    final = max(0.0, 1.0 - float(error) - float(penalty))
    return {"final_utility": final, "cost_adjusted_utility": final / max(float(cost), 1e-6)}
