from __future__ import annotations


def requires_rollout(uncertainty: float, projected_roi: float, budget_pressure: float) -> bool:
    return bool(uncertainty > 0.55 and projected_roi > 0.05 and budget_pressure < 0.8)

