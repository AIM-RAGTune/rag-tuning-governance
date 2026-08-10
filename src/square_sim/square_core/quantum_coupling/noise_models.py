from __future__ import annotations


def noise_rate(regime: str, field_strength: float) -> float:
    if regime == "optimistic":
        return max(0.005, 0.04 - 0.02 * abs(field_strength))
    if regime == "neutral":
        return 0.035
    return 0.035 + 0.035 * abs(field_strength)
