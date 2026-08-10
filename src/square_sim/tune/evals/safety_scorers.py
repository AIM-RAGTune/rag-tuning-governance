from __future__ import annotations


def safety_regression(before: float, after: float, threshold: float = 0.02) -> bool:
    return after < before - threshold

