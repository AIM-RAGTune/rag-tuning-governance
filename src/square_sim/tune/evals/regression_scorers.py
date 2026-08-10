from __future__ import annotations


def regression_count(before: dict[str, float], after: dict[str, float], threshold: float = 0.02) -> int:
    return sum(1 for key, value in before.items() if after.get(key, value) < value - threshold)

