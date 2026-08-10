from __future__ import annotations


def memory_match_score(known_good: int, known_bad: int) -> float:
    return float(min(1.0, max(0.0, (known_good + known_bad) / 10.0)))
