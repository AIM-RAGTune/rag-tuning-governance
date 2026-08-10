from __future__ import annotations


def rag_tradeoff_score(faithfulness: float, latency: float) -> float:
    return float(faithfulness - 0.15 * latency)

