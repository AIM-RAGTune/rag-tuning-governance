from __future__ import annotations

import pandas as pd


def rag_proxy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "faithfulness_proxy": float(frame.get("faithfulness", pd.Series([0.5])).mean()),
        "answer_relevance_proxy": float((1.0 - frame.get("uncertainty", pd.Series([0.5])) * 0.35).mean()),
        "context_precision_proxy": float(frame.get("context_precision", pd.Series([0.5])).mean()),
        "context_recall_proxy": float(frame.get("context_recall", pd.Series([0.5])).mean()),
        "hallucination_reduction_proxy": float((1.0 - frame.get("hallucination_risk", pd.Series([0.5]))).mean()),
        "abstention_correctness": float((frame.get("policy_risk", pd.Series([0.5])) > 0.55).mean()),
        "source_grounding_score": float(frame.get("faithfulness", pd.Series([0.5])).mean() * 0.9),
        "citation_support_score": float(frame.get("context_precision", pd.Series([0.5])).mean() * 0.95),
    }
