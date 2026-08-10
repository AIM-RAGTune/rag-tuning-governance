from __future__ import annotations

from typing import Any

import pandas as pd

SUBSET_NAMES = [
    "high_retrieval_conflict",
    "low_context_confidence",
    "high_hallucination_risk",
    "citation_sensitive",
    "multi_source_disagreement",
    "long_context",
    "abstention_borderline",
    "high_uncertainty",
    "high_cost_variance",
    "hard_composite",
]


def required_columns() -> set[str]:
    return {"uncertainty", "retrieval_confidence", "retrieval_conflict", "hallucination_labels_optional"}


def subset_availability(frame: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(required_columns() - set(frame.columns))
    return {"available": not missing, "missing_columns": missing}


def detect_hard_subsets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    availability = subset_availability(frame)
    if not availability["available"] or frame.empty:
        return pd.DataFrame(index=frame.index), {**availability, "subsets": {name: {"available": False} for name in SUBSET_NAMES}}
    uncertainty = frame["uncertainty"].astype(float)
    retrieval = frame["retrieval_confidence"].astype(float)
    conflict = frame["retrieval_conflict"].astype(float)
    hallucination = frame["hallucination_labels_optional"].astype(float)
    context_len = frame.get("retrieved_contexts", pd.Series([""] * len(frame), index=frame.index)).astype(str).str.len()
    cost_var = (context_len - context_len.median()).abs()
    composite = uncertainty + conflict + hallucination + (1.0 - retrieval)
    masks = pd.DataFrame(
        {
            "high_retrieval_conflict": conflict.ge(conflict.quantile(0.70)),
            "low_context_confidence": retrieval.le(retrieval.quantile(0.30)),
            "high_hallucination_risk": hallucination.ge(hallucination.quantile(0.70)),
            "citation_sensitive": context_len.ge(context_len.quantile(0.50)) & retrieval.le(retrieval.quantile(0.50)),
            "multi_source_disagreement": conflict.ge(conflict.quantile(0.80)),
            "long_context": context_len.ge(context_len.quantile(0.75)),
            "abstention_borderline": uncertainty.between(0.45, 0.65, inclusive="both"),
            "high_uncertainty": uncertainty.ge(uncertainty.quantile(0.70)),
            "high_cost_variance": cost_var.ge(cost_var.quantile(0.75)),
            "hard_composite": composite.ge(composite.quantile(0.75)),
        },
        index=frame.index,
    )
    profile = {
        **availability,
        "subsets": {
            name: {"available": True, "row_count": int(masks[name].sum()), "fraction": float(masks[name].mean())}
            for name in masks.columns
        },
    }
    return masks, profile

