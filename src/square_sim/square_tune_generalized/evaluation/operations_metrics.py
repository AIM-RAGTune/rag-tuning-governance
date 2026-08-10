from __future__ import annotations

import pandas as pd


def patient_flow_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "predictive_quality": float((1.0 - frame.get("boarding_risk", pd.Series([0.5])) * 0.35).mean()),
        "explanation_faithfulness": float((1.0 - frame.get("unsafe_recommendation_risk", pd.Series([0.2]))).mean()),
        "operational_action_score": float((frame.get("boarding_risk", pd.Series([0.5])) > 0.55).mean()),
        "bottleneck_identification_score": float(frame.get("bottleneck_label", pd.Series(["x"])).nunique() / 3.0),
        "escalation_precision": float((frame.get("boarding_risk", pd.Series([0.5])) > 0.65).mean()),
        "escalation_recall": float((frame.get("admission_risk", pd.Series([0.5])) > 0.60).mean()),
        "unsafe_recommendation_count": float((frame.get("unsafe_recommendation_risk", pd.Series([0.0])) > 0.55).sum()),
        "human_review_burden_proxy": float(frame.get("boarding_risk", pd.Series([0.5])).mean()),
    }
