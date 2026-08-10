from __future__ import annotations

import pandas as pd


def ml_to_llm_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "predictive_performance": float(frame.get("classical_confidence", pd.Series([0.6])).mean()),
        "explanation_faithfulness": float((1.0 - frame.get("evidence_gap", pd.Series([0.3])) * 0.4).mean()),
        "action_correctness": float((1.0 - frame.get("action_risk", pd.Series([0.3]))).mean()),
        "calibration": float((1.0 - frame.get("calibration_error", pd.Series([0.2]))).mean()),
        "exception_handling_score": float(frame.get("human_review_needed", pd.Series([0])).mean()),
        "human_review_reduction_proxy": float((1.0 - frame.get("explanation_need", pd.Series([0.4]))).mean()),
        "classical_model_preservation_score": float(frame.get("classical_confidence", pd.Series([0.6])).mean()),
    }
