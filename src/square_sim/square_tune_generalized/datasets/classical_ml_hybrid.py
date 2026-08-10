from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ml_to_llm_hybrid(rows: int, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classical_confidence = rng.beta(5, 2, size=rows)
    exception_risk = rng.beta(2, 5, size=rows)
    evidence_gap = rng.beta(2.5, 4.0, size=rows)
    explanation_need = np.clip(exception_risk * 0.55 + evidence_gap * 0.35 + rng.normal(0, 0.05, rows), 0, 1)
    return pd.DataFrame(
        {
            "row_id": [f"hy-{seed}-{i}" for i in range(rows)],
            "source_dataset": "ml_to_llm_hybrid_proxy_v1",
            "track": "ml_to_llm",
            "classical_confidence": classical_confidence,
            "exception_risk": exception_risk,
            "evidence_gap": evidence_gap,
            "explanation_need": explanation_need,
            "calibration_error": np.clip((1 - classical_confidence) * 0.35 + rng.normal(0, 0.04, rows), 0, 1),
            "human_review_needed": (explanation_need > 0.55).astype(int),
            "action_risk": np.clip(exception_risk * 0.45 + evidence_gap * 0.25, 0, 1),
            "input_text": [f"Hybrid decision case {i}: prediction, evidence, and policy context." for i in range(rows)],
        }
    )
