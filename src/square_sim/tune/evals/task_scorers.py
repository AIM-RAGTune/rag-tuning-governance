from __future__ import annotations

from typing import Any

import pandas as pd


def final_metrics(trajectory: pd.DataFrame, branch_diagnostics: pd.DataFrame) -> dict[str, Any]:
    if trajectory.empty:
        return {}
    first = trajectory.iloc[0]
    last = trajectory.iloc[-1]
    final_utility = float(last["state_utility"])
    initial_utility = float(first["state_utility"])
    cost = max(float(last["cost_so_far"]), 1e-9)
    regression_count = int(last["regression_count"])
    return {
        "final_utility": final_utility,
        "utility_improvement": float(final_utility - initial_utility),
        "area_under_improvement_curve": float(trajectory["state_utility"].mean()),
        "regression_count": regression_count,
        "worst_regression": float(branch_diagnostics["realized_regression"].max()) if not branch_diagnostics.empty else 0.0,
        "preserved_known_good_score": float(max(0.0, 1.0 - regression_count / max(len(trajectory), 1))),
        "cost_adjusted_improvement": float((final_utility - initial_utility) / cost),
        "experiments_to_threshold": int(_experiments_to_threshold(trajectory)),
        "data_efficiency": float((final_utility - initial_utility) / max(float(last.get("data_volume", 1)), 1.0)),
        "adapter_efficiency": float((final_utility - initial_utility) / max(float(last.get("adapter_parameter_proxy", 8)), 1.0)),
        "branch_efficiency": float((final_utility - initial_utility) / max(float(branch_diagnostics.shape[0]), 1.0)),
        "calibration_score": _calibration(branch_diagnostics),
        "hallucination_proxy_reduction": float(max(0.0, last.get("retrieval_faithfulness", 0.0) - first.get("retrieval_faithfulness", 0.0))),
        "faithfulness_proxy_improvement": float(last.get("retrieval_faithfulness", 0.0) - first.get("retrieval_faithfulness", 0.0)),
        "safety_regression_rate": float(regression_count / max(len(trajectory), 1)),
        "simulated_gpu_hours": cost,
    }


def _experiments_to_threshold(trajectory: pd.DataFrame, threshold: float = 0.62) -> int:
    hits = trajectory[trajectory["state_utility"] >= threshold]
    return int(hits.iloc[0]["round_idx"]) if not hits.empty else len(trajectory)


def _calibration(branch_diagnostics: pd.DataFrame) -> float:
    if branch_diagnostics.empty:
        return 0.0
    err = (branch_diagnostics["predicted_utility"] - branch_diagnostics["realized_utility"]).abs().mean()
    return float(max(0.0, 1.0 - err))
