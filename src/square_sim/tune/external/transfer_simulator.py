from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.config import TuneBudget
from square_sim.tune.simulator.square_tune_optimizer import run_optimizer

MECHANISM_BY_SCENARIO = {
    "rag_policy_optimization": "rag_policy_conflict",
    "hallucination_faithfulness_reduction": "data_poison_regression",
    "prompt_regression_optimization": "prompt_regression",
    "data_curation_preference_optimization": "nonmonotonic_data_mix",
    "adapter_planning_simulation": "adapter_tradeoff",
    "tool_routing_policy_optimization": "tool_routing",
}


def run_external_transfer_optimizer(
    *,
    optimizer_name: str,
    train_path: Path,
    scenario_family: str,
    seed: int,
    budget: TuneBudget,
    objective_weights: dict[str, float],
) -> dict[str, Any]:
    df = pd.read_parquet(train_path)
    mechanism_name = MECHANISM_BY_SCENARIO.get(scenario_family, scenario_family)
    result = run_optimizer(
        optimizer_name,
        df,
        mechanism_name=mechanism_name,
        seed=seed,
        budget=budget,
        objective_weights=objective_weights,
    )
    metrics = dict(result.metrics)
    final_utility = float(metrics.get("final_utility", 0.0))
    regression_count = float(metrics.get("regression_count", 0.0))
    cost = max(float(metrics.get("simulated_gpu_hours", 0.0)), 1e-6)
    metrics.update(
        {
            "scenario_family": scenario_family,
            "external_transfer_proxy": True,
            "rag_faithfulness_proxy": float(min(1.0, max(0.0, final_utility - 0.03 * regression_count))),
            "hallucination_reduction_proxy": float(min(1.0, max(0.0, final_utility - 0.04 * regression_count))),
            "prompt_instruction_pass_proxy": float(min(1.0, max(0.0, final_utility - 0.02 * regression_count))),
            "tool_policy_accuracy_proxy": float(min(1.0, max(0.0, final_utility - 0.01 * cost))),
            "budget_consumed": {
                "response_surface_evaluations": metrics.get("actual_response_surface_evaluations"),
                "candidate_actions_scored": metrics.get("actual_candidate_actions_scored"),
                "simulated_gpu_hours": metrics.get("simulated_gpu_hours"),
            },
        }
    )
    return {
        "metrics": metrics,
        "trajectory": result.trajectory,
        "branch_diagnostics": result.branch_diagnostics,
        "adaptive_diagnostics": result.adaptive_diagnostics,
        "final_policy": result.final_policy,
        "initial_state": result.initial_state.to_dict(),
        "final_state": result.final_state.to_dict(),
    }
