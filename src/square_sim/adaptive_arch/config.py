from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ARCH_TASKS = [
    "local_regime_shift",
    "future_rollout_required",
    "merge_required_architecture",
    "memory_prevents_repeated_failure",
    "dynamic_topology_routing",
    "compute_allocation_trap",
    "nonlinear_extrapolation_required",
    "protect_known_good_while_adapting",
    "linear_static_control",
    "random_unlearnable_control",
]

ARCH_SYSTEMS = [
    "static_policy",
    "greedy_immediate",
    "greedy_regression_aware",
    "coordinate_descent",
    "evolutionary_search",
    "optuna_tpe_optional",
    "random_search",
    "linear_static_baseline",
    "square_tune_full",
    "square_tune_no_fork",
    "square_tune_no_merge",
    "square_tune_no_memory",
    "square_tune_no_snapshot",
    "square_tune_no_feedback",
    "square_tune_no_cost_sensor",
    "square_tune_no_regression_sensor",
    "square_tune_adaptive_compute",
    "square_adaptive_arch_full",
    "square_adaptive_arch_no_local_reconfiguration",
    "square_adaptive_arch_no_snapshot",
    "square_adaptive_arch_no_fork",
    "square_adaptive_arch_no_rollout",
    "square_adaptive_arch_linear_rollout",
    "square_adaptive_arch_no_merge",
    "square_adaptive_arch_no_memory",
    "square_adaptive_arch_static_topology",
    "square_adaptive_arch_always_fork",
    "square_adaptive_arch_never_fork",
    "square_adaptive_arch_no_compute_gate",
    "square_adaptive_arch_no_regression_protection",
]

TASK_TO_MECHANISM = {
    "local_regime_shift": "failure_cluster_routing",
    "future_rollout_required": "rag_policy_conflict",
    "merge_required_architecture": "merge_required",
    "memory_prevents_repeated_failure": "repeated_regression_memory",
    "dynamic_topology_routing": "tool_routing",
    "compute_allocation_trap": "rag_policy_conflict",
    "nonlinear_extrapolation_required": "nonmonotonic_data_mix",
    "protect_known_good_while_adapting": "regression_veto",
    "linear_static_control": "linear_control",
    "random_unlearnable_control": "random_label",
    "rag_policy_adaptive_arch": "rag_policy_conflict",
    "claim_level_faithfulness_adaptive_arch": "data_poison_regression",
    "tool_routing_dynamic_topology_proxy": "tool_routing",
    "prompt_regression_memory_proxy": "prompt_regression",
    "data_curation_regime_shift_proxy": "nonmonotonic_data_mix",
}

SYSTEM_TO_OPTIMIZER = {
    "static_policy": "linear_utility_optimizer",
    "greedy_immediate": "greedy_eval_improvement",
    "linear_static_baseline": "linear_utility_optimizer",
    "square_adaptive_arch_full": "square_tune_adaptive_compute",
    "square_adaptive_arch_no_local_reconfiguration": "square_tune_no_snapshot",
    "square_adaptive_arch_no_snapshot": "square_tune_no_snapshot",
    "square_adaptive_arch_no_fork": "square_tune_no_fork",
    "square_adaptive_arch_no_rollout": "square_tune_no_fork",
    "square_adaptive_arch_linear_rollout": "square_tune_linear_rollout",
    "square_adaptive_arch_no_merge": "square_tune_no_merge",
    "square_adaptive_arch_no_memory": "square_tune_no_memory",
    "square_adaptive_arch_static_topology": "square_tune_global_only",
    "square_adaptive_arch_always_fork": "square_tune_adaptive_compute_always_fork",
    "square_adaptive_arch_never_fork": "square_tune_adaptive_compute_never_fork",
    "square_adaptive_arch_no_compute_gate": "square_tune_adaptive_compute_always_fork",
    "square_adaptive_arch_no_regression_protection": "square_tune_no_regression_sensor",
}


@dataclass(frozen=True)
class AdaptiveArchConfig:
    experiment_name: str
    tasks: list[str]
    seeds: list[int]
    systems: list[str]
    dataset_root: str | None = None
    max_rounds: int = 6
    num_branches: int = 6
    rollout_steps: int = 3
    max_response_surface_evaluations: int = 144
    max_candidate_actions: int = 144
    simulated_gpu_hour_budget: float = 8.0

    @classmethod
    def from_path(cls, path: Path) -> AdaptiveArchConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        budget = raw.get("budget") or raw.get("square_tune") or {}
        return cls(
            experiment_name=str(raw.get("experiment_name", path.stem)),
            tasks=list(raw.get("tasks", raw.get("scenarios", []))),
            seeds=[int(seed) for seed in raw.get("seeds", [101])],
            systems=list(raw.get("systems", raw.get("optimizers", []))),
            dataset_root=str(raw.get("dataset_root")) if raw.get("dataset_root") else None,
            max_rounds=int(budget.get("max_rounds", 6)),
            num_branches=int(budget.get("num_branches", 6)),
            rollout_steps=int(budget.get("rollout_steps", 3)),
            max_response_surface_evaluations=int(budget.get("max_response_surface_evaluations", 144)),
            max_candidate_actions=int(budget.get("max_candidate_actions", 144)),
            simulated_gpu_hour_budget=float(budget.get("simulated_gpu_hour_budget", 8.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

