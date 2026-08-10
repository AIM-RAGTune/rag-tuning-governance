from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from square_sim.utils.hashing import stable_hash

TUNE_DATASETS = [
    "synthetic_llm_linear_control",
    "synthetic_llm_random_label",
    "synthetic_llm_failure_cluster_routing",
    "synthetic_llm_nonmonotonic_data_mix",
    "synthetic_llm_adapter_tradeoff",
    "synthetic_llm_rag_policy_conflict",
    "synthetic_llm_prompt_regression",
    "synthetic_llm_tool_routing",
    "synthetic_llm_data_poison_regression",
    "synthetic_llm_merge_required",
    "synthetic_llm_curriculum_order",
    "synthetic_llm_hard_external_transfer_proxy",
    "synthetic_llm_repeated_regression_memory",
    "synthetic_llm_regression_veto",
    "synthetic_llm_cost_tradeoff",
]

SQUARE_TUNE_VARIANTS = [
    "square_tune_full",
    "square_tune_no_snapshot",
    "square_tune_no_fork",
    "square_tune_linear_rollout",
    "square_tune_no_merge",
    "square_tune_no_memory",
    "square_tune_no_feedback",
    "square_tune_random_branch",
    "square_tune_global_only",
    "square_tune_no_regression_sensor",
    "square_tune_no_cost_sensor",
    "square_tune_adaptive_compute",
    "square_tune_adaptive_compute_no_uncertainty_gate",
    "square_tune_adaptive_compute_no_conflict_gate",
    "square_tune_adaptive_compute_no_roi_gate",
    "square_tune_adaptive_compute_no_budget_gate",
    "square_tune_adaptive_compute_no_regression_escalation",
    "square_tune_adaptive_compute_no_memory_reuse",
    "square_tune_adaptive_compute_always_fork",
    "square_tune_adaptive_compute_never_fork",
]

BASELINE_OPTIMIZERS = [
    "linear_utility_optimizer",
    "ridge_utility_optimizer",
    "coordinate_descent",
    "greedy_oracle_feature_baseline",
    "optuna_tpe_optional",
    "evolutionary_search",
    "random_search",
    "grid_search",
    "greedy_eval_improvement",
    "greedy_regression_aware",
    "bayesian_optional",
    "hyperband_optional",
    "evolutionary",
    "oracle_upper_bound",
]

TUNE_OPTIMIZERS = BASELINE_OPTIMIZERS + SQUARE_TUNE_VARIANTS

EVAL_METRICS = [
    "domain_accuracy",
    "retrieval_faithfulness",
    "instruction_following",
    "style_match",
    "safety",
    "latency",
    "cost",
    "calibration",
    "regression_score",
]

DEFAULT_OBJECTIVE_WEIGHTS = {
    "domain_accuracy": 0.30,
    "retrieval_faithfulness": 0.20,
    "instruction_following": 0.15,
    "safety": 0.15,
    "style_match": 0.05,
    "regression_score": 0.10,
    "cost": -0.05,
}


@dataclass(frozen=True)
class TuneBudget:
    max_rounds: int = 8
    num_branches: int = 6
    rollout_steps: int = 3
    simulated_gpu_hour_budget: float = 8.0
    eval_budget: int = 256
    max_response_surface_evaluations: int = 144
    max_candidate_actions: int = 144
    token_cost_proxy_budget: float = 1_000.0
    budget_ledger_enabled: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> TuneBudget:
        cfg = cfg or {}
        return cls(
            max_rounds=int(cfg.get("max_rounds", 8)),
            num_branches=int(cfg.get("num_branches", 6)),
            rollout_steps=int(cfg.get("rollout_steps", 3)),
            simulated_gpu_hour_budget=float(cfg.get("simulated_gpu_hour_budget", 8.0)),
            eval_budget=int(cfg.get("eval_budget", 256)),
            max_response_surface_evaluations=int(
                cfg.get("max_response_surface_evaluations", cfg.get("eval_budget", 144))
            ),
            max_candidate_actions=int(cfg.get("max_candidate_actions", cfg.get("eval_budget", 144))),
            token_cost_proxy_budget=float(cfg.get("token_cost_proxy_budget", 1_000.0)),
            budget_ledger_enabled=bool(cfg.get("budget_ledger_enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TuneExperimentConfig:
    experiment_name: str
    datasets: list[str]
    seeds: list[int]
    optimizers: list[str]
    dataset_root: str | None = None
    protocol_path: str = "protocols/square_tune/synthetic_mechanism_protocol_v1.yaml"
    bootstrap_samples: int = 100
    budget: TuneBudget = field(default_factory=TuneBudget)
    objective_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy())

    @classmethod
    def from_path(cls, path: Path) -> TuneExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        square_tune = raw.get("square_tune", {})
        return cls(
            experiment_name=str(raw.get("experiment_name", path.stem)),
            datasets=list(raw.get("datasets", [])),
            seeds=[int(v) for v in raw.get("seeds", [101])],
            optimizers=list(raw.get("optimizers", [])),
            dataset_root=os.path.expandvars(str(raw["dataset_root"])) if raw.get("dataset_root") else None,
            protocol_path=str(raw.get("protocol_path", cls.protocol_path)),
            bootstrap_samples=int(raw.get("bootstrap_samples", 100)),
            budget=TuneBudget.from_config(square_tune or raw.get("budget")),
            objective_weights=dict(square_tune.get("objective_weights", raw.get("objective_weights", DEFAULT_OBJECTIVE_WEIGHTS))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "datasets": self.datasets,
            "seeds": self.seeds,
            "optimizers": self.optimizers,
            "dataset_root": self.dataset_root,
            "protocol_path": self.protocol_path,
            "bootstrap_samples": self.bootstrap_samples,
            "budget": self.budget.to_dict(),
            "objective_weights": self.objective_weights,
        }


def experiment_id_for(config_path: Path, cfg: TuneExperimentConfig) -> str:
    return f"{cfg.experiment_name}-{stable_hash({'path': str(config_path), **cfg.to_dict()}, 10)}"


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
