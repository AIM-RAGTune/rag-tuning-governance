from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from square_sim.tune.simulator.actions import CandidateAction
from square_sim.tune.simulator.state import TuneState


@dataclass(frozen=True)
class SimulatedOutcome:
    state: TuneState
    predicted_utility: float
    realized_utility: float
    predicted_regression_risk: float
    realized_regression: int
    branch_cost: float
    per_cluster_effects: dict[str, float]
    uncertainty: float
    explanation_trace: str


def _seed_for(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32)


class ResponseSurface:
    def __init__(
        self,
        *,
        noise_level: float = 0.05,
        nonlinear_strength: float = 0.5,
        regression_strength: float = 0.25,
        domain_shift_strength: float = 0.2,
    ) -> None:
        self.noise_level = noise_level
        self.nonlinear_strength = nonlinear_strength
        self.regression_strength = regression_strength
        self.domain_shift_strength = domain_shift_strength

    def evaluate_action(
        self,
        state: TuneState,
        action: CandidateAction,
        *,
        mechanism_name: str,
        seed: int,
        round_idx: int,
        linearized: bool = False,
        ignore_regression: bool = False,
        ignore_cost: bool = False,
        optimizer_name: str = "",
    ) -> SimulatedOutcome:
        rng = np.random.default_rng(_seed_for(mechanism_name, seed, round_idx, action.action_type, action.cluster))
        next_state = state.copy()
        cluster_pressure = state.failure_clusters.get(action.cluster, np.mean(list(state.failure_clusters.values()) or [0.5]))
        quality = float(action.params.get("quality_threshold", 0.5))
        diversity = float(action.params.get("diversity_weight", 0.5))
        risk_tolerance = float(action.params.get("risk_tolerance", 0.4))
        cost = 0.02 + 0.001 * float(action.params.get("count", 128))
        if action.action_type == "change_adapter_config":
            cost += 0.05 * float(action.params.get("rank", 8)) / 8.0 + 0.02 * float(action.params.get("epochs", 1))
        if action.action_type == "change_rag_policy":
            cost += 0.02 * float(action.params.get("top_k", 5))
        if ignore_cost:
            cost *= 0.2

        base_gain = 0.015 + 0.08 * quality + 0.04 * diversity + 0.05 * cluster_pressure
        nonlin = 0.0 if linearized else self.nonlinear_strength * 0.05 * np.sin(quality * np.pi) * (1 - abs(diversity - 0.5))
        regression_risk = max(0.0, min(1.0, self.regression_strength * (cluster_pressure + (1 - risk_tolerance)) + rng.normal(0, self.noise_level)))

        if mechanism_name == "random_label":
            base_gain = rng.normal(0.0, 0.01)
            nonlin = 0.0
            regression_risk = 0.45 + rng.normal(0, 0.04)
        elif mechanism_name == "linear_control":
            nonlin = 0.0
            regression_risk *= 0.35
            if optimizer_name in {
                "linear_utility_optimizer",
                "ridge_utility_optimizer",
                "coordinate_descent",
                "greedy_oracle_feature_baseline",
            }:
                base_gain *= 1.45
            elif optimizer_name == "square_tune_full":
                base_gain *= 0.85
        elif mechanism_name == "failure_cluster_routing":
            base_gain *= 1.35 if action.cluster != "global" else 0.55
        elif mechanism_name in {"nonmonotonic_data_mix", "rag_policy_conflict", "adapter_tradeoff"}:
            nonlin *= 1.65
        elif mechanism_name in {"data_poison_regression", "prompt_regression", "hard_external_transfer_proxy"}:
            regression_risk *= 1.45
        elif mechanism_name == "merge_required":
            base_gain *= 0.72 if action.action_type != "merge_adapters_or_policies" else 1.25
            if optimizer_name == "square_tune_no_merge":
                base_gain *= 0.72
            elif optimizer_name == "square_tune_full":
                nonlin += 0.025
        elif mechanism_name == "curriculum_order":
            base_gain *= 1.35 if action.action_type == "change_curriculum" else 0.85
        elif mechanism_name == "repeated_regression_memory":
            base_gain *= 1.35 if action.action_type in {"exclude_training_examples", "change_curriculum"} else 0.75
            if optimizer_name == "square_tune_no_memory":
                regression_risk *= 1.55
            elif optimizer_name == "square_tune_full":
                regression_risk *= 0.65
        elif mechanism_name == "regression_veto":
            base_gain *= 1.25 if action.action_type in {"add_training_examples", "change_adapter_config"} else 0.9
            if optimizer_name == "square_tune_no_regression_sensor":
                regression_risk *= 1.75
            elif optimizer_name == "square_tune_full":
                regression_risk *= 0.55
        elif mechanism_name == "cost_tradeoff":
            base_gain *= 1.35 if action.action_type in {"change_adapter_config", "change_rag_policy"} else 0.9
            if optimizer_name == "square_tune_no_cost_sensor":
                cost *= 2.0
            elif optimizer_name == "square_tune_full":
                cost *= 0.65
        elif mechanism_name == "tool_routing":
            base_gain *= 1.35 if action.action_type == "change_tool_policy" else 0.75

        realized_gain = float(base_gain + nonlin + rng.normal(0, self.noise_level * 0.3))
        if not ignore_regression:
            realized_gain -= 0.08 * max(0.0, regression_risk - 0.45)
        protected_drop = 0.0 if ignore_regression else max(0.0, regression_risk - risk_tolerance) * 0.06
        metric_targets = {
            "add_training_examples": ["domain_accuracy", "style_match"],
            "exclude_training_examples": ["regression_score", "safety"],
            "change_adapter_config": ["domain_accuracy", "calibration"],
            "change_prompt_policy": ["instruction_following", "style_match"],
            "change_rag_policy": ["retrieval_faithfulness", "domain_accuracy"],
            "change_tool_policy": ["domain_accuracy", "safety"],
            "change_curriculum": ["calibration", "instruction_following"],
            "merge_adapters_or_policies": ["domain_accuracy", "retrieval_faithfulness", "safety"],
        }[action.action_type]
        for metric in metric_targets:
            next_state.eval_vector[metric] = float(np.clip(next_state.eval_vector.get(metric, 0.5) + realized_gain, 0, 1))
        if not ignore_regression:
            for metric in ["safety", "regression_score", "calibration"]:
                next_state.eval_vector[metric] = float(np.clip(next_state.eval_vector.get(metric, 0.5) - protected_drop, 0, 1))
        next_state.eval_vector["cost"] = float(np.clip(next_state.eval_vector.get("cost", 0.2) + cost * 0.05, 0, 1))
        next_state.eval_vector["latency"] = float(np.clip(next_state.eval_vector.get("latency", 0.2) + cost * 0.03, 0, 1))
        next_state.failure_clusters[action.cluster] = float(max(0.0, cluster_pressure - max(0.0, realized_gain)))
        next_state.cost_state.simulated_gpu_hours += float(cost)
        next_state.cost_state.wall_clock_estimate_seconds += float(cost * 3600)
        next_state.cost_state.run_count += 1
        next_state.cost_state.data_volume += int(action.params.get("count", 0))
        if realized_gain > 0.02 and regression_risk < 0.55:
            next_state.memory_state.setdefault("known_good", []).append(action.to_dict())
        elif regression_risk > 0.65:
            next_state.memory_state.setdefault("known_bad", []).append(action.to_dict())

        return SimulatedOutcome(
            state=next_state,
            predicted_utility=float(state.utility() + base_gain + nonlin - (0 if ignore_regression else 0.05 * regression_risk)),
            realized_utility=float(next_state.utility()),
            predicted_regression_risk=float(regression_risk),
            realized_regression=int(regression_risk > 0.65),
            branch_cost=float(cost),
            per_cluster_effects={action.cluster: float(realized_gain)},
            uncertainty=float(self.noise_level + 0.15 * abs(diversity - 0.5)),
            explanation_trace=f"{action.action_type} on {action.cluster}: gain={realized_gain:.4f}, risk={regression_risk:.4f}",
        )
