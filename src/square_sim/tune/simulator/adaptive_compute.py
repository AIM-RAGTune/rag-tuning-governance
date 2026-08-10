from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from square_sim.tune.simulator.state import TuneState


@dataclass
class ComputeGateDecision:
    round_idx: int
    selected_region: str
    scenario_family: str
    uncertainty_score: float
    objective_conflict_score: float
    regression_risk_score: float
    hallucination_risk_score: float
    expected_value_of_fork: float
    expected_value_of_merge: float
    budget_remaining: float
    budget_pressure: float
    memory_match_score: float
    decision: str
    reason: str
    estimated_cost: float
    fork_invoked: bool
    merge_invoked: bool
    memory_reused: bool
    regression_repair_invoked: bool
    variant: str = "square_tune_adaptive_compute"
    realized_cost: float = 0.0
    realized_utility_gain: float = 0.0
    realized_regression_delta: float = 0.0
    fork_roi: float = 0.0
    merge_roi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ADAPTIVE_COMPUTE_VARIANTS = {
    "square_tune_adaptive_compute",
    "square_tune_adaptive_compute_no_uncertainty_gate",
    "square_tune_adaptive_compute_no_conflict_gate",
    "square_tune_adaptive_compute_no_roi_gate",
    "square_tune_adaptive_compute_no_budget_gate",
    "square_tune_adaptive_compute_no_regression_escalation",
    "square_tune_adaptive_compute_no_memory_reuse",
    "square_tune_adaptive_compute_always_fork",
    "square_tune_adaptive_compute_never_fork",
}


def is_adaptive_compute(optimizer_name: str) -> bool:
    return optimizer_name in ADAPTIVE_COMPUTE_VARIANTS


def fork_roi(realized_utility_gain: float, additional_cost: float) -> float:
    if additional_cost <= 0:
        return 0.0
    return float(realized_utility_gain / additional_cost)


class ComputeGatePolicy:
    def __init__(self, optimizer_name: str) -> None:
        self.optimizer_name = optimizer_name

    @property
    def no_uncertainty_gate(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_uncertainty_gate"

    @property
    def no_conflict_gate(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_conflict_gate"

    @property
    def no_roi_gate(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_roi_gate"

    @property
    def no_budget_gate(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_budget_gate"

    @property
    def no_regression_escalation(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_regression_escalation"

    @property
    def no_memory_reuse(self) -> bool:
        return self.optimizer_name == "square_tune_adaptive_compute_no_memory_reuse"

    def decide(
        self,
        *,
        state: TuneState,
        round_idx: int,
        selected_region: str,
        scenario_family: str,
        max_rounds: int,
        num_branches: int,
        estimated_branch_cost: float = 0.08,
    ) -> ComputeGateDecision:
        pressures = list(state.failure_clusters.values()) or [0.5]
        region_pressure = float(state.failure_clusters.get(selected_region, np.mean(pressures)))
        uncertainty = float(min(1.0, 0.25 + 0.75 * region_pressure))
        metrics = state.eval_vector
        protected = [metrics.get(key, 0.5) for key in ["retrieval_faithfulness", "safety", "regression_score", "calibration"]]
        task = [metrics.get(key, 0.5) for key in ["domain_accuracy", "instruction_following", "style_match"]]
        objective_conflict = float(min(1.0, max(task or [0.5]) - min(protected or [0.5]) + 0.35 * region_pressure))
        regression_risk = float(min(1.0, 1.0 - metrics.get("regression_score", 0.5) + 0.4 * region_pressure))
        hallucination_risk = float(
            min(
                1.0,
                (1.0 - metrics.get("retrieval_faithfulness", 0.5))
                + (0.30 if "hallucination" in scenario_family or "rag" in scenario_family else 0.0)
                + 0.25 * region_pressure,
            )
        )
        budget_fraction = float(min(1.0, max(0.0, round_idx / max(max_rounds, 1))))
        budget_remaining = float(max(0.0, 1.0 - budget_fraction))
        budget_pressure = budget_fraction
        memory_items = len(state.memory_state.get("known_good", [])) + len(state.memory_state.get("known_bad", []))
        memory_match = float(min(1.0, memory_items / 8.0))
        expected_fork_value = (
            0.12 * uncertainty
            + 0.10 * objective_conflict
            + 0.08 * hallucination_risk
            + 0.06 * regression_risk
            - 0.06 * budget_pressure
            - 0.04 * estimated_branch_cost * max(num_branches, 1)
        )
        if self.no_roi_gate:
            expected_fork_value += 0.10
        expected_merge_value = 0.12 * objective_conflict + 0.05 * hallucination_risk - 0.04 * budget_pressure

        if self.optimizer_name == "square_tune_adaptive_compute_always_fork":
            return self._decision(
                round_idx,
                selected_region,
                scenario_family,
                uncertainty,
                objective_conflict,
                regression_risk,
                hallucination_risk,
                expected_fork_value,
                expected_merge_value,
                budget_remaining,
                budget_pressure,
                memory_match,
                "fork_and_merge",
                "always_fork variant",
                estimated_branch_cost * num_branches,
            )
        if self.optimizer_name == "square_tune_adaptive_compute_never_fork":
            return self._decision(
                round_idx,
                selected_region,
                scenario_family,
                uncertainty,
                objective_conflict,
                regression_risk,
                hallucination_risk,
                expected_fork_value,
                expected_merge_value,
                budget_remaining,
                budget_pressure,
                memory_match,
                "cheap_local_search",
                "never_fork variant",
                estimated_branch_cost,
            )

        if not self.no_memory_reuse and memory_match > 0.55 and uncertainty < 0.62 and budget_pressure > 0.25:
            decision, reason = "memory_reuse", "similar prior intervention and moderate budget pressure"
        elif not self.no_regression_escalation and regression_risk > 0.72:
            decision, reason = "regression_repair", "high regression risk"
        elif (
            not self.no_conflict_gate
            and objective_conflict > 0.60
            and expected_merge_value > 0.04
            and (self.no_budget_gate or budget_pressure < 0.65)
        ):
            decision, reason = "fork_and_merge", "objective conflict justifies merge"
        elif (
            (
                (not self.no_uncertainty_gate and uncertainty > 0.78)
                or (hallucination_risk > (0.96 if self.no_uncertainty_gate else 0.86))
                or (
                    not self.no_uncertainty_gate
                    and
                    not self.no_roi_gate
                    and hallucination_risk > 0.64
                    and expected_fork_value > 0.115
                )
            )
            and (self.no_roi_gate or expected_fork_value > 0.06)
            and (self.no_budget_gate or budget_pressure < 0.58)
        ):
            decision, reason = "multi_branch_fork", "high uncertainty or hallucination risk with positive expected ROI"
        elif uncertainty > 0.52 and (self.no_budget_gate or budget_pressure < 0.70):
            decision, reason = "single_branch_rollout", "medium uncertainty"
        else:
            decision, reason = "cheap_local_search", "low expected value for fork"
        return self._decision(
            round_idx,
            selected_region,
            scenario_family,
            uncertainty,
            objective_conflict,
            regression_risk,
            hallucination_risk,
            expected_fork_value,
            expected_merge_value,
            budget_remaining,
            budget_pressure,
            memory_match,
            decision,
            reason,
            estimated_branch_cost * (num_branches if decision in {"multi_branch_fork", "fork_and_merge"} else 1),
        )

    def _decision(
        self,
        round_idx: int,
        selected_region: str,
        scenario_family: str,
        uncertainty: float,
        objective_conflict: float,
        regression_risk: float,
        hallucination_risk: float,
        expected_fork_value: float,
        expected_merge_value: float,
        budget_remaining: float,
        budget_pressure: float,
        memory_match: float,
        decision: str,
        reason: str,
        estimated_cost: float,
    ) -> ComputeGateDecision:
        return ComputeGateDecision(
            round_idx=round_idx,
            selected_region=selected_region,
            scenario_family=scenario_family,
            uncertainty_score=float(uncertainty),
            objective_conflict_score=float(objective_conflict),
            regression_risk_score=float(regression_risk),
            hallucination_risk_score=float(hallucination_risk),
            expected_value_of_fork=float(expected_fork_value),
            expected_value_of_merge=float(expected_merge_value),
            budget_remaining=float(budget_remaining),
            budget_pressure=float(budget_pressure),
            memory_match_score=float(memory_match),
            decision=decision,
            reason=reason,
            estimated_cost=float(estimated_cost),
            fork_invoked=decision in {"multi_branch_fork", "fork_and_merge"},
            merge_invoked=decision == "fork_and_merge",
            memory_reused=decision == "memory_reuse",
            regression_repair_invoked=decision == "regression_repair",
            variant=self.optimizer_name,
        )


def summarize_compute_gate(decisions: list[dict[str, Any]], *, full_cost: float | None = None, no_fork_cost: float | None = None) -> dict[str, Any]:
    total = max(len(decisions), 1)
    forked = [row for row in decisions if row.get("fork_invoked")]
    merged = [row for row in decisions if row.get("merge_invoked")]
    positive = [row for row in forked if float(row.get("fork_roi", 0.0)) > 0.0]
    harmful = [row for row in forked if float(row.get("realized_utility_gain", 0.0)) < 0.0]
    wasted = [row for row in forked if float(row.get("fork_roi", 0.0)) <= 0.0]
    fork_rate = len(forked) / total
    summary = {
        "total_rounds": len(decisions),
        "cheap_local_rate": sum(row.get("decision") == "cheap_local_search" for row in decisions) / total,
        "single_branch_rate": sum(row.get("decision") == "single_branch_rollout" for row in decisions) / total,
        "multi_branch_rate": sum(row.get("decision") == "multi_branch_fork" for row in decisions) / total,
        "fork_and_merge_rate": sum(row.get("decision") == "fork_and_merge" for row in decisions) / total,
        "memory_reuse_rate": sum(bool(row.get("memory_reused", False)) for row in decisions) / total,
        "regression_repair_rate": sum(bool(row.get("regression_repair_invoked", False)) for row in decisions) / total,
        "fork_invocation_rate": fork_rate,
        "merge_invocation_rate": len(merged) / total,
        "positive_fork_roi_rate": len(positive) / max(len(forked), 1),
        "wasted_fork_rate": len(wasted) / max(len(forked), 1),
        "harmful_fork_rate": len(harmful) / max(len(forked), 1),
        "average_expected_fork_roi": float(np.mean([row.get("expected_value_of_fork", 0.0) for row in decisions])) if decisions else 0.0,
        "average_realized_fork_roi": float(np.mean([row.get("fork_roi", 0.0) for row in forked])) if forked else 0.0,
    }
    flags: list[str] = []
    if fork_rate < 0.01:
        flags.append("behaves_like_no_fork")
    if fork_rate > 0.90:
        flags.append("behaves_like_full")
    if forked and summary["positive_fork_roi_rate"] < 0.50:
        flags.append("fork_not_cost_effective")
    summary["degenerate_behavior_flag"] = ",".join(flags) if flags else ""
    summary["interpretation"] = "adaptive compute uses conditional fork/merge" if not flags else f"diagnostic flags: {summary['degenerate_behavior_flag']}"
    if full_cost is not None:
        summary["cost_saved_vs_full"] = float(full_cost - sum(row.get("realized_cost", 0.0) for row in decisions))
    if no_fork_cost is not None:
        summary["cost_delta_vs_no_fork"] = float(sum(row.get("realized_cost", 0.0) for row in decisions) - no_fork_cost)
    return summary
