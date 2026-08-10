from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from square_sim.tune.config import TuneBudget


@dataclass
class BudgetLedger:
    max_response_surface_evaluations: int
    max_candidate_actions: int
    max_simulated_gpu_hours: float
    max_rounds: int
    max_branches: int
    token_cost_proxy_budget: float = 1_000.0
    response_surface_evaluations: int = 0
    candidate_actions_scored: int = 0
    simulated_gpu_hours: float = 0.0
    adaptation_rounds: int = 0
    branch_rollouts: int = 0
    token_cost_proxy: float = 0.0
    data_examples_selected: int = 0
    wall_time_proxy: float = 0.0
    exhausted_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_budget(cls, budget: TuneBudget) -> BudgetLedger:
        return cls(
            max_response_surface_evaluations=budget.max_response_surface_evaluations,
            max_candidate_actions=budget.max_candidate_actions,
            max_simulated_gpu_hours=budget.simulated_gpu_hour_budget,
            max_rounds=budget.max_rounds,
            max_branches=budget.num_branches,
            token_cost_proxy_budget=budget.token_cost_proxy_budget,
        )

    def can_consume(self, *, evaluations: int = 1, candidates: int = 1, gpu_hours: float = 0.0) -> bool:
        return (
            self.response_surface_evaluations + evaluations <= self.max_response_surface_evaluations
            and self.candidate_actions_scored + candidates <= self.max_candidate_actions
            and self.simulated_gpu_hours + gpu_hours <= self.max_simulated_gpu_hours
        )

    def consume(
        self,
        *,
        evaluations: int = 1,
        candidates: int = 1,
        gpu_hours: float = 0.0,
        branch_rollouts: int = 1,
        token_cost: float = 0.0,
        data_examples: int = 0,
    ) -> bool:
        if not self.can_consume(evaluations=evaluations, candidates=candidates, gpu_hours=gpu_hours):
            if self.response_surface_evaluations + evaluations > self.max_response_surface_evaluations:
                self.exhausted_flags.append("response_surface_evaluations")
            if self.candidate_actions_scored + candidates > self.max_candidate_actions:
                self.exhausted_flags.append("candidate_actions_scored")
            if self.simulated_gpu_hours + gpu_hours > self.max_simulated_gpu_hours:
                self.exhausted_flags.append("simulated_gpu_hours")
            self.exhausted_flags = sorted(set(self.exhausted_flags))
            return False
        self.response_surface_evaluations += evaluations
        self.candidate_actions_scored += candidates
        self.simulated_gpu_hours += gpu_hours
        self.branch_rollouts += branch_rollouts
        self.token_cost_proxy += token_cost
        self.data_examples_selected += data_examples
        self.wall_time_proxy += gpu_hours * 3600.0
        return True

    def start_round(self) -> bool:
        if self.adaptation_rounds >= self.max_rounds:
            self.exhausted_flags.append("adaptation_rounds")
            self.exhausted_flags = sorted(set(self.exhausted_flags))
            return False
        self.adaptation_rounds += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_response_surface_evaluations": self.max_response_surface_evaluations,
            "max_candidate_actions": self.max_candidate_actions,
            "max_simulated_gpu_hours": self.max_simulated_gpu_hours,
            "max_rounds": self.max_rounds,
            "max_branches": self.max_branches,
            "token_cost_proxy_budget": self.token_cost_proxy_budget,
            "response_surface_evaluations": self.response_surface_evaluations,
            "candidate_actions_scored": self.candidate_actions_scored,
            "simulated_gpu_hours": self.simulated_gpu_hours,
            "adaptation_rounds": self.adaptation_rounds,
            "branch_rollouts": self.branch_rollouts,
            "token_cost_proxy": self.token_cost_proxy,
            "data_examples_selected": self.data_examples_selected,
            "wall_time_proxy": self.wall_time_proxy,
            "exhausted_flags": list(self.exhausted_flags),
        }

