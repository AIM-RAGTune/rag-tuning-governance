from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SYSTEMS = [
    "static_default_rag_policy",
    "best_single_policy_on_validation",
    "square_tune_full",
    "square_tune_no_fork",
    "square_tune_no_merge",
    "square_tune_no_cost_sensor",
    "square_tune_no_regression_sensor",
    "square_tune_adaptive_compute",
    "random_gating_matched_cost",
    "uncertainty_threshold_gating_matched_cost",
    "retrieval_confidence_gating_matched_cost",
    "entropy_or_margin_gating_matched_cost",
    "optuna_tpe_matched_budget_optional",
    "bayesian_optimization_matched_budget_optional",
    "greedy_regression_aware_search",
    "coordinate_descent_matched_budget",
    "evolutionary_search_matched_budget",
    "oracle_upper_bound_diagnostic",
]

UTILITY_WEIGHTS = {
    "default": {"quality": 1.0, "cost": 0.25, "latency": 0.10, "regression": 0.50},
    "quality_heavy": {"quality": 1.0, "cost": 0.10, "latency": 0.05, "regression": 0.50},
    "cost_heavy": {"quality": 1.0, "cost": 0.50, "latency": 0.20, "regression": 0.50},
    "regression_heavy": {"quality": 1.0, "cost": 0.25, "latency": 0.10, "regression": 1.00},
    "latency_heavy": {"quality": 1.0, "cost": 0.25, "latency": 0.50, "regression": 0.50},
    "raw_quality_only": {"quality": 1.0, "cost": 0.0, "latency": 0.0, "regression": 0.0},
}


@dataclass(frozen=True)
class MatchedCostRAGConfig:
    matrix_name: str
    seeds: list[int]
    systems: list[str]
    max_rows: int | None = None
    bootstrap_samples: int = 1000
    matched_cost_tolerance_pct: float = 2.5
    real_data_required: bool = True
    continue_on_failure: bool = True

    @classmethod
    def from_path(cls, path: Path) -> MatchedCostRAGConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        simulation = raw.get("simulation") or {}
        max_rows = simulation.get("max_rows", raw.get("max_rows"))
        return cls(
            matrix_name=str(raw.get("matrix_name", path.stem)),
            seeds=[int(seed) for seed in raw.get("seeds", [101])],
            systems=[str(system) for system in raw.get("systems", SYSTEMS)],
            max_rows=None if max_rows in {None, "null"} else int(max_rows),
            bootstrap_samples=int(simulation.get("bootstrap_samples", raw.get("bootstrap_samples", 1000))),
            matched_cost_tolerance_pct=float(
                simulation.get("matched_cost_tolerance_pct", raw.get("matched_cost_tolerance_pct", 2.5))
            ),
            real_data_required=bool(raw.get("real_data_required", True)),
            continue_on_failure=bool(raw.get("continue_on_failure", True)),
        )

    def planned_runs(self) -> list[dict[str, Any]]:
        return [
            {"scenario": "real_rag_policy_matched_cost", "system": system, "seed": seed}
            for seed in self.seeds
            for system in self.systems
        ]

