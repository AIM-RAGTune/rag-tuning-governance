from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRACKS = [
    "rag_hard_subset_v1",
    "no_fork_robustness_v1",
    "adaptive_escalation_v2",
    "claim_level_faithfulness_v1",
    "elastic_compute_policy_v1",
    "square_core_v2_field_substrate_targeted",
    "square_core_v2_closed_loop_targeted",
]

TRACK_SYSTEMS = {
    "rag_hard_subset_v1": [
        "square_tune_no_fork",
        "square_tune_adaptive_compute",
        "square_tune_full",
        "random_gating_matched_cost",
        "uncertainty_threshold_gating_matched_cost",
        "retrieval_confidence_gating_matched_cost",
        "entropy_or_margin_gating_matched_cost",
        "best_single_policy_on_validation",
        "greedy_regression_aware_search",
        "optuna_tpe_matched_budget_optional",
    ],
    "no_fork_robustness_v1": [
        "static_default_rag_policy",
        "best_single_policy_on_validation",
        "greedy_regression_aware_search",
        "coordinate_descent_matched_budget",
        "optuna_tpe_matched_budget_optional",
        "bayesian_optimization_matched_budget_optional",
        "evolutionary_search_matched_budget",
        "square_tune_no_fork",
        "square_tune_adaptive_compute",
        "square_tune_full",
        "oracle_upper_bound_diagnostic",
    ],
    "adaptive_escalation_v2": [
        "square_tune_no_fork_default",
        "square_tune_hard_subset_escalation",
        "square_tune_claim_risk_escalation",
        "square_tune_retrieval_conflict_escalation",
        "square_tune_budget_guarded_escalation",
        "square_tune_three_tier_escalation",
        "square_tune_full",
        "square_tune_adaptive_compute",
        "square_tune_no_fork",
    ],
    "claim_level_faithfulness_v1": [
        "static_claim_policy",
        "no_fork",
        "adaptive_compute",
        "claim_risk_escalation",
        "retrieval_confidence_gating",
        "uncertainty_gating",
        "random_matched_cost_gating",
        "full",
    ],
    "elastic_compute_policy_v1": [
        "static_threshold_policy",
        "greedy_policy",
        "random_search",
        "coordinate_descent",
        "optuna_tpe_optional",
        "bayesian_optimizer_optional",
        "square_tune_no_fork",
        "square_tune_adaptive_compute",
        "square_adaptive_arch_adaptive_compute",
    ],
    "square_core_v2_field_substrate_targeted": [
        "random_emitter_activation",
        "static_field_layout",
        "global_field_update",
        "no_feedback_control",
        "square_field_feedback",
        "square_field_adaptive_compute",
        "square_field_crosstalk_aware",
    ],
    "square_core_v2_closed_loop_targeted": [
        "open_loop_script",
        "pid_like_controller",
        "model_predictive_control",
        "bayesian_optimizer_controller",
        "square_adaptive_controller",
        "square_adaptive_controller_with_memory",
        "square_adaptive_controller_with_topology",
    ],
}


@dataclass(frozen=True)
class NextSimConfig:
    matrix_name: str
    tracks: list[str]
    seeds: list[int]
    systems_by_track: dict[str, list[str]]
    smoke: bool = False
    continue_on_failure: bool = True
    real_rag_required: bool = False
    bootstrap_samples: int = 500

    @classmethod
    def from_path(cls, path: Path) -> NextSimConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tracks = [str(track) for track in raw.get("tracks", TRACKS)]
        systems_by_track = {
            track: [str(system) for system in (raw.get("systems_by_track", {}).get(track) or TRACK_SYSTEMS[track])]
            for track in tracks
        }
        return cls(
            matrix_name=str(raw.get("matrix_name", path.stem)),
            tracks=tracks,
            seeds=[int(seed) for seed in raw.get("seeds", [101])],
            systems_by_track=systems_by_track,
            smoke=bool(raw.get("smoke", False)),
            continue_on_failure=bool(raw.get("continue_on_failure", True)),
            real_rag_required=bool(raw.get("real_rag_required", False)),
            bootstrap_samples=int(raw.get("bootstrap_samples", 500)),
        )

    def planned_runs(self) -> list[dict[str, Any]]:
        return [
            {"track": track, "system": system, "seed": seed}
            for track in self.tracks
            for seed in self.seeds
            for system in self.systems_by_track[track]
        ]

