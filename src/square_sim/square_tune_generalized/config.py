from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRACK_SCENARIOS: dict[str, list[str]] = {
    "rag": [
        "rag_policy_optimization",
        "claim_level_faithfulness",
        "source_selection_policy",
        "abstention_threshold_policy",
        "retrieval_cost_tradeoff",
    ],
    "patient_flow": [
        "ed_boarding_risk_explanation",
        "admission_risk_explanation",
        "patient_flow_bottleneck_summary",
        "resource_escalation_recommendation",
        "bed_demand_scenario_explanation",
        "triage_acuity_context_retrieval",
    ],
    "elastic_compute": [
        "autoscaling_threshold_policy",
        "scale_up_scale_down_decisioning",
        "gpu_job_scheduling_proxy",
        "reserved_vs_burst_capacity_policy",
        "batch_vs_online_workload_placement",
        "cost_slo_tradeoff_optimization",
        "incident_aware_capacity_recommendation",
    ],
    "ml_to_llm": [
        "prediction_only_baseline",
        "prediction_plus_explanation",
        "prediction_plus_evidence_retrieval",
        "prediction_plus_action_recommendation",
        "prediction_plus_exception_handling",
        "policy_interpretation_augmented_prediction",
    ],
}


TRACK_DATASET_KEYS: dict[str, str] = {
    "rag": "rag_proxy",
    "patient_flow": "patient_flow_synthetic_proxy_v1",
    "elastic_compute": "elastic_compute_synthetic_trace_v1",
    "ml_to_llm": "ml_to_llm_hybrid_proxy_v1",
}


SYSTEMS: list[str] = [
    "static_default_policy",
    "random_search",
    "greedy_eval_improvement",
    "greedy_regression_aware",
    "coordinate_descent",
    "evolutionary_search",
    "optuna_tpe_optional",
    "bayesian_optimizer_optional",
    "classical_only_baseline",
    "threshold_policy_baseline",
    "square_tune_full",
    "square_tune_no_fork",
    "square_tune_no_merge",
    "square_tune_no_snapshot",
    "square_tune_no_cost_sensor",
    "square_tune_no_regression_sensor",
    "square_tune_adaptive_compute",
    "square_adaptive_arch_full",
    "square_adaptive_arch_adaptive_compute",
    "square_adaptive_arch_no_fork",
    "square_adaptive_arch_static_topology",
]


@dataclass(frozen=True)
class GeneralizedConfig:
    matrix_name: str
    tracks: list[str]
    seeds: list[int]
    scenarios: dict[str, list[str]]
    systems: list[str]
    stress_profiles: list[str] | None = None
    rows_per_track: int = 2000
    scenario_max_rows: int = 2000
    device: str = "cpu"
    continue_on_failure: bool = True
    publication_mode: bool = False

    @classmethod
    def from_path(cls, path: Path) -> GeneralizedConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tracks = [str(track) for track in raw.get("tracks", [])]
        if not tracks:
            raise ValueError("Generalized benchmark config requires at least one track.")
        raw_scenarios = raw.get("scenarios") or {}
        simulation = raw.get("simulation") or {}
        stress_profiles = raw.get("stress_profiles", simulation.get("stress_profiles", ["nominal"]))
        return cls(
            matrix_name=str(raw.get("matrix_name", path.stem)),
            tracks=tracks,
            seeds=[int(seed) for seed in raw.get("seeds", [101])],
            scenarios={track: list(raw_scenarios.get(track, TRACK_SCENARIOS[track])) for track in tracks},
            systems=[str(system) for system in raw.get("systems", SYSTEMS)],
            stress_profiles=[str(profile) for profile in stress_profiles],
            rows_per_track=int(simulation.get("rows_per_track", raw.get("rows_per_track", 2000))),
            scenario_max_rows=int(simulation.get("scenario_max_rows", raw.get("scenario_max_rows", 2000))),
            device=str(raw.get("device", "cpu")),
            continue_on_failure=bool(raw.get("continue_on_failure", True)),
            publication_mode=bool(raw.get("publication_mode", False)),
        )

    def planned_runs(self) -> list[dict[str, Any]]:
        profiles = self.stress_profiles or ["nominal"]
        return [
            {"track": track, "scenario": scenario, "system": system, "seed": seed, "stress_profile": profile}
            for track in self.tracks
            for scenario in self.scenarios[track]
            for seed in self.seeds
            for profile in profiles
            for system in self.systems
        ]


def load_dataset_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
