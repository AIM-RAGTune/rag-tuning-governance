from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRACK_TASKS: dict[str, list[str]] = {
    "adaptive_arch": [
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
    ],
    "field_substrate": [
        "emitter_target_field_reconstruction",
        "overlapping_logic_zone_creation",
        "memory_well_stability",
        "zone_merge_and_split",
        "local_reconfiguration_without_global_damage",
        "field_crosstalk_map",
        "field_topology_switching",
    ],
    "closed_loop": [
        "closed_loop_field_stabilization",
        "adaptive_recovery_after_perturbation",
        "sensor_latency_stress",
        "emitter_failure_reconfiguration",
        "crosstalk_compensation",
        "feedback_vs_open_loop",
        "controller_memory_reuse",
    ],
    "quantum_coupling": [
        "single_qubit_field_modulation",
        "field_defined_memory_qubit",
        "two_qubit_field_coupling",
        "field_defined_reset_zone",
        "coherence_zone_lindblad_model",
        "adversarial_noise_model",
        "continuous_vs_gate_sequence",
    ],
    "soliton": [
        "soliton_formation_threshold",
        "soliton_stability_under_noise",
        "soliton_transport_between_wells",
        "soliton_collision_logic",
        "domain_wall_memory_boundary",
        "linear_wave_vs_soliton",
        "feedback_stabilized_soliton",
    ],
}


TRACK_SYSTEMS: dict[str, list[str]] = {
    "adaptive_arch": [
        "static_policy",
        "greedy_immediate",
        "random_search",
        "coordinate_descent",
        "evolutionary_search",
        "square_adaptive_arch_full",
        "square_adaptive_arch_adaptive_compute",
        "square_adaptive_arch_no_fork",
        "square_adaptive_arch_no_merge",
        "square_adaptive_arch_no_memory",
        "square_adaptive_arch_static_topology",
        "square_adaptive_arch_always_fork",
        "square_adaptive_arch_never_fork",
        "square_adaptive_arch_no_compute_gate",
        "square_adaptive_arch_no_regression_protection",
    ],
    "field_substrate": [
        "static_field_layout",
        "random_emitter_activation",
        "global_field_update",
        "no_feedback_control",
        "fixed_topology",
        "square_field_feedback",
        "square_field_adaptive_arch",
        "square_field_adaptive_compute",
    ],
    "closed_loop": [
        "open_loop_script",
        "pid_like_controller",
        "model_predictive_control",
        "bayesian_optimizer_controller",
        "square_adaptive_controller",
        "square_adaptive_controller_with_memory",
    ],
    "quantum_coupling": [
        "uncontrolled_evolution",
        "static_field_control",
        "gate_sequence_baseline",
        "square_field_control",
        "square_adaptive_field_control",
    ],
    "soliton": [
        "linear_wave_packet",
        "diffusive_field_update",
        "static_potential_well",
        "no_feedback_soliton",
        "random_emitter_pulse",
        "square_feedback_soliton",
    ],
}


@dataclass(frozen=True)
class CoreConfig:
    matrix_name: str
    tracks: list[str]
    seeds: list[int]
    tasks: dict[str, list[str]]
    systems: dict[str, list[str]]
    device: str = "cpu"
    precision: str = "float32"
    grid_size: int = 16
    emitter_count: int = 8
    steps: int = 24
    save_large_tensors: bool = False
    continue_on_failure: bool = True

    @classmethod
    def from_path(cls, path: Path) -> CoreConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        tracks = list(raw.get("tracks", []))
        if not tracks:
            raise ValueError("SQUARE core config requires at least one track.")
        raw_tasks = raw.get("tasks") or {}
        raw_systems = raw.get("systems") or {}
        sim = raw.get("simulation") or {}
        diagnostics = raw.get("diagnostics") or {}
        return cls(
            matrix_name=str(raw.get("matrix_name", path.stem)),
            tracks=tracks,
            seeds=[int(seed) for seed in raw.get("seeds", [101])],
            tasks={track: list(raw_tasks.get(track, TRACK_TASKS[track])) for track in tracks},
            systems={track: list(raw_systems.get(track, TRACK_SYSTEMS[track])) for track in tracks},
            device=str(raw.get("device", "cpu")),
            precision=str(raw.get("precision", "float32")),
            grid_size=int(sim.get("grid_size", 16)),
            emitter_count=int(sim.get("emitter_count", 8)),
            steps=int(sim.get("steps", 24)),
            save_large_tensors=bool(diagnostics.get("save_large_tensors", False)),
            continue_on_failure=bool(raw.get("continue_on_failure", True)),
        )

    def planned_runs(self) -> list[dict[str, Any]]:
        return [
            {"track": track, "task": task, "system": system, "seed": seed}
            for track in self.tracks
            for task in self.tasks[track]
            for seed in self.seeds
            for system in self.systems[track]
        ]
