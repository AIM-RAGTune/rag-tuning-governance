from __future__ import annotations

from typing import Any

import pandas as pd

CAUTION = (
    "This SQUARE Core Validation Certificate is based on software simulation only. "
    "It does not validate physical SQUARE hardware or quantum advantage."
)


COMPONENTS = [
    "adaptive_architecture",
    "local_reconfiguration",
    "snapshotting",
    "conditional_forking",
    "nonlinear_rollout",
    "merge_reintegration",
    "architecture_memory",
    "dynamic_topology",
    "adaptive_compute_allocation",
    "field_controllability",
    "bounded_crosstalk",
    "field_memory_well",
    "protected_region_control",
    "closed_loop_feedback",
    "perturbation_recovery",
    "sensor_latency_tolerance",
    "emitter_failure_recovery",
    "quantum_state_coupling",
    "coherence_zone_candidate",
    "field_defined_reset",
    "soliton_formation",
    "soliton_transport",
    "domain_wall_boundary",
    "feedback_stabilized_soliton",
]


def _mean(group: pd.DataFrame, system: str, metric: str = "cost_adjusted_utility") -> float | None:
    rows = group[group["system"] == system]
    return None if rows.empty else float(rows[metric].mean())


def certificate_for_group(track: str, task: str, group: pd.DataFrame) -> dict[str, Any]:
    supported = {component: "inconclusive" for component in COMPONENTS}
    if bool(group.get("numerical_instability", pd.Series([False])).any()):
        status, reason = "Numerical instability", "One or more runs produced non-finite numerical output."
    elif "random" in task or "unlearnable" in task:
        status, reason = "Refused", "Random/unlearnable control does not support a SQUARE core claim."
    elif "linear_static_control" in task:
        static = max([v for v in [_mean(group, "static_policy"), _mean(group, "linear_static_baseline"), _mean(group, "greedy_immediate")] if v is not None] or [0.0])
        full = _mean(group, "square_adaptive_arch_full") or 0.0
        if static >= full - 0.01:
            status, reason = "Refused", "Static/linear control was won or tied by a simple/static system."
        else:
            status, reason = "Control failed", "Static/linear control was not won by a simple/static system."
    else:
        best = group.groupby("system")["cost_adjusted_utility"].mean().sort_values(ascending=False)
        best_system = str(best.index[0]) if not best.empty else ""
        best_score = float(best.iloc[0]) if not best.empty else 0.0
        baseline = float(group[~group["system"].str.contains("square", case=False, na=False)]["cost_adjusted_utility"].max() or 0.0)
        status = "Candidate signal" if best_score > baseline + 0.01 else "Inconclusive"
        reason = "Best SQUARE-family system beat non-SQUARE baselines." if status == "Candidate signal" else "SQUARE-family systems did not clearly beat baselines."
        if best_system:
            supported["adaptive_architecture"] = "supported" if track == "adaptive_arch" and "square" in best_system else supported["adaptive_architecture"]
        component_by_task = {
            "local_regime_shift": "local_reconfiguration",
            "future_rollout_required": "conditional_forking",
            "merge_required_architecture": "merge_reintegration",
            "memory_prevents_repeated_failure": "architecture_memory",
            "dynamic_topology_routing": "dynamic_topology",
            "compute_allocation_trap": "adaptive_compute_allocation",
            "nonlinear_extrapolation_required": "nonlinear_rollout",
            "protect_known_good_while_adapting": "protected_region_control",
            "emitter_target_field_reconstruction": "field_controllability",
            "field_crosstalk_map": "bounded_crosstalk",
            "memory_well_stability": "field_memory_well",
            "local_reconfiguration_without_global_damage": "protected_region_control",
            "closed_loop_field_stabilization": "closed_loop_feedback",
            "adaptive_recovery_after_perturbation": "perturbation_recovery",
            "sensor_latency_stress": "sensor_latency_tolerance",
            "emitter_failure_reconfiguration": "emitter_failure_recovery",
            "single_qubit_field_modulation": "quantum_state_coupling",
            "coherence_zone_lindblad_model": "coherence_zone_candidate",
            "field_defined_reset_zone": "field_defined_reset",
            "soliton_formation_threshold": "soliton_formation",
            "soliton_transport_between_wells": "soliton_transport",
            "domain_wall_memory_boundary": "domain_wall_boundary",
            "feedback_stabilized_soliton": "feedback_stabilized_soliton",
        }
        component = component_by_task.get(task)
        if component and status == "Candidate signal":
            supported[component] = "supported"
        if task in {"future_rollout_required", "compute_allocation_trap"} and status == "Candidate signal":
            supported["snapshotting"] = "supported"
        if task in {"future_rollout_required", "merge_required_architecture", "compute_allocation_trap"} and status == "Candidate signal":
            supported["conditional_forking"] = "supported"
        if task == "protect_known_good_while_adapting" and status == "Candidate signal":
            supported["regression_protection"] = "supported"
        if track == "quantum_coupling":
            reason += " Quantum-coupling result is toy-model only."
    return {
        "track": track,
        "task": task,
        "certificate_type": "SQUARE Core Validation Certificate",
        "status": status,
        "supported_components": supported,
        "evidence": {
            "systems": sorted(group["system"].unique().tolist()) if not group.empty else [],
            "mean_cost_adjusted_utility": group.groupby("system")["cost_adjusted_utility"].mean().to_dict() if not group.empty else {},
        },
        "caveats": [CAUTION],
    }
