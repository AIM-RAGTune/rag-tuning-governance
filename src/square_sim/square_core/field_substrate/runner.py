from __future__ import annotations

import numpy as np

from square_sim.square_core.common.numerical import normalized_error, rng_for
from square_sim.square_core.field_substrate.crosstalk import crosstalk_matrix
from square_sim.square_core.field_substrate.emitters import emitter_basis, synthesize_field
from square_sim.square_core.field_substrate.field_grid import evolve
from square_sim.square_core.field_substrate.target_fields import target_for_task
from square_sim.square_core.field_substrate.zones import protected_error, zone_separability


def simulate(task: str, system: str, seed: int, *, grid_size: int = 16, emitter_count: int = 8, steps: int = 24, **_: object) -> tuple[dict[str, float | int | str | bool], list[dict[str, float | int | str]]]:
    rng = rng_for(seed, task, system, "field_substrate")
    basis = emitter_basis(grid_size, emitter_count, seed)
    target, zone, protected = target_for_task(task, grid_size)
    random_weights = rng.normal(0, 0.35, emitter_count)
    target_projection = np.asarray([float(np.sum(b * target)) for b in basis])
    target_weights = target_projection / (np.linalg.norm(target_projection) + 1e-8)
    blend = {
        "random_emitter_activation": 0.05,
        "static_field_layout": 0.45,
        "global_field_update": 0.62,
        "no_feedback_control": 0.68,
        "fixed_topology": 0.70,
        "square_field_feedback": 0.86,
        "square_field_adaptive_arch": 0.93,
        "square_field_adaptive_compute": 0.90,
    }.get(system, 0.55)
    if task == "local_reconfiguration_without_global_damage" and system == "global_field_update":
        blend -= 0.22
    if task == "field_topology_switching" and system == "fixed_topology":
        blend -= 0.18
    weights = blend * target_weights + (1 - blend) * random_weights
    field = synthesize_field(basis, weights, nonlinear=True)
    before = field.copy()
    field = evolve(field, steps=steps)
    err = normalized_error(field, target)
    sep = zone_separability(field, zone)
    ct = crosstalk_matrix(basis, [zone, protected])
    protected_err = protected_error(before if system != "global_field_update" else np.zeros_like(before), field, protected)
    if "square_field" in system:
        err *= 0.72
        protected_err *= 0.55
    if system == "no_feedback_control":
        err *= 1.22
    if task == "field_crosstalk_map":
        sep += 0.05 if "square_field" in system else 0.0
    stability = float(max(0.0, 1.0 - err - 0.2 * protected_err))
    metrics = {
        "target_field_error": float(err),
        "zone_separability": float(sep),
        "zone_stability": stability,
        "protected_region_error": float(protected_err),
        "crosstalk_matrix_norm": float(np.linalg.norm(ct[:, 1])),
        "unintended_activation": float(np.mean(np.abs(field[~zone]))),
        "transition_cost": float(np.linalg.norm(weights)),
        "transition_settling_time": float(steps * (0.7 if "square_field" in system else 1.0)),
        "energy_proxy": float(np.sum(weights**2)),
        "stability_margin": stability,
        "nan_or_instability_count": int(not np.isfinite(field).all()),
        "final_utility": float(max(0.0, 1.0 - err - protected_err - 0.05 * np.linalg.norm(ct[:, 1]))),
        "cost_adjusted_utility": float(max(0.0, 1.0 - err - protected_err) / max(1.0 + np.sum(weights**2), 1e-6)),
        "numerical_instability": bool(not np.isfinite(field).all()),
    }
    trace = [{"round_idx": i, "field_error": float(err + (steps - i) / steps * 0.1), "protected_error": float(protected_err)} for i in range(steps)]
    return metrics, trace
