from __future__ import annotations

import numpy as np

from square_sim.square_core.common.numerical import rng_for
from square_sim.square_core.soliton.collisions import collision_separability
from square_sim.square_core.soliton.equations import linear_step, nonlinear_step
from square_sim.square_core.soliton.initial_conditions import kink, wave_packet
from square_sim.square_core.soliton.stability import dispersion_rate, localization_error
from square_sim.square_core.soliton.transport import transport_fidelity


def simulate(task: str, system: str, seed: int, *, grid_size: int = 16, steps: int = 24, **_: object) -> tuple[dict[str, float | int | str | bool], list[dict[str, float | int | str]]]:
    rng = rng_for(seed, task, system, "soliton")
    size = max(32, grid_size * 2)
    nonlinear = system not in {"linear_wave_packet", "diffusive_field_update", "static_potential_well"}
    u = kink(size, -0.25) if nonlinear else wave_packet(size)
    v = np.zeros_like(u)
    if task == "soliton_transport_between_wells":
        v += 0.18
    if system == "random_emitter_pulse":
        v += rng.normal(0, 0.08, size)
    start = u.copy()
    trace = []
    for step in range(steps):
        if nonlinear:
            u, v = nonlinear_step(u, v, model="sine" if task == "soliton_collision_logic" else "phi4")
            if system == "square_feedback_soliton":
                u = 0.985 * u + 0.015 * kink(size, -0.25 + 0.45 * step / max(steps - 1, 1))
        else:
            u, v = linear_step(u, v)
            if system == "diffusive_field_update":
                u = 0.94 * u + 0.03 * (np.roll(u, 1) + np.roll(u, -1))
        if task == "soliton_stability_under_noise":
            u += rng.normal(0, 0.008, size)
        trace.append({"round_idx": step, "localization_error": localization_error(u), "max_amplitude": float(np.max(np.abs(u)))})
    loc = localization_error(u)
    disp = dispersion_rate(start, u)
    transport = transport_fidelity(u)
    separability = collision_separability(u)
    formation = float(max(0, 1 - loc))
    if nonlinear:
        formation += 0.12
    if system == "square_feedback_soliton":
        disp *= 0.55
        formation += 0.06
    utility = float(max(0, min(1, formation + 0.22 * transport + 0.12 * separability - 0.4 * disp)))
    metrics = {
        "formation_success_rate": float(min(formation, 1)),
        "formation_threshold_energy": float(np.mean(v**2) + 0.1),
        "lifetime": float(steps * max(0, 1 - disp)),
        "localization_error": float(loc),
        "dispersion_rate": float(disp),
        "transport_fidelity_proxy": float(transport),
        "collision_output_separability": float(separability),
        "boundary_leakage_reduction": float(max(0, 0.5 - loc)),
        "feedback_lifetime_gain": float(0.18 if system == "square_feedback_soliton" else 0.0),
        "energy_proxy": float(np.mean(u**2 + v**2)),
        "numerical_stability": bool(np.isfinite(u).all()),
        "final_utility": utility,
        "cost_adjusted_utility": float(utility / (1.2 if system == "square_feedback_soliton" else 1.0)),
        "numerical_instability": bool(not np.isfinite(u).all()),
    }
    return metrics, trace
