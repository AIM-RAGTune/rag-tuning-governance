from __future__ import annotations

import numpy as np

from square_sim.square_core.common.numerical import rng_for


def disturbance(task: str, shape: tuple[int, int], step: int, seed: int) -> np.ndarray:
    rng = rng_for(seed, task, step, "disturbance")
    noise = rng.normal(0, 0.015, shape)
    if task == "adaptive_recovery_after_perturbation" and step == max(2, shape[0] // 3):
        noise[shape[0] // 3 : 2 * shape[0] // 3, shape[1] // 3 : 2 * shape[1] // 3] += 0.45
    if task == "emitter_failure_reconfiguration" and step > shape[0] // 2:
        noise[:, : shape[1] // 4] -= 0.08
    return noise
