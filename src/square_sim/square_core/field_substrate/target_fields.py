from __future__ import annotations

import numpy as np

from square_sim.square_core.common.numerical import gaussian_kernel_grid


def target_for_task(task: str, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = gaussian_kernel_grid(size, (-0.45, -0.1), 0.22)
    right = gaussian_kernel_grid(size, (0.45, 0.1), 0.22)
    top = gaussian_kernel_grid(size, (0.0, 0.55), 0.18)
    protected = gaussian_kernel_grid(size, (-0.65, 0.65), 0.18) > 0.35
    if task == "zone_merge_and_split":
        target = np.maximum(left, right)
    elif task == "field_topology_switching":
        target = np.maximum(right, top)
    elif task == "memory_well_stability":
        target = left
    else:
        target = 0.65 * left + 0.55 * right
    zone = target > 0.4
    return target.astype(float), zone.astype(bool), protected.astype(bool)
