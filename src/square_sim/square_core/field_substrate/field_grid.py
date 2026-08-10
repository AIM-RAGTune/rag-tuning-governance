from __future__ import annotations

import numpy as np


def laplacian(field: np.ndarray) -> np.ndarray:
    return (
        np.roll(field, 1, 0)
        + np.roll(field, -1, 0)
        + np.roll(field, 1, 1)
        + np.roll(field, -1, 1)
        - 4.0 * field
    )


def evolve(field: np.ndarray, *, steps: int, diffusion: float = 0.04, damping: float = 0.01) -> np.ndarray:
    out = field.copy()
    for _ in range(steps):
        out = out + diffusion * laplacian(out) - damping * out
    return out
