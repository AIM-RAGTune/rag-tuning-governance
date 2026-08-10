from __future__ import annotations

import numpy as np


def drift_field(shape: tuple[int, int], step: int, strength: float = 0.006) -> np.ndarray:
    axis = np.linspace(-1, 1, shape[0])
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return strength * step * (0.4 * xx + 0.2 * yy)
