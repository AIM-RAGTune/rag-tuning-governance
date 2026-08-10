from __future__ import annotations

import numpy as np


def make_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size)
    return np.meshgrid(axis, axis, indexing="ij")
