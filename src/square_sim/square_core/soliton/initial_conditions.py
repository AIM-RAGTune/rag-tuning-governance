from __future__ import annotations

import numpy as np


def kink(size: int, center: float = 0.0, width: float = 0.12) -> np.ndarray:
    x = np.linspace(-1, 1, size)
    return np.tanh((x - center) / width)


def wave_packet(size: int, center: float = -0.35, width: float = 0.12) -> np.ndarray:
    x = np.linspace(-1, 1, size)
    return np.exp(-((x - center) ** 2) / (2 * width**2))
