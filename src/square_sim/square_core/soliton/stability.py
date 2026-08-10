from __future__ import annotations

import numpy as np


def localization_error(u: np.ndarray) -> float:
    mass = np.abs(u - np.mean(u))
    if mass.sum() <= 1e-8:
        return 1.0
    x = np.linspace(-1, 1, len(u))
    center = float(np.sum(x * mass) / np.sum(mass))
    spread = float(np.sqrt(np.sum(((x - center) ** 2) * mass) / np.sum(mass)))
    return spread


def dispersion_rate(start: np.ndarray, end: np.ndarray) -> float:
    return float(max(0, localization_error(end) - localization_error(start)))
