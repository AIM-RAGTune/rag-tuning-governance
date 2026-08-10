from __future__ import annotations

import numpy as np


def collision_separability(u: np.ndarray) -> float:
    mid = len(u) // 2
    return float(abs(np.mean(u[:mid]) - np.mean(u[mid:])))
