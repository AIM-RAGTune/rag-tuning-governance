from __future__ import annotations

import numpy as np


def transport_fidelity(u: np.ndarray, target_center: float = 0.35) -> float:
    x = np.linspace(-1, 1, len(u))
    mass = np.abs(u - np.mean(u))
    center = float(np.sum(x * mass) / max(np.sum(mass), 1e-8))
    return float(max(0, 1 - abs(center - target_center)))
