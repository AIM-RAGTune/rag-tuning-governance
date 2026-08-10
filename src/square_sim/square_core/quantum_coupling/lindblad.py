from __future__ import annotations

import numpy as np


def lindblad_dephase(rho: np.ndarray, gamma: float, dt: float) -> np.ndarray:
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    return rho + dt * gamma * (z @ rho @ z - rho)
