from __future__ import annotations

import numpy as np


def pure_density(theta: float = 0.0) -> np.ndarray:
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    return np.outer(psi, psi.conj())


def fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ target)).clip(0, 1))
