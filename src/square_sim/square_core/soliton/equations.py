from __future__ import annotations

import numpy as np


def laplacian_1d(u: np.ndarray) -> np.ndarray:
    return np.roll(u, 1) + np.roll(u, -1) - 2 * u


def nonlinear_step(u: np.ndarray, v: np.ndarray, *, dt: float = 0.04, dx: float = 1.0, model: str = "phi4") -> tuple[np.ndarray, np.ndarray]:
    lap = laplacian_1d(u) / (dx**2)
    force = -u * (u**2 - 1.0) if model == "phi4" else -np.sin(u)
    v2 = 0.995 * v + dt * (lap + force)
    u2 = u + dt * v2
    return u2, v2


def linear_step(u: np.ndarray, v: np.ndarray, *, dt: float = 0.04, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    v2 = 0.998 * v + dt * laplacian_1d(u) / (dx**2)
    return u + dt * v2, v2
