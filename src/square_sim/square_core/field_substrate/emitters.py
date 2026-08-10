from __future__ import annotations

import numpy as np

from square_sim.square_core.common.numerical import gaussian_kernel_grid, rng_for


def emitter_basis(size: int, count: int, seed: int) -> np.ndarray:
    rng = rng_for(seed, "emitters")
    centers = rng.uniform(-0.8, 0.8, size=(count, 2))
    sigmas = rng.uniform(0.12, 0.28, size=count)
    return np.stack([gaussian_kernel_grid(size, tuple(center), float(sigma)) for center, sigma in zip(centers, sigmas, strict=True)])


def synthesize_field(basis: np.ndarray, weights: np.ndarray, nonlinear: bool = True) -> np.ndarray:
    field = np.tensordot(weights, basis, axes=(0, 0))
    return np.tanh(field) if nonlinear else field
