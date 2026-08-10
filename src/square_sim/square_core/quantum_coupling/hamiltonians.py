from __future__ import annotations

import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def hamiltonian(field: float, coupling: float = 1.0) -> np.ndarray:
    return 0.5 * SIGMA_Z + coupling * field * SIGMA_X
