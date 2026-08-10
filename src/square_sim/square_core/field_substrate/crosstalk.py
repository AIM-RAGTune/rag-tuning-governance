from __future__ import annotations

import numpy as np


def crosstalk_matrix(basis: np.ndarray, zones: list[np.ndarray]) -> np.ndarray:
    rows = []
    for emitter in basis:
        total = float(np.abs(emitter).sum() + 1e-8)
        rows.append([float(np.abs(emitter[zone]).sum() / total) for zone in zones])
    return np.asarray(rows)
