from __future__ import annotations

import numpy as np


def zone_separability(field: np.ndarray, zone: np.ndarray) -> float:
    inside = float(np.mean(field[zone])) if np.any(zone) else 0.0
    outside = float(np.mean(field[~zone])) if np.any(~zone) else 0.0
    return float(max(inside - outside, 0.0))


def protected_error(before: np.ndarray, after: np.ndarray, protected: np.ndarray) -> float:
    if not np.any(protected):
        return 0.0
    return float(np.mean(np.abs(before[protected] - after[protected])))
