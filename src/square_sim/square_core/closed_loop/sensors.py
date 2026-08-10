from __future__ import annotations

import numpy as np


def sense(field: np.ndarray, target: np.ndarray, latency_buffer: list[np.ndarray]) -> np.ndarray:
    error = target - field
    latency_buffer.append(error)
    return latency_buffer.pop(0) if latency_buffer else error
