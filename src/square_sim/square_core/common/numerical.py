from __future__ import annotations

import math
from typing import Any

import numpy as np


def rng_for(seed: int, *parts: Any) -> np.random.Generator:
    offset = abs(hash((int(seed),) + tuple(parts))) % (2**32)
    return np.random.default_rng(offset)


def finite_metrics(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float, np.floating)) and not math.isfinite(float(value)):
            return False
    return True


def gaussian_kernel_grid(size: int, center: tuple[float, float], sigma: float) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(2.0 * sigma**2, 1e-6)))


def normalized_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b) + 1e-8)
    return float(np.linalg.norm(a - b) / denom)


def stable_sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -40, 40))))
