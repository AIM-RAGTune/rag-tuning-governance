from __future__ import annotations

import numpy as np


def controller_gain(system: str) -> tuple[float, float]:
    if system == "open_loop_script":
        return 0.0, 0.0
    if system == "pid_like_controller":
        return 0.35, 0.05
    if system == "model_predictive_control":
        return 0.50, 0.08
    if system == "bayesian_optimizer_controller":
        return 0.42, 0.06
    if system == "square_adaptive_controller":
        return 0.62, 0.11
    if system == "square_adaptive_controller_with_memory":
        return 0.68, 0.16
    return 0.2, 0.03


def control_update(error: np.ndarray, integral: np.ndarray, system: str) -> np.ndarray:
    kp, ki = controller_gain(system)
    return kp * error + ki * integral
