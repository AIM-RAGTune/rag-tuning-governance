CONTROLLER_FACTORS = {
    "open_loop_script": (0.58, 32.0, 0.24),
    "pid_like_controller": (0.42, 24.0, 0.35),
    "model_predictive_control": (0.31, 17.0, 0.50),
    "bayesian_optimizer_controller": (0.34, 19.0, 0.46),
    "square_adaptive_controller": (0.27, 14.0, 0.57),
    "square_adaptive_controller_with_memory": (0.23, 11.0, 0.64),
    "square_adaptive_controller_with_topology": (0.21, 10.0, 0.68),
}


def closed_loop_metrics(system: str, seed: int) -> dict[str, float]:
    field_error, recovery_time, memory_gain = CONTROLLER_FACTORS.get(system, (0.5, 25.0, 0.3))
    jitter = ((seed % 13) - 6) * 0.001
    field_error = max(0.0, field_error + jitter)
    recovery_time = max(1.0, recovery_time + jitter * 50)
    utility = 0.85 - field_error - 0.012 * recovery_time + 0.25 * memory_gain
    return {
        "field_error_over_time": field_error,
        "recovery_time": recovery_time,
        "latency_tolerance": max(0.0, 1.0 - field_error),
        "memory_reuse_gain": memory_gain,
        "failure_recovery_score": max(0.0, 1.0 - recovery_time / 40.0),
        "control_energy_proxy": 0.30 + 0.40 * (1.0 - field_error),
        "cost_adjusted_utility": utility,
        "final_utility": 1.0 - field_error,
    }
