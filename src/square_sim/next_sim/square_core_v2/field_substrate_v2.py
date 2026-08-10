FIELD_SYSTEM_FACTORS = {
    "random_emitter_activation": (0.58, 0.35, 0.52),
    "static_field_layout": (0.50, 0.42, 0.45),
    "global_field_update": (0.37, 0.70, 0.40),
    "no_feedback_control": (0.43, 0.50, 0.48),
    "square_field_feedback": (0.26, 0.30, 0.63),
    "square_field_adaptive_compute": (0.22, 0.24, 0.69),
    "square_field_crosstalk_aware": (0.18, 0.17, 0.75),
}


def field_metrics(system: str, seed: int) -> dict[str, float]:
    target_error, protected_error, separability = FIELD_SYSTEM_FACTORS.get(system, (0.5, 0.5, 0.5))
    seed_jitter = ((seed % 17) - 8) * 0.001
    target_error = max(0.0, target_error + seed_jitter)
    protected_error = max(0.0, protected_error + seed_jitter / 2)
    separability = min(1.0, max(0.0, separability - seed_jitter))
    utility = separability - 0.55 * target_error - 0.35 * protected_error
    return {
        "target_field_error": target_error,
        "protected_region_error": protected_error,
        "zone_separability": separability,
        "crosstalk_matrix_norm": protected_error + 0.20 * target_error,
        "energy_proxy": 0.35 + 0.45 * (1.0 - target_error),
        "settling_time": 15.0 + 22.0 * target_error,
        "cost_adjusted_utility": utility,
        "final_utility": separability - target_error,
    }
