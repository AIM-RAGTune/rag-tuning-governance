from __future__ import annotations

NON_SQUARE_BASELINES = {
    "static_default_policy",
    "random_search",
    "greedy_eval_improvement",
    "greedy_regression_aware",
    "coordinate_descent",
    "evolutionary_search",
    "optuna_tpe_optional",
    "bayesian_optimizer_optional",
    "classical_only_baseline",
    "threshold_policy_baseline",
}

SQUARE_VARIANTS = {
    "square_tune_full",
    "square_tune_no_fork",
    "square_tune_no_merge",
    "square_tune_no_snapshot",
    "square_tune_no_cost_sensor",
    "square_tune_no_regression_sensor",
    "square_tune_adaptive_compute",
    "square_adaptive_arch_full",
    "square_adaptive_arch_adaptive_compute",
    "square_adaptive_arch_no_fork",
    "square_adaptive_arch_static_topology",
}


def is_square_variant(system: str) -> bool:
    return system.startswith("square_")
