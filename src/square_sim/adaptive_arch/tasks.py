from __future__ import annotations

TASK_CARDS = {
    "local_regime_shift": {
        "tests": "local regime detection and local reconfiguration",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_no_local_reconfiguration", "square_adaptive_arch_static_topology"],
    },
    "future_rollout_required": {
        "tests": "snapshot, conditional forking, and nonlinear future rollout",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_no_fork", "square_adaptive_arch_no_rollout"],
    },
    "merge_required_architecture": {
        "tests": "merge and reintegration of non-conflicting local improvements",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_no_merge"],
    },
    "memory_prevents_repeated_failure": {
        "tests": "architecture-level memory and repeated failure avoidance",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_no_memory"],
    },
    "dynamic_topology_routing": {
        "tests": "dynamic topology and rerouting around local obstruction",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_static_topology"],
    },
    "compute_allocation_trap": {
        "tests": "conditional compute allocation and fork ROI gating",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_always_fork", "square_adaptive_arch_never_fork"],
    },
    "nonlinear_extrapolation_required": {
        "tests": "nonlinear rollout rather than linear extrapolation",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_linear_rollout"],
    },
    "protect_known_good_while_adapting": {
        "tests": "protected region preservation during local adaptation",
        "expected_winner": "square_adaptive_arch_full",
        "failing_ablations": ["square_adaptive_arch_no_regression_protection", "square_adaptive_arch_no_memory"],
    },
    "linear_static_control": {
        "tests": "static linear sanity control",
        "expected_winner": "linear_static_baseline",
        "failing_ablations": [],
    },
    "random_unlearnable_control": {
        "tests": "refusal control with no learnable architecture signal",
        "expected_winner": "none",
        "failing_ablations": [],
    },
}

