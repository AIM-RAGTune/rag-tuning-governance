from __future__ import annotations

ELASTIC_COMPUTE_POLICY_SPACE = {
    "scale_up_threshold": [0.55, 0.70, 0.85],
    "scale_down_threshold": [0.25, 0.40],
    "cooldown_period": [2, 5, 10],
    "burst_capacity_threshold": [0.15, 0.35, 0.55],
    "reserved_capacity_ratio": [0.55, 0.70, 0.85],
    "job_priority_weight": [0.2, 0.5, 0.8],
    "queue_delay_threshold": [0.15, 0.30, 0.45],
    "SLO_violation_penalty": [1.0, 2.0],
    "cost_penalty": [0.5, 1.0],
    "prediction_window": [5, 15, 30],
}
