from __future__ import annotations

import numpy as np
import pandas as pd


def generate_elastic_compute_trace(rows: int, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    demand = np.clip(rng.gamma(2.2, 0.28, rows), 0, 2)
    reserved = np.clip(rng.normal(0.72, 0.12, rows), 0.2, 1.2)
    burst = np.clip(demand - reserved, 0, 1.2)
    queue = np.clip((demand - reserved) * 0.7 + rng.normal(0, 0.06, rows), 0, 1)
    cost = np.clip(reserved * 0.45 + burst * 0.85 + rng.normal(0, 0.03, rows), 0, 2)
    return pd.DataFrame(
        {
            "row_id": [f"ec-{seed}-{i}" for i in range(rows)],
            "source_dataset": "elastic_compute_synthetic_trace_v1",
            "track": "elastic_compute",
            "workload_demand": demand,
            "reserved_capacity": reserved,
            "burst_capacity": burst,
            "queue_time": queue,
            "latency_proxy": np.clip(queue * 0.8 + rng.normal(0, 0.05, rows), 0, 1),
            "slo_violation": (queue > 0.35).astype(int),
            "cost": cost,
            "wasted_capacity": np.clip(reserved - demand, 0, 1),
            "incident_flag": (rng.random(rows) < np.clip(queue * 0.25, 0, 0.5)).astype(int),
        }
    )
