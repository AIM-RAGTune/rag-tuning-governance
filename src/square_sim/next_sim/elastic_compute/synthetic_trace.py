from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_trace(*, seed: int, rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    periodic = 0.35 + 0.25 * np.sin(t / 24.0)
    burst = rng.binomial(1, 0.08, rows) * rng.uniform(0.25, 0.65, rows)
    demand = np.clip(periodic + burst + rng.normal(0, 0.07, rows), 0.02, 1.0)
    reserved = np.clip(0.42 + rng.normal(0, 0.05, rows), 0.1, 0.9)
    return pd.DataFrame(
        {
            "time_idx": t,
            "demand": demand,
            "reserved_capacity": reserved,
            "queue_pressure": np.clip(demand - reserved + rng.normal(0, 0.03, rows), 0, 1),
            "slo_risk": np.clip((demand - reserved) * 1.7 + burst * 0.7, 0, 1),
            "burst_indicator": burst > 0,
        }
    )

