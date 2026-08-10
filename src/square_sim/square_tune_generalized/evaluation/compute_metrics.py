from __future__ import annotations

import pandas as pd


def elastic_compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "SLO_violation_rate": float(frame.get("slo_violation", pd.Series([0])).mean()),
        "cost": float(frame.get("cost", pd.Series([0.5])).mean()),
        "wasted_capacity": float(frame.get("wasted_capacity", pd.Series([0.2])).mean()),
        "utilization": float((frame.get("workload_demand", pd.Series([0.5])) / (frame.get("reserved_capacity", pd.Series([1.0])) + 1e-6)).clip(0, 1.5).mean()),
        "queue_time": float(frame.get("queue_time", pd.Series([0.2])).mean()),
        "policy_change_count": float((frame.get("incident_flag", pd.Series([0])) > 0).sum()),
        "cost_slo_utility": float(1.0 - frame.get("slo_violation", pd.Series([0])).mean() - frame.get("cost", pd.Series([0.5])).mean() * 0.2),
    }
