from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from square_sim.square_tune_matched_cost.config import UTILITY_WEIGHTS
from square_sim.square_tune_matched_cost.metrics import cost_adjusted_utility

PRIMARY = "held_out_test_cost_adjusted_utility"


def aggregate_statistics(metrics: pd.DataFrame, *, bootstrap_samples: int = 1000, seed: int = 101) -> dict[str, pd.DataFrame]:
    if metrics.empty:
        empty = pd.DataFrame()
        return {"summary": empty, "paired_deltas": empty, "bootstrap_intervals": empty, "rankings": empty}
    summary = (
        metrics.groupby("system", as_index=False)
        .agg(
            mean_cost_adjusted=(PRIMARY, "mean"),
            std_cost_adjusted=(PRIMARY, "std"),
            median_cost_adjusted=(PRIMARY, "median"),
            mean_raw_quality=("held_out_test_raw_quality", "mean"),
            mean_cost=("total_cost_proxy", "mean"),
            mean_expensive_rate=("expensive_compute_invocation_rate", "mean"),
            mean_roi_rate=("positive_expensive_compute_roi_rate", "mean"),
        )
        .fillna(0.0)
        .sort_values("mean_cost_adjusted", ascending=False)
    )
    rankings = (
        metrics.assign(rank=metrics.groupby("seed")[PRIMARY].rank(ascending=False, method="min"))
        .groupby("system", as_index=False)
        .agg(mean_rank=("rank", "mean"), best_seed_count=("rank", lambda x: int((x == 1).sum())))
        .sort_values("mean_rank")
    )
    adaptive = metrics[metrics["system"] == "square_tune_adaptive_compute"].set_index("seed")
    deltas = []
    intervals = []
    rng = np.random.default_rng(seed)
    for system, group in metrics.groupby("system"):
        if system == "square_tune_adaptive_compute":
            continue
        other = group.set_index("seed")
        shared = sorted(set(adaptive.index) & set(other.index))
        if not shared:
            continue
        diff = adaptive.loc[shared, PRIMARY].to_numpy() - other.loc[shared, PRIMARY].to_numpy()
        deltas.append(
            {
                "baseline": system,
                "mean_delta": float(diff.mean()),
                "median_delta": float(np.median(diff)),
                "effect_size": float(diff.mean() / (diff.std(ddof=1) + 1e-9)) if len(diff) > 1 else 0.0,
                "win_rate": float((diff > 0).mean()),
                "seed_count": len(diff),
            }
        )
        samples = []
        for _ in range(max(1, bootstrap_samples)):
            idx = rng.integers(0, len(diff), len(diff))
            samples.append(float(diff[idx].mean()))
        intervals.append(
            {
                "comparison": f"square_tune_adaptive_compute_vs_{system}",
                "mean_delta": float(diff.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_samples": int(max(1, bootstrap_samples)),
            }
        )
    return {
        "summary": summary,
        "paired_deltas": pd.DataFrame(deltas),
        "bootstrap_intervals": pd.DataFrame(intervals),
        "rankings": rankings,
    }


def utility_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, weights in UTILITY_WEIGHTS.items():
        frame = metrics.copy()
        frame["sensitivity_utility"] = [
            cost_adjusted_utility(
                quality=float(row["held_out_test_raw_quality"]),
                cost=float(row["total_cost_proxy"]),
                latency=float(row["simulated_latency_cost"]),
                regression=float(row["regression_count"]),
                weights=weights,
            )
            for _, row in frame.iterrows()
        ]
        agg = frame.groupby("system", as_index=False)["sensitivity_utility"].mean().sort_values("sensitivity_utility", ascending=False)
        for rank, row in enumerate(agg.itertuples(index=False), start=1):
            rows.append(
                {
                    "weight_setting": name,
                    "system": row.system,
                    "rank": rank,
                    "utility": float(row.sensitivity_utility),
                    "is_adaptive_compute": row.system == "square_tune_adaptive_compute",
                }
            )
    return pd.DataFrame(rows)

