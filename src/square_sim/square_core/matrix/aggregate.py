from __future__ import annotations

from typing import Any

import pandas as pd


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {"rows": 0, "track_winners": []}
    winners = []
    for (track, task), group in metrics.groupby(["track", "task"]):
        scores = group.groupby("system")["cost_adjusted_utility"].mean().sort_values(ascending=False)
        winners.append(
            {
                "track": str(track),
                "task": str(task),
                "best_system": str(scores.index[0]),
                "best_cost_adjusted_utility": float(scores.iloc[0]),
                "best_final_utility": float(group[group["system"] == scores.index[0]]["final_utility"].mean()),
            }
        )
    return {"rows": len(metrics), "track_winners": winners}
