from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text


def write_commercial_value_report(output_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        write_text(output_dir / "commercial_value_report.md", "# SQUARETune Commercial Value Report\n\nNo data.\n")
        return {"path": str(output_dir / "commercial_value_report.md"), "rows": 0}
    table = (
        metrics.groupby("optimizer_name")
        .agg(
            final_utility=("final_utility", "mean"),
            cost_adjusted_improvement=("cost_adjusted_improvement", "mean"),
            regression_count=("regression_count", "mean"),
            experiments_to_threshold=("experiments_to_threshold", "mean"),
            simulated_gpu_hours=("simulated_gpu_hours", "mean"),
        )
        .reset_index()
        .sort_values("cost_adjusted_improvement", ascending=False)
    )
    table.to_parquet(output_dir / "commercial_value_table.parquet", index=False)
    lines = [f"# SQUARETune Commercial Value Report: {experiment_id}", "", "| Optimizer | Utility | Cost-adjusted improvement | Regressions | Experiments to threshold | Sim GPU-hours |", "|---|---:|---:|---:|---:|---:|"]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"| {row['optimizer_name']} | {row['final_utility']:.4f} | {row['cost_adjusted_improvement']:.4f} | {row['regression_count']:.2f} | {row['experiments_to_threshold']:.2f} | {row['simulated_gpu_hours']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Commercial interpretation should focus on eval improvement per experiment, regression reduction, and simulated cost efficiency.",
        ]
    )
    write_text(output_dir / "commercial_value_report.md", "\n".join(lines) + "\n")
    payload = {
        "experiment_id": experiment_id,
        "rows": len(metrics),
        "report_path": str(output_dir / "commercial_value_report.md"),
    }
    write_json(output_dir / "commercial_value_report.json", payload)
    return payload
