from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text


def write_mechanism_report(output_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        text = "# SQUARETune Mechanism Report\n\nNo successful runs were found.\n"
        write_text(output_dir / "mechanism_report.md", text)
        return {"path": str(output_dir / "mechanism_report.md"), "rows": 0}
    grouped = (
        metrics.groupby(["dataset_key", "optimizer_name"])["final_utility"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset_key", "mean"], ascending=[True, False])
    )
    grouped.to_parquet(output_dir / "mechanism_table.parquet", index=False)
    lines = [f"# SQUARETune Mechanism Report: {experiment_id}", "", "| Dataset | Optimizer | Mean utility | Std | Runs |", "|---|---|---:|---:|---:|"]
    for row in grouped.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset_key']} | {row['optimizer_name']} | {row['mean']:.4f} | {0.0 if pd.isna(row['std']) else row['std']:.4f} | {int(row['count'])} |"
        )
    lines.append("")
    lines.append("This report is deterministic and uses synthetic mechanism diagnostics only.")
    write_text(output_dir / "mechanism_report.md", "\n".join(lines) + "\n")
    payload = {
        "experiment_id": experiment_id,
        "rows": len(metrics),
        "report_path": str(output_dir / "mechanism_report.md"),
    }
    write_json(output_dir / "mechanism_report.json", payload)
    return payload
