from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.square_core.common.diagnostics import CAUTION
from square_sim.square_core.matrix.aggregate import summarize_metrics
from square_sim.utils.files import write_json, write_text


def write_core_reports(report_dir: Path, experiment_id: str, summary: dict[str, Any], metrics: pd.DataFrame) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate = summarize_metrics(metrics)
    write_json(report_dir / "experiment_summary.json", {**summary, "aggregate": aggregate, "caution": CAUTION})
    lines = [f"# SQUARE Core Validation Matrix v1: {experiment_id}", "", CAUTION, "", "## Run Summary"]
    lines += [
        f"- Planned: `{summary.get('total_planned', 0)}`",
        f"- Succeeded: `{summary.get('succeeded', 0)}`",
        f"- Failed: `{summary.get('failed', 0)}`",
        f"- Skipped: `{summary.get('skipped', 0)}`",
    ]
    lines += ["", "## Task Winners"]
    for row in aggregate["track_winners"]:
        lines.append(
            f"- `{row['track']}/{row['task']}`: `{row['best_system']}` cost-adjusted `{row['best_cost_adjusted_utility']:.4f}`"
        )
    write_text(report_dir / "experiment_summary.md", "\n".join(lines) + "\n")
    if not metrics.empty:
        metrics.to_parquet(report_dir / "metrics.parquet", index=False)
        metrics.to_csv(report_dir / "metrics.csv", index=False)
        component = (
            metrics.groupby(["track", "task", "system"])["cost_adjusted_utility"]
            .mean()
            .reset_index()
            .sort_values(["track", "task", "cost_adjusted_utility"], ascending=[True, True, False])
        )
        component.to_parquet(report_dir / "component_support_table.parquet", index=False)
    for track in sorted(metrics["track"].unique().tolist()) if not metrics.empty else []:
        sub = metrics[metrics["track"] == track]
        write_json(report_dir / "track_summaries" / f"{track}.json", summarize_metrics(sub))
        write_text(report_dir / "track_summaries" / f"{track}.md", f"# {track}\n\n{CAUTION}\n")
    return {
        "experiment_summary_json": str(report_dir / "experiment_summary.json"),
        "experiment_summary_md": str(report_dir / "experiment_summary.md"),
        "metrics_parquet": str(report_dir / "metrics.parquet"),
    }
