from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.external.reporting import write_no_overwrite_audit
from square_sim.utils.files import write_json, write_text

CAUTION = (
    "This benchmark tests adaptive computational architecture in software. "
    "It does not prove SQUARE hardware, physical quantum behavior, or commercial ROI."
)


def write_report(output_dir: Path, *, experiment_id: str, summary: dict[str, Any], metrics: pd.DataFrame) -> dict[str, Any]:
    winners = []
    for task, group in metrics.groupby("task") if not metrics.empty else []:
        best = group.sort_values("cost_adjusted_utility", ascending=False).iloc[0]
        winners.append({"task": task, "best_system": best["system"], "best_cost_adjusted_utility": float(best["cost_adjusted_utility"])})
    payload = {"experiment_id": experiment_id, "caution": CAUTION, "summary": summary, "task_winners": winners}
    write_json(output_dir / "benchmark_summary.json", payload)
    lines = [
        f"# SQUARE Adaptive Architecture Benchmark: {experiment_id}",
        "",
        CAUTION,
        "",
        f"- Planned: `{summary.get('total_planned')}`",
        f"- Succeeded: `{summary.get('succeeded')}`",
        f"- Skipped: `{summary.get('skipped')}`",
        f"- Failed: `{summary.get('failed')}`",
        "",
        "| Task | Best System | Best Cost-Adjusted Utility |",
        "|---|---|---:|",
    ]
    for row in winners:
        lines.append(f"| {row['task']} | {row['best_system']} | {row['best_cost_adjusted_utility']} |")
    write_text(output_dir / "benchmark_summary.md", "\n".join(lines) + "\n")
    for name in [
        "adaptive_arch_component_support",
        "compute_allocation_report",
        "fork_rollout_merge_report",
        "memory_and_regression_report",
        "dynamic_topology_report",
        "external_proxy_report",
    ]:
        write_json(output_dir / f"{name}.json", payload)
        write_text(output_dir / f"{name}.md", "\n".join(lines) + "\n")
    return payload


__all__ = ["write_no_overwrite_audit", "write_report"]

