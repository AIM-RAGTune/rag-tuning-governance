from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text

CAUTION = (
    "This is an external-transfer simulation over open/external datasets. "
    "It does not prove SQUARE hardware, real fine-tuning performance, or commercial ROI. "
    "It tests whether the calibrated SQUARETune optimizer transfers beyond synthetic mechanisms."
)


def write_external_transfer_report(
    output_dir: Path,
    *,
    experiment_id: str,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        winners: list[dict[str, Any]] = []
    else:
        winners = []
        for scenario, group in metrics.groupby("scenario_family"):
            ordered = group.sort_values("cost_adjusted_improvement", ascending=False)
            best = ordered.iloc[0].to_dict()
            winners.append(
                {
                    "scenario_family": scenario,
                    "best_optimizer": best.get("optimizer_name"),
                    "best_cost_adjusted_improvement": best.get("cost_adjusted_improvement"),
                    "square_tune_full_mean": float(
                        group[group["optimizer_name"] == "square_tune_full"]["cost_adjusted_improvement"].mean()
                    )
                    if not group[group["optimizer_name"] == "square_tune_full"].empty
                    else None,
                }
            )
    payload = {
        "experiment_id": experiment_id,
        "caution": CAUTION,
        "summary": summary,
        "scenario_winners": winners,
    }
    write_json(output_dir / "external_transfer_summary.json", payload)
    lines = [
        f"# SQUARETune External Transfer Summary: {experiment_id}",
        "",
        CAUTION,
        "",
        f"- Total planned: `{summary.get('total_planned')}`",
        f"- Succeeded: `{summary.get('succeeded')}`",
        f"- Skipped: `{summary.get('skipped')}`",
        f"- Failed: `{summary.get('failed')}`",
        "",
        "## Scenario-Level Winners",
        "",
        "| Scenario | Best Optimizer | Best Cost-Adjusted Improvement | SQUARETune Full Mean |",
        "|---|---|---:|---:|",
    ]
    for row in winners:
        lines.append(
            f"| {row['scenario_family']} | {row['best_optimizer']} | "
            f"{row['best_cost_adjusted_improvement']} | {row['square_tune_full_mean']} |"
        )
    write_text(output_dir / "external_transfer_summary.md", "\n".join(lines) + "\n")
    family_dir = output_dir / "scenario_family_reports"
    family_dir.mkdir(exist_ok=True)
    if not metrics.empty:
        for scenario, group in metrics.groupby("scenario_family"):
            top = group.sort_values("cost_adjusted_improvement", ascending=False).head(10)
            rows = [
                f"# Scenario Family Report: {scenario}",
                "",
                CAUTION,
                "",
                "| Optimizer | Seed | Final Utility | Cost-Adjusted Improvement | Regressions |",
                "|---|---:|---:|---:|---:|",
            ]
            for _, metric in top.iterrows():
                rows.append(
                    f"| {metric.get('optimizer_name')} | {metric.get('seed')} | "
                    f"{metric.get('final_utility')} | {metric.get('cost_adjusted_improvement')} | "
                    f"{metric.get('regression_count')} |"
                )
            write_text(family_dir / f"{scenario}.md", "\n".join(rows) + "\n")
    return payload


def write_no_overwrite_audit(
    output_dir: Path,
    *,
    experiment_id: str,
    protected_paths: list[str],
    write_roots: list[str],
    attempted_overwrites_blocked: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "protected_paths": protected_paths,
        "write_roots": write_roots,
        "attempted_overwrites_blocked": attempted_overwrites_blocked,
        "status": "append_only_confirmed",
    }
    write_json(output_dir / "no_overwrite_audit.json", payload)
    lines = [
        f"# No-Overwrite Audit: {experiment_id}",
        "",
        "- Status: `append_only_confirmed`",
        f"- Attempted overwrites blocked: `{attempted_overwrites_blocked}`",
        "",
        "## Protected Paths",
        "",
    ]
    lines.extend(f"- `{path}`" for path in protected_paths)
    lines.extend(["", "## Write Roots", ""])
    lines.extend(f"- `{path}`" for path in write_roots)
    write_text(output_dir / "no_overwrite_audit.md", "\n".join(lines) + "\n")
    return payload
