from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text


def write_reports(report_dir: Path, *, experiment_id: str, summary: dict[str, Any], metrics: pd.DataFrame, certificate: dict[str, Any], no_overwrite_audit: dict[str, Any]) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_dir / "aggregate_metrics.csv", index=False)
    metrics.to_parquet(report_dir / "aggregate_metrics.parquet", index=False)
    write_json(report_dir / "executive_summary.json", summary)
    write_json(report_dir / "track_summary.json", {"tracks": summary.get("track_summaries", {})})
    write_json(report_dir / "certificate_index.json", certificate)
    write_json(report_dir / "no_overwrite_audit.json", no_overwrite_audit)
    track_names = [
        "rag_hard_subset",
        "no_fork_robustness",
        "adaptive_escalation",
        "claim_level_faithfulness",
        "elastic_compute_policy",
        "square_core_v2_targeted",
        "ablation",
        "utility_sensitivity",
        "hard_subset",
    ]
    for name in track_names:
        frame = metrics[metrics["track"].str.contains(name.replace("_report", ""), regex=False)] if not metrics.empty else metrics
        write_json(report_dir / f"{name}_report.json", {"rows": frame.to_dict(orient="records")})
        write_text(report_dir / f"{name}_report.md", f"# {name.replace('_', ' ').title()} Report\n\nSee JSON/CSV artifacts for metric details.\n")
    statement = _executive_statement(certificate)
    lines = [
        "# SQUARE Next Simulation Package v1",
        "",
        statement,
        "",
        f"- Experiment: `{experiment_id}`",
        f"- Planned: {summary.get('planned', 0)}",
        f"- Succeeded: {summary.get('succeeded', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Skipped: {summary.get('skipped', 0)}",
        f"- No-overwrite audit: {no_overwrite_audit.get('status')}",
        "",
        "This benchmark tests software simulations and computational ontology only.",
    ]
    write_text(report_dir / "executive_summary.md", "\n".join(lines))
    write_text(report_dir / "track_summary.md", _track_summary_md(summary))
    write_text(report_dir / "no_overwrite_audit.md", f"# No-Overwrite Audit\n\nStatus: {no_overwrite_audit.get('status')}\n")
    return {path.name: str(path) for path in report_dir.iterdir() if path.is_file()}


def _executive_statement(certificate: dict[str, Any]) -> str:
    statuses = {cert["track"]: cert["status"] for cert in certificate.get("track_certificates", [])}
    return (
        "No-fork remains the RAG default candidate unless hard-subset or tiered escalation certificates show support. "
        f"Track statuses: {statuses}."
    )


def _track_summary_md(summary: dict[str, Any]) -> str:
    lines = ["# Track Summary", ""]
    for track, payload in summary.get("track_summaries", {}).items():
        lines.append(f"## {track}")
        lines.append(f"- Winner: {payload.get('winner')}")
        lines.append(f"- Mean cost-adjusted utility: {payload.get('best_cost_adjusted_utility')}")
        lines.append("")
    return "\n".join(lines)

