from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text

CAUTION = (
    "This is a matched-cost real-RAG software kill-test. It does not prove SQUARE hardware, "
    "quantum architecture, commercial viability, or broad RAG superiority."
)


def write_reports(
    report_dir: Path,
    *,
    experiment_id: str,
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    stats: dict[str, pd.DataFrame],
    sensitivity: pd.DataFrame,
    certificate: dict[str, Any],
    no_overwrite_audit: dict[str, Any],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    metrics.to_parquet(report_dir / "aggregate_metrics.parquet", index=False)
    metrics.to_csv(report_dir / "aggregate_metrics.csv", index=False)
    for name, frame in stats.items():
        frame.to_parquet(report_dir / f"{name}.parquet", index=False)
        frame.to_csv(report_dir / f"{name}.csv", index=False)
    sensitivity.to_parquet(report_dir / "utility_sensitivity_results.parquet", index=False)
    sensitivity.to_csv(report_dir / "utility_sensitivity_summary.csv", index=False)
    write_json(report_dir / "executive_summary.json", {**summary, "certificate_status": certificate.get("status")})
    statement = {
        "Signal supported": "Adaptive compute survived matched-cost baselines on real RAG evaluation data under the predeclared primary metric.",
        "Candidate signal": "Adaptive compute showed partial advantage but did not satisfy all matched-cost support criteria.",
        "Inconclusive": "Results were mixed, unstable, or utility-sensitive.",
        "Refused": "Adaptive compute did not survive matched-cost controls; simpler gating or no-fork methods were sufficient.",
        "Negative result": "Adaptive compute did not survive matched-cost controls; simpler gating or no-fork methods were sufficient.",
    }.get(str(certificate.get("status")), "Results were mixed, unstable, or utility-sensitive.")
    write_text(
        report_dir / "executive_summary.md",
        f"# SQUARETune Matched-Cost Real-RAG Kill-Test: {experiment_id}\n\n{CAUTION}\n\n"
        f"## Conclusion\n\n{statement}\n\n"
        f"- Planned: `{summary.get('planned')}`\n"
        f"- Succeeded: `{summary.get('succeeded')}`\n"
        f"- Failed: `{summary.get('failed')}`\n"
        f"- No-overwrite: `{no_overwrite_audit.get('status')}`\n"
        f"- Certificate: `{certificate.get('status')}`\n",
    )
    write_json(report_dir / "methods.json", {"matched_cost_design": "All gating controls match adaptive compute expensive-compute invocation rate within tolerance.", "caution": CAUTION})
    write_text(report_dir / "methods.md", "# Methods\n\nMatched-cost controls compare selective expensive-compute policies over held-out RAG examples.\n")
    write_json(report_dir / "dataset_report.json", summary.get("dataset", {}))
    write_json(report_dir / "scenario_report.json", summary.get("scenario", {}))
    write_json(report_dir / "matched_cost_baseline_report.json", stats.get("summary", pd.DataFrame()).to_dict(orient="records"))
    write_json(report_dir / "primary_results_report.json", stats.get("summary", pd.DataFrame()).to_dict(orient="records"))
    write_json(report_dir / "utility_sensitivity_report.json", sensitivity.to_dict(orient="records"))
    write_json(report_dir / "statistical_analysis_report.json", {k: v.to_dict(orient="records") for k, v in stats.items()})
    write_json(report_dir / "ablation_report.json", stats.get("paired_deltas", pd.DataFrame()).to_dict(orient="records"))
    write_json(report_dir / "kill_criteria_report.json", certificate.get("kill_criteria", {}))
    write_text(report_dir / "negative_result_report.md", "# Negative Result Analysis\n\nNegative or mixed results are preserved as valid kill-test evidence.\n")
    write_json(report_dir / "certificate_report.json", certificate)
    write_json(report_dir / "no_overwrite_audit.json", no_overwrite_audit)
    write_text(report_dir / "no_overwrite_audit.md", f"# No-Overwrite Audit\n\nStatus: `{no_overwrite_audit.get('status')}`\n")
    paths["executive_summary_md"] = str(report_dir / "executive_summary.md")
    paths["aggregate_metrics_csv"] = str(report_dir / "aggregate_metrics.csv")
    return paths

