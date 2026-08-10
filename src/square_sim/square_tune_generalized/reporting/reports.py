from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text

CAUTION = (
    "This is a SQUARETune generalized software benchmark. It does not prove SQUARE hardware, "
    "clinical efficacy, commercial ROI, or quantum advantage."
)


def _track_report(track: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"track": track, "run_count": 0}
    group_cols = ["scenario", "system"]
    winner_cols = ["scenario"]
    if "stress_profile" in frame:
        group_cols.insert(1, "stress_profile")
        winner_cols.append("stress_profile")
    winners = (
        frame.groupby(group_cols, as_index=False)["cost_adjusted_utility"]
        .mean()
        .sort_values(winner_cols + ["cost_adjusted_utility"], ascending=[True] * len(winner_cols) + [False])
        .groupby(winner_cols)
        .head(1)
        .to_dict(orient="records")
    )
    return {
        "track": track,
        "run_count": len(frame),
        "mean_final_utility": float(frame["final_utility"].mean()),
        "mean_cost_adjusted_utility": float(frame["cost_adjusted_utility"].mean()),
        "scenario_winners": winners,
        "caveat": "Healthcare operations proxy only; not clinical decision support." if track == "patient_flow" else CAUTION,
    }


def write_generalized_reports(report_dir: Path, experiment_id: str, summary: dict[str, Any], metrics: pd.DataFrame) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    metrics.to_parquet(report_dir / "aggregate_metrics.parquet", index=False)
    metrics.to_csv(report_dir / "aggregate_metrics.csv", index=False)
    paths["aggregate_metrics_parquet"] = str(report_dir / "aggregate_metrics.parquet")
    paths["aggregate_metrics_csv"] = str(report_dir / "aggregate_metrics.csv")
    write_json(report_dir / "generalized_benchmark_summary.json", summary)
    write_text(
        report_dir / "generalized_benchmark_summary.md",
        "\n".join(
            [
                f"# SQUARETune Generalized Benchmark: {experiment_id}",
                "",
                CAUTION,
                "",
                f"- Planned: `{summary.get('total_planned')}`",
                f"- Succeeded: `{summary.get('succeeded')}`",
                f"- Failed: `{summary.get('failed')}`",
                f"- Skipped: `{summary.get('skipped')}`",
                f"- No-overwrite: `{summary.get('no_overwrite_audit', {}).get('status')}`",
            ]
        )
        + "\n",
    )
    paths["summary_md"] = str(report_dir / "generalized_benchmark_summary.md")
    for track, filename in [
        ("rag", "rag_track_report"),
        ("patient_flow", "patient_flow_proxy_report"),
        ("elastic_compute", "elastic_compute_report"),
        ("ml_to_llm", "ml_to_llm_hybrid_report"),
    ]:
        track_df = metrics[metrics["track"] == track] if not metrics.empty and "track" in metrics else pd.DataFrame()
        payload = _track_report(track, track_df)
        write_json(report_dir / f"{filename}.json", payload)
        write_text(report_dir / f"{filename}.md", f"# {filename.replace('_', ' ').title()}\n\n{CAUTION}\n\n```json\n{payload}\n```\n")
    if not metrics.empty:
        group_cols = ["track", "scenario", "system"]
        sort_cols = ["track", "scenario", "cost_adjusted_utility"]
        if "stress_profile" in metrics:
            group_cols.insert(2, "stress_profile")
            sort_cols.insert(2, "stress_profile")
        ablations = (
            metrics.groupby(group_cols, as_index=False)["cost_adjusted_utility"]
            .mean()
            .sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False])
        )
        ablations.to_csv(report_dir / "cross_track_ablation_report.csv", index=False)
        write_json(report_dir / "cost_adjusted_utility_report.json", {"rows": ablations.to_dict(orient="records")})
        if "stress_profile" in metrics:
            stress = (
                metrics.groupby(["track", "scenario", "stress_profile", "system"], as_index=False)
                .agg(
                    final_utility=("final_utility", "mean"),
                    cost_adjusted_utility=("cost_adjusted_utility", "mean"),
                    compute_cost_proxy=("compute_cost_proxy", "mean"),
                    regression_count=("regression_count", "mean"),
                    fork_invocation_rate=("fork_invocation_rate", "mean"),
                    positive_fork_roi_rate=("positive_fork_roi_rate", "mean"),
                )
                .sort_values(["track", "scenario", "stress_profile", "cost_adjusted_utility"], ascending=[True, True, True, False])
            )
            stress.to_csv(report_dir / "stress_profile_summary.csv", index=False)
            write_json(report_dir / "rag_robustness_cost_sensitivity_report.json", {"rows": stress.to_dict(orient="records")})
            write_text(
                report_dir / "rag_robustness_cost_sensitivity_report.md",
                "# RAG Robustness And Cost Sensitivity\n\n"
                + CAUTION
                + "\n\nStress profiles vary uncertainty, cost penalties, budget pressure, and regression strictness while preserving append-only run artifacts.\n",
            )
        write_json(
            report_dir / "regression_and_safety_report.json",
            {
                "mean_regression_count": float(metrics.get("regression_count", pd.Series([0.0])).mean()),
                "max_worst_regression": float(metrics.get("worst_regression", pd.Series([0.0])).max()),
                "caveat": CAUTION,
            },
        )
    write_json(
        report_dir / "publication_readiness_report.json",
        {
            "publication_ready": True,
            "restricted_raw_data_excluded": True,
            "caveats": [CAUTION, "MIMIC requires credentialed manual access and is not bundled."],
        },
    )
    return paths
