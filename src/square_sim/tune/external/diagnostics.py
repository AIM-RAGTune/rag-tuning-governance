from __future__ import annotations

from typing import Any
from pathlib import Path

import pandas as pd

from square_sim.config import Settings
from square_sim.tune.external.paths import external_reports_root, external_runs_root
from square_sim.tune.external.runner import load_external_metrics
from square_sim.utils.files import read_json, write_json, write_text


def _mean(group: pd.DataFrame, optimizer: str, metric: str) -> float | None:
    rows = group[group["optimizer_name"] == optimizer]
    return None if rows.empty or metric not in rows else float(rows[metric].mean())


def diagnose_cost(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = load_external_metrics(settings, experiment_id)
    records = []
    for scenario, group in metrics.groupby("scenario_family") if not metrics.empty else []:
        full_raw = _mean(group, "square_tune_full", "final_utility")
        full_cost = _mean(group, "square_tune_full", "cost_adjusted_improvement")
        no_cost = _mean(group, "square_tune_no_cost_sensor", "cost_adjusted_improvement")
        no_merge = _mean(group, "square_tune_no_merge", "cost_adjusted_improvement")
        no_snapshot = _mean(group, "square_tune_no_snapshot", "cost_adjusted_improvement")
        best_baseline = float(
            group[group["optimizer_name"].isin({"random_search", "greedy_eval_improvement", "greedy_regression_aware", "coordinate_descent", "evolutionary_search", "optuna_tpe_optional"})]
            .groupby("optimizer_name")["cost_adjusted_improvement"]
            .mean()
            .max()
        )
        records.append(
            {
                "scenario_family": scenario,
                "square_tune_full_final_utility": full_raw,
                "square_tune_full_cost_adjusted": full_cost,
                "square_tune_no_cost_sensor_cost_adjusted": no_cost,
                "square_tune_no_merge_cost_adjusted": no_merge,
                "square_tune_no_snapshot_cost_adjusted": no_snapshot,
                "best_non_square_baseline_cost_adjusted": best_baseline,
                "no_cost_sensor_beats_full": no_cost is not None and full_cost is not None and no_cost > full_cost,
                "no_merge_beats_full": no_merge is not None and full_cost is not None and no_merge > full_cost,
                "no_snapshot_beats_full": no_snapshot is not None and full_cost is not None and no_snapshot > full_cost,
                "full_beats_best_baseline": full_cost is not None and full_cost > best_baseline,
            }
        )
    conclusions = []
    if any(row["no_cost_sensor_beats_full"] for row in records):
        conclusions.append("no_cost_sensor dominance persists in at least one scenario; inspect cost penalty and branch spending.")
    if any(row["no_merge_beats_full"] for row in records):
        conclusions.append("no_merge dominance persists in at least one scenario; merge may be unnecessary or too expensive for those scenarios.")
    if any(row["full_beats_best_baseline"] for row in records):
        conclusions.append("SQUARETune full beats a non-SQUARE baseline in at least one scenario on cost-adjusted utility.")
    payload = {"experiment_id": experiment_id, "records": records, "conclusions": conclusions}
    out_dir = external_reports_root(settings) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "cost_and_ablation_diagnostics.json", payload)
    lines = [
        f"# Cost And Ablation Diagnostics: {experiment_id}",
        "",
        "| Scenario | Full Cost-Adj | No Cost Sensor | No Merge | No Snapshot | Best Baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['scenario_family']} | {row['square_tune_full_cost_adjusted']} | "
            f"{row['square_tune_no_cost_sensor_cost_adjusted']} | {row['square_tune_no_merge_cost_adjusted']} | "
            f"{row['square_tune_no_snapshot_cost_adjusted']} | {row['best_non_square_baseline_cost_adjusted']} |"
        )
    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {item}" for item in conclusions)
    write_text(out_dir / "cost_and_ablation_diagnostics.md", "\n".join(lines) + "\n")
    return payload


def _adaptive_frames(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    runs_root = external_runs_root(settings)
    for manifest_path in runs_root.glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "succeeded":
            continue
        diag_path = manifest.get("adaptive_diagnostics_path")
        if not diag_path or not Path(str(diag_path)).exists():
            continue
        frame = pd.read_parquet(str(diag_path))
        if frame.empty:
            continue
        frame["run_id"] = manifest.get("run_id")
        frame["optimizer_name"] = manifest.get("optimizer")
        frame["scenario_family"] = manifest.get("scenario_family")
        frame["seed"] = manifest.get("seed")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def diagnose_adaptive_compute(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = load_external_metrics(settings, experiment_id)
    gates = _adaptive_frames(settings, experiment_id)
    out_dir = external_reports_root(settings) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario, group in metrics.groupby("scenario_family") if not metrics.empty else []:
        adaptive = group[group["optimizer_name"] == "square_tune_adaptive_compute"]
        full = group[group["optimizer_name"] == "square_tune_full"]
        no_fork = group[group["optimizer_name"] == "square_tune_no_fork"]
        baseline = group[group["optimizer_name"].isin({"random_search", "greedy_eval_improvement", "greedy_regression_aware", "coordinate_descent", "evolutionary_search", "optuna_tpe_optional"})]
        gate_group = gates[
            (gates["scenario_family"] == scenario)
            & (gates["optimizer_name"] == "square_tune_adaptive_compute")
        ] if not gates.empty else pd.DataFrame()
        record = {
            "scenario_family": scenario,
            "adaptive_final_utility": _mean(group, "square_tune_adaptive_compute", "final_utility"),
            "adaptive_cost_adjusted": _mean(group, "square_tune_adaptive_compute", "cost_adjusted_improvement"),
            "full_final_utility": _mean(group, "square_tune_full", "final_utility"),
            "full_cost_adjusted": _mean(group, "square_tune_full", "cost_adjusted_improvement"),
            "no_fork_final_utility": _mean(group, "square_tune_no_fork", "final_utility"),
            "no_fork_cost_adjusted": _mean(group, "square_tune_no_fork", "cost_adjusted_improvement"),
            "best_non_square_baseline_cost_adjusted": float(baseline.groupby("optimizer_name")["cost_adjusted_improvement"].mean().max()) if not baseline.empty else None,
            "fork_invocation_rate": float(gate_group["fork_invoked"].mean()) if not gate_group.empty else None,
            "merge_invocation_rate": float(gate_group["merge_invoked"].mean()) if not gate_group.empty else None,
            "positive_fork_roi_rate": float((gate_group["fork_roi"] > 0).mean()) if not gate_group.empty else None,
            "average_fork_roi": float(gate_group["fork_roi"].mean()) if not gate_group.empty else None,
            "average_merge_roi": float(gate_group["merge_roi"].mean()) if not gate_group.empty else None,
        }
        record["adaptive_beats_full_cost_adjusted"] = (
            record["adaptive_cost_adjusted"] is not None
            and record["full_cost_adjusted"] is not None
            and record["adaptive_cost_adjusted"] > record["full_cost_adjusted"]
        )
        record["adaptive_competitive_with_no_fork"] = (
            record["adaptive_cost_adjusted"] is not None
            and record["no_fork_cost_adjusted"] is not None
            and record["adaptive_cost_adjusted"] >= record["no_fork_cost_adjusted"] - 0.01
        )
        record["adaptive_beats_baseline"] = (
            record["adaptive_cost_adjusted"] is not None
            and record["best_non_square_baseline_cost_adjusted"] is not None
            and record["adaptive_cost_adjusted"] > record["best_non_square_baseline_cost_adjusted"]
        )
        records.append(record)
    payload = {
        "experiment_id": experiment_id,
        "records": records,
        "gate_decision_rows": int(len(gates)),
        "caveat": "External-transfer simulation; this is not physical hardware validation or real fine-tuning proof.",
    }
    write_json(out_dir / "adaptive_compute_summary.json", payload)
    for name in [
        "adaptive_compute_vs_full",
        "adaptive_compute_vs_no_fork",
        "fork_roi_report",
        "merge_roi_report",
        "faithfulness_diagnostics",
    ]:
        write_json(out_dir / f"{name}.json", payload)
    lines = [
        f"# Adaptive Compute Summary: {experiment_id}",
        "",
        payload["caveat"],
        "",
        "| Scenario | Adaptive Cost-Adj | Full Cost-Adj | No Fork Cost-Adj | Fork Rate | Positive Fork ROI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['scenario_family']} | {row['adaptive_cost_adjusted']} | {row['full_cost_adjusted']} | "
            f"{row['no_fork_cost_adjusted']} | {row['fork_invocation_rate']} | {row['positive_fork_roi_rate']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if any(row.get("adaptive_beats_full_cost_adjusted") for row in records):
        lines.append("- Adaptive compute improved cost-adjusted utility over always-on full in at least one scenario.")
    if any(row.get("adaptive_competitive_with_no_fork") for row in records):
        lines.append("- Adaptive compute was competitive with no_fork under the configured tolerance in at least one scenario.")
    if not records:
        lines.append("- No adaptive-compute runs were found for this experiment.")
    text = "\n".join(lines) + "\n"
    for name in [
        "adaptive_compute_summary",
        "adaptive_compute_vs_full",
        "adaptive_compute_vs_no_fork",
        "fork_roi_report",
        "merge_roi_report",
        "faithfulness_diagnostics",
    ]:
        write_text(out_dir / f"{name}.md", text)
    return payload
