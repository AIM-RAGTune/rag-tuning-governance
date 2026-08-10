from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.config import Settings
from square_sim.utils.files import read_json, write_json, write_text


def write_run_diagnostics(
    output_dir: Path,
    *,
    run_id: str,
    task: str,
    seed: int,
    system: str,
    trajectory: pd.DataFrame,
    adaptive: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    diag_dir = output_dir / "adaptive_arch_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in trajectory.iterrows():
        round_idx = int(row.get("round_idx", 0))
        gate = adaptive[adaptive["round_idx"] == round_idx].iloc[0].to_dict() if not adaptive.empty and not adaptive[adaptive["round_idx"] == round_idx].empty else {}
        rows.append(
            {
                "run_id": run_id,
                "task": task,
                "seed": seed,
                "round_idx": round_idx,
                "current_regime": row.get("selected_region", "global"),
                "selected_region": row.get("selected_region", "global"),
                "action": row.get("selected_action_type", "none"),
                "architecture_before_hash": f"{system}:{round_idx}:before",
                "architecture_after_hash": f"{system}:{round_idx}:after",
                "local_error": 1.0 - float(row.get("state_utility", 0.0)),
                "uncertainty_score": gate.get("uncertainty_score", 0.0),
                "conflict_score": gate.get("objective_conflict_score", 0.0),
                "regression_risk_score": gate.get("regression_risk_score", 0.0),
                "budget_pressure": gate.get("budget_pressure", 0.0),
                "fork_invoked": bool(gate.get("fork_invoked", False)),
                "merge_invoked": bool(gate.get("merge_invoked", False)),
                "topology_changed": system == "square_adaptive_arch_full" and task == "dynamic_topology_routing",
                "memory_reused": bool(gate.get("memory_reused", False)),
                "nonlinear_rollout_used": bool(gate.get("fork_invoked", False)) and "linear" not in system,
                "expected_roi": gate.get("expected_value_of_fork", 0.0),
                "realized_roi": gate.get("fork_roi", 0.0),
                "utility_before": None,
                "utility_after": float(row.get("state_utility", 0.0)),
                "cost": float(row.get("cost_so_far", 0.0)),
                "notes": row.get("notes", ""),
            }
        )
    trace = pd.DataFrame(rows)
    trace.to_parquet(diag_dir / "architecture_trace.parquet", index=False)
    trace[trace["action"].astype(str).ne("none")].to_parquet(diag_dir / "local_reconfiguration_events.parquet", index=False)
    trace[trace["topology_changed"]].to_parquet(diag_dir / "topology_changes.parquet", index=False)
    trace[["round_idx", "expected_roi", "realized_roi", "fork_invoked"]].to_parquet(diag_dir / "fork_roi.parquet", index=False)
    trace[["round_idx", "realized_roi", "merge_invoked"]].to_parquet(diag_dir / "merge_roi.parquet", index=False)
    trace[trace["memory_reused"]].to_parquet(diag_dir / "memory_events.parquet", index=False)
    trace[trace["regression_risk_score"].astype(float) > 0.65].to_parquet(diag_dir / "protected_region_events.parquet", index=False)
    trace.to_parquet(diag_dir / "hard_subset_performance.parquet", index=False)
    summary = {
        "always_fork_equivalent_flag": bool(metrics.get("fork_invocation_rate", 0.0) > 0.90),
        "never_fork_equivalent_flag": bool(metrics.get("fork_invocation_rate", 0.0) < 0.01),
        "adaptive_compute_valid_flag": bool(0.01 <= metrics.get("fork_invocation_rate", 0.0) <= 0.90),
        "fork_invocation_rate": metrics.get("fork_invocation_rate", 0.0),
        "positive_fork_roi_rate": metrics.get("positive_fork_roi_rate", 0.0),
        "easy_subset_overcompute_rate": metrics.get("easy_subset_overcompute_rate", 0.0),
        "hard_subset_undercompute_rate": max(0.0, 0.75 - float(metrics.get("hard_subset_performance", 0.0))),
        "budget_saved_vs_full": metrics.get("budget_saved_vs_full", 0.0),
        "utility_preserved_vs_full": metrics.get("raw_utility_delta_vs_full", 0.0),
        "cost_adjusted_gain_vs_full": metrics.get("cost_adjusted_delta_vs_full", 0.0),
        "cost_adjusted_gain_vs_no_fork": metrics.get("cost_adjusted_delta_vs_no_fork", 0.0),
    }
    write_json(diag_dir / "architecture_summary.json", metrics)
    write_json(diag_dir / "compute_allocation_summary.json", summary)
    write_text(
        diag_dir / "adaptive_arch_summary.md",
        f"# Adaptive Architecture Run\n\n- Run: `{run_id}`\n- Task: `{task}`\n- System: `{system}`\n- Cost-adjusted utility: `{metrics.get('cost_adjusted_utility')}`\n",
    )
    return {"diagnostics_dir": str(diag_dir), "trace_path": str(diag_dir / "architecture_trace.parquet")}


def diagnose_experiment(settings: Settings, experiment_id: str) -> dict[str, Any]:
    report_dir = settings.project_root / "reports" / "square_adaptive_arch" / "v1" / experiment_id
    metrics_path = report_dir / "metrics.parquet"
    metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
    records = []
    for task, group in metrics.groupby("task") if not metrics.empty else []:
        records.append(
            {
                "task": task,
                "best_system": group.sort_values("cost_adjusted_utility", ascending=False).iloc[0]["system"],
                "best_cost_adjusted_utility": float(group["cost_adjusted_utility"].max()),
                "adaptive_arch_full_mean": float(group[group["system"] == "square_adaptive_arch_full"]["cost_adjusted_utility"].mean()) if not group[group["system"] == "square_adaptive_arch_full"].empty else None,
            }
        )
    payload = {"experiment_id": experiment_id, "records": records}
    write_json(report_dir / "adaptive_arch_diagnostics.json", payload)
    write_text(report_dir / "adaptive_arch_diagnostics.md", "# Adaptive Architecture Diagnostics\n\n" + "\n".join(f"- {r['task']}: {r['best_system']}" for r in records) + "\n")
    return payload

