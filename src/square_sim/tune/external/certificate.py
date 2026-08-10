from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.external.reporting import CAUTION
from square_sim.utils.files import read_json, write_json, write_text

NON_SQUARE_BASELINES = {
    "random_search",
    "greedy_eval_improvement",
    "greedy_regression_aware",
    "coordinate_descent",
    "evolutionary_search",
    "linear_utility_optimizer",
    "ridge_utility_optimizer",
    "optuna_tpe_optional",
}
RELEVANT_ABLATIONS = {
    "square_tune_no_snapshot",
    "square_tune_no_fork",
    "square_tune_linear_rollout",
    "square_tune_no_merge",
    "square_tune_no_memory",
    "square_tune_no_feedback",
    "square_tune_no_regression_sensor",
    "square_tune_no_cost_sensor",
}
ADAPTIVE_VARIANTS = {
    "square_tune_adaptive_compute",
    "square_tune_adaptive_compute_no_uncertainty_gate",
    "square_tune_adaptive_compute_no_conflict_gate",
    "square_tune_adaptive_compute_no_roi_gate",
    "square_tune_adaptive_compute_no_budget_gate",
    "square_tune_adaptive_compute_no_regression_escalation",
    "square_tune_adaptive_compute_no_memory_reuse",
    "square_tune_adaptive_compute_always_fork",
    "square_tune_adaptive_compute_never_fork",
}


def _calibration_reference(settings_root: Path) -> dict[str, Any]:
    base = settings_root / "certificates" / "square_tune" / "calibration"
    preferred = base / "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd" / "certificate_index.json"
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted(base.glob("*/certificate_index.json"), reverse=True))
    for path in candidates:
        try:
            payload = read_json(path)
        except Exception:
            continue
        gates = payload.get("calibration_gates", {})
        if gates.get("global_status") == "passed":
            return {
                "status": "passed",
                "calibration_experiment_id": payload.get("experiment_id") or path.parent.name,
                "calibration_certificate_path": str(path),
                "calibration_global_gate_status": "passed",
            }
    return {
        "status": "missing",
        "calibration_experiment_id": None,
        "calibration_certificate_path": None,
        "calibration_global_gate_status": "missing",
    }


def _budget_parity(group: pd.DataFrame) -> bool:
    if "response_surface_evaluations" not in group:
        return False
    values = group["response_surface_evaluations"].dropna().astype(float)
    if values.empty:
        return False
    return float(values.max() - values.min()) <= max(1.0, 0.02 * float(values.max()))


def _license_status(group: pd.DataFrame) -> str:
    values = sorted({str(v) for v in group.get("source_license_status", pd.Series(dtype=str)).dropna().tolist()})
    if not values:
        return "unknown"
    if any(value in {"missing", "unknown"} for value in values):
        return "unknown"
    if any(value in {"override", "captured"} for value in values):
        return "captured"
    return values[0]


def certificate_for_scenario(
    scenario_family: str,
    group: pd.DataFrame,
    *,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    license_status = _license_status(group)
    source_appropriate = bool(group.get("source_appropriate", pd.Series([False])).fillna(False).astype(bool).all())
    if calibration.get("status") != "passed":
        status = "Calibration prerequisite missing"
        reason = "Calibration v2 passed-gate certificate was not found."
    elif not _budget_parity(group):
        status = "Budget confounded"
        reason = "Optimizers were not compared under equalized response-surface evaluation budgets."
    elif license_status in {"unknown", "missing"}:
        status = "License restricted / internal only"
        reason = "One or more source datasets have unknown or missing license metadata."
    elif not source_appropriate:
        status = "Inconclusive"
        reason = "Scenario-source mapping was not marked source-appropriate."
    elif "square_tune_adaptive_compute" in set(group.get("optimizer_name", pd.Series(dtype=str)).astype(str)):
        adaptive_rows = group[group["optimizer_name"] == "square_tune_adaptive_compute"]
        full_rows = group[group["optimizer_name"] == "square_tune_full"]
        no_fork_rows = group[group["optimizer_name"] == "square_tune_no_fork"]
        baseline_rows = group[group["optimizer_name"].isin(NON_SQUARE_BASELINES)]
        if adaptive_rows.empty or baseline_rows.empty:
            status = "Inconclusive"
            reason = "Missing adaptive-compute or non-SQUARE baseline runs."
        else:
            adaptive_cost = float(adaptive_rows["cost_adjusted_improvement"].mean())
            adaptive_raw = float(adaptive_rows["final_utility"].mean())
            best_baseline = float(baseline_rows.groupby("optimizer_name")["cost_adjusted_improvement"].mean().max())
            full_cost = float(full_rows["cost_adjusted_improvement"].mean()) if not full_rows.empty else None
            full_raw = float(full_rows["final_utility"].mean()) if not full_rows.empty else None
            no_fork_cost = float(no_fork_rows["cost_adjusted_improvement"].mean()) if not no_fork_rows.empty else None
            fork_rate = float(adaptive_rows.get("fork_invocation_rate", pd.Series([0.0])).fillna(0.0).mean())
            positive_roi = float(adaptive_rows.get("positive_fork_roi_rate", pd.Series([0.0])).fillna(0.0).mean())
            if fork_rate < 0.01:
                status = "Degenerate no_fork equivalent"
                reason = "Adaptive compute almost never invoked fork; interpret as a no_fork-equivalent result."
            elif fork_rate > 0.90:
                status = "Degenerate full equivalent"
                reason = "Adaptive compute forked on nearly every round; interpret as full always-on compute."
            elif adaptive_cost <= best_baseline:
                status = "Refused"
                reason = "A non-SQUARE baseline matched or beat adaptive compute on cost-adjusted improvement."
            elif (
                full_cost is not None
                and full_raw is not None
                and no_fork_cost is not None
                and adaptive_cost > full_cost + 0.003
                and adaptive_raw >= full_raw - 0.01
                and adaptive_cost >= no_fork_cost - 0.01
                and positive_roi >= 0.50
            ):
                status = "Adaptive external signal supported"
                reason = "Adaptive compute beat baselines and full on cost-adjusted utility while remaining close to full raw utility and non-degenerate."
            elif full_cost is not None and adaptive_cost > full_cost and adaptive_cost > best_baseline:
                status = "Candidate adaptive external signal"
                reason = "Adaptive compute improved over full and baselines on cost-adjusted utility, but no_fork or fork-ROI evidence remains incomplete."
            elif adaptive_cost > best_baseline:
                status = "Candidate adaptive external signal"
                reason = "Adaptive compute beat non-SQUARE baselines, but adaptive-family comparisons remain mixed."
            else:
                status = "Inconclusive"
                reason = "The adaptive transfer signal is mixed or below threshold."
    else:
        full_rows = group[group["optimizer_name"] == "square_tune_full"]
        baseline_rows = group[group["optimizer_name"].isin(NON_SQUARE_BASELINES)]
        ablation_rows = group[group["optimizer_name"].isin(RELEVANT_ABLATIONS)]
        if full_rows.empty or baseline_rows.empty:
            status = "Inconclusive"
            reason = "Missing SQUARETune full or baseline runs."
        else:
            full_score = float(full_rows["cost_adjusted_improvement"].mean())
            best_baseline = float(baseline_rows.groupby("optimizer_name")["cost_adjusted_improvement"].mean().max())
            best_ablation = (
                float(ablation_rows.groupby("optimizer_name")["cost_adjusted_improvement"].mean().max())
                if not ablation_rows.empty
                else None
            )
            if full_score > best_baseline + 0.01 and (best_ablation is None or full_score > best_ablation + 0.003):
                status = "External signal supported"
                reason = "SQUARETune full beat the best non-SQUARE baseline and relevant ablations on cost-adjusted improvement."
            elif full_score > best_baseline + 0.005:
                status = "Candidate external signal"
                reason = "SQUARETune full beat baselines, but ablation or stability evidence is incomplete."
            elif full_score <= best_baseline:
                status = "Refused"
                reason = "A non-SQUARE baseline matched or beat SQUARETune full."
            else:
                status = "Inconclusive"
                reason = "The transfer signal is mixed or below threshold."
    return {
        "scenario_family": scenario_family,
        "certificate_type": "External Transfer Certificate",
        "status": status,
        "reason": reason,
        "calibration_reference": calibration,
        "license_status": license_status,
        "budget_parity": _budget_parity(group),
        "source_appropriate": source_appropriate,
        "source_datasets": sorted(set(group.get("source_datasets", pd.Series(dtype=str)).dropna().astype(str))),
        "caveats": [
            CAUTION,
            "External transfer smoke results are separate from synthetic mechanism verification.",
        ],
    }


def write_external_certificates(project_root: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    output_dir = project_root / "certificates" / "square_tune" / "external_transfer" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _calibration_reference(project_root)
    certs = []
    for scenario, group in metrics.groupby("scenario_family") if not metrics.empty else []:
        cert = certificate_for_scenario(str(scenario), group, calibration=calibration)
        scenario_dir = output_dir / str(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        write_json(scenario_dir / "certificate.json", cert)
        lines = [
            f"# SQUARETune External Transfer Certificate: {scenario}",
            "",
            f"Status: **{cert['status']}**",
            "",
            cert["reason"],
            "",
            f"- Calibration gate status: `{calibration.get('calibration_global_gate_status')}`",
            f"- Budget parity: `{cert['budget_parity']}`",
            f"- License status: `{cert['license_status']}`",
            "",
            "## Caveats",
            "",
        ]
        lines.extend(f"- {item}" for item in cert["caveats"])
        write_text(scenario_dir / "certificate.md", "\n".join(lines) + "\n")
        if not group.empty:
            group.to_parquet(scenario_dir / "comparison_table.parquet", index=False)
            group[
                [
                    "optimizer_name",
                    "seed",
                    "response_surface_evaluations",
                    "actual_response_surface_evaluations",
                    "simulated_gpu_hours",
                ]
            ].to_parquet(scenario_dir / "budget_table.parquet", index=False)
        write_json(scenario_dir / "license_summary.json", {"license_status": cert["license_status"]})
        certs.append(cert)
    index = {
        "experiment_id": experiment_id,
        "certificate_type": "External Transfer Certificate Index",
        "calibration_reference": calibration,
        "certificate_count": len(certs),
        "certificates": certs,
    }
    write_json(output_dir / "certificate_index.json", index)
    lines = [
        f"# SQUARETune External Transfer Certificate Index: {experiment_id}",
        "",
        CAUTION,
        "",
        "| Scenario | Status |",
        "|---|---|",
    ]
    for cert in certs:
        lines.append(f"| {cert['scenario_family']} | {cert['status']} |")
    write_text(output_dir / "certificate_index.md", "\n".join(lines) + "\n")
    return index
