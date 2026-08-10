from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.reporting.calibration import CLASSICAL_BASELINES, evaluate_calibration_gates
from square_sim.utils.files import write_json, write_text

MECHANISM_REQUIREMENTS = {
    "local_snapshot_selection": ["square_tune_no_snapshot", "square_tune_global_only"],
    "branch_forking": ["square_tune_no_fork"],
    "nonlinear_rollout": ["square_tune_linear_rollout"],
    "merge_reintegration": ["square_tune_no_merge"],
    "memory_preservation": ["square_tune_no_memory"],
    "feedback_adaptation": ["square_tune_no_feedback"],
    "regression_awareness": ["square_tune_no_regression_sensor"],
    "cost_awareness": ["square_tune_no_cost_sensor"],
}


def _mean_metric(df: pd.DataFrame, optimizer: str, metric: str) -> float | None:
    rows = df[df["optimizer_name"] == optimizer]
    return None if rows.empty else float(rows[metric].mean())


def _support_table(df: pd.DataFrame) -> dict[str, str]:
    full = _mean_metric(df, "square_tune_full", "final_utility")
    support: dict[str, str] = {}
    if full is None:
        return {key: "inconclusive" for key in MECHANISM_REQUIREMENTS}
    for mechanism, ablations in MECHANISM_REQUIREMENTS.items():
        ablation_scores = [_mean_metric(df, name, "final_utility") for name in ablations]
        available = [score for score in ablation_scores if score is not None]
        if not available:
            support[mechanism] = "inconclusive"
        elif all(full > score + 0.005 for score in available):
            support[mechanism] = "supported"
        else:
            support[mechanism] = "not_supported"
    return support


def certificate_for_dataset(
    dataset_key: str,
    df: pd.DataFrame,
    calibration_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control = str(df["control_type"].iloc[0]) if "control_type" in df and not df.empty else "unknown"
    full = _mean_metric(df, "square_tune_full", "final_utility")
    baseline_values = [_mean_metric(df, optimizer, "final_utility") for optimizer in CLASSICAL_BASELINES]
    best_baseline = max([v for v in baseline_values if v is not None], default=None)
    support = _support_table(df)
    if control == "refusal_control":
        status = "Refusal control passed"
        if full is not None and best_baseline is not None and full > best_baseline + 0.05:
            status = "Refusal control failed"
    elif control == "classical_control":
        status = "Classical control passed"
        if full is not None and best_baseline is not None and full > best_baseline + 0.03:
            status = "Classical control failed"
    elif full is None or best_baseline is None:
        status = "Inconclusive"
    elif full > best_baseline + 0.01 and any(v == "supported" for v in support.values()):
        status = "Supported"
    elif full <= best_baseline:
        status = "Not supported"
    else:
        status = "Inconclusive"
    gate_payload = calibration_gates or {"global_status": "not_applicable", "gates": []}
    failed_gates = set(gate_payload.get("failed_gates", []))
    if status == "Supported" and failed_gates:
        if "budget_parity" in failed_gates:
            status = "Budget confounded"
        elif "oracle_leakage" in failed_gates:
            status = "Oracle/leakage invalid"
        else:
            status = "Inconclusive pending calibration"
    if "merge_required" in failed_gates:
        support["merge_reintegration"] = "not_supported"
    if "memory_required" in failed_gates:
        support["memory_preservation"] = "not_supported"
    if "regression_awareness" in failed_gates:
        support["regression_awareness"] = "not_supported"
    if "cost_awareness" in failed_gates:
        support["cost_awareness"] = "not_supported"
    return {
        "dataset_key": dataset_key,
        "status": status,
        "full_final_utility": full,
        "best_baseline_final_utility": best_baseline,
        "mechanism_support": support,
        "calibration_gates": gate_payload,
        "caveats": [
            "Mechanism verification certificate only for synthetic diagnostics.",
            "Not a physical hardware claim.",
            "External benchmark claims require separate external benchmark certificates.",
        ],
    }


def write_certificates(output_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    certificates = []
    gate_payload = evaluate_calibration_gates(metrics) if not metrics.empty else {"global_status": "warning", "gates": [], "failed_gates": []}
    for dataset_key, group in metrics.groupby("dataset_key") if not metrics.empty else []:
        cert = certificate_for_dataset(str(dataset_key), group, gate_payload)
        cert_dir = output_dir / str(dataset_key)
        cert_dir.mkdir(parents=True, exist_ok=True)
        write_json(cert_dir / "certificate.json", cert)
        lines = [
            f"# SQUARETune Mechanism Verification Certificate: {dataset_key}",
            "",
            f"Status: **{cert['status']}**",
            "",
            f"- SQUARETune full utility: `{cert['full_final_utility']}`",
            f"- Best baseline utility: `{cert['best_baseline_final_utility']}`",
            "",
            "## Mechanism Support",
            "",
        ]
        for key, value in cert["mechanism_support"].items():
            lines.append(f"- {key}: `{value}`")
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {item}" for item in cert["caveats"])
        lines.extend(["", "## Calibration Gates", ""])
        lines.append(f"- Global status: `{gate_payload.get('global_status')}`")
        for gate in gate_payload.get("gates", []):
            lines.append(f"- {gate['gate_name']}: `{gate['status']}`")
        write_text(cert_dir / "certificate.md", "\n".join(lines) + "\n")
        certificates.append(cert)
    index = {
        "experiment_id": experiment_id,
        "certificate_count": len(certificates),
        "calibration_gates": gate_payload,
        "certificates": certificates,
    }
    write_json(output_dir / "certificate_index.json", index)
    index_lines = [f"# SQUARETune Certificate Index: {experiment_id}", "", "| Dataset | Status |", "|---|---|"]
    for cert in certificates:
        index_lines.append(f"| {cert['dataset_key']} | {cert['status']} |")
    write_text(output_dir / "certificate_index.md", "\n".join(index_lines) + "\n")
    if certificates:
        pd.DataFrame(
            [
                {
                    "dataset_key": cert["dataset_key"],
                    "status": cert["status"],
                    "full_final_utility": cert["full_final_utility"],
                    "best_baseline_final_utility": cert["best_baseline_final_utility"],
                    "calibration_global_status": gate_payload.get("global_status"),
                }
                for cert in certificates
            ]
        ).to_parquet(output_dir / "certificate_index.parquet", index=False)
    return index
