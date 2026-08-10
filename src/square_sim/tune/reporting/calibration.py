from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import read_json, write_json, write_text


@dataclass(frozen=True)
class CalibrationGateResult:
    gate_name: str
    status: str
    evidence: dict[str, Any]
    failure_reason: str | None
    affected_certificates: list[str]
    required_action: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _mean(df: pd.DataFrame, optimizer: str, metric: str) -> float | None:
    rows = df[df["optimizer_name"] == optimizer]
    return None if rows.empty or metric not in rows else float(rows[metric].mean())


def _best(df: pd.DataFrame, optimizers: list[str], metric: str) -> tuple[str | None, float | None]:
    values = [(opt, _mean(df, opt, metric)) for opt in optimizers]
    available = [(opt, value) for opt, value in values if value is not None]
    if not available:
        return None, None
    return max(available, key=lambda item: item[1])


CLASSICAL_BASELINES = [
    "linear_utility_optimizer",
    "ridge_utility_optimizer",
    "coordinate_descent",
    "greedy_oracle_feature_baseline",
    "grid_search",
    "greedy_eval_improvement",
    "greedy_regression_aware",
    "random_search",
]


def evaluate_calibration_gates(metrics: pd.DataFrame) -> dict[str, Any]:
    gates: list[CalibrationGateResult] = []
    datasets = sorted(metrics["dataset_key"].unique()) if not metrics.empty else []

    random_rows = metrics[metrics["dataset_key"] == "synthetic_llm_random_label"]
    full_random = _mean(random_rows, "square_tune_full", "final_utility")
    best_random_name, best_random = _best(random_rows, CLASSICAL_BASELINES, "final_utility")
    random_pass = full_random is not None and best_random is not None and full_random <= best_random + 0.02
    gates.append(
        CalibrationGateResult(
            "random_label_refusal",
            "passed" if random_pass else "failed",
            {"square_tune_full": full_random, "best_baseline": best_random, "best_baseline_name": best_random_name},
            None if random_pass else "Random-label refusal control allowed a SQUARETune advantage-like result.",
            datasets,
            "Invalidate positive mechanism certificates until random-label control passes.",
        )
    )

    linear_rows = metrics[metrics["dataset_key"] == "synthetic_llm_linear_control"]
    full_linear = _mean(linear_rows, "square_tune_full", "final_utility")
    best_linear_name, best_linear = _best(linear_rows, CLASSICAL_BASELINES, "final_utility")
    linear_pass = full_linear is not None and best_linear is not None and best_linear >= full_linear - 0.01
    gates.append(
        CalibrationGateResult(
            "linear_control_classical_sanity",
            "passed" if linear_pass else "failed",
            {"square_tune_full": full_linear, "best_classical": best_linear, "best_classical_name": best_linear_name},
            None if linear_pass else "Classical linear-control task was not won or tied by a classical baseline.",
            datasets,
            "Downgrade positive certificates to Inconclusive pending calibration.",
        )
    )

    merge_rows = metrics[metrics["dataset_key"] == "synthetic_llm_merge_required"]
    full_merge = _mean(merge_rows, "square_tune_full", "final_utility")
    no_merge = _mean(merge_rows, "square_tune_no_merge", "final_utility")
    merge_pass = full_merge is not None and no_merge is not None and full_merge > no_merge + 0.005
    gates.append(
        CalibrationGateResult(
            "merge_required",
            "passed" if merge_pass else "failed",
            {"square_tune_full": full_merge, "square_tune_no_merge": no_merge},
            None if merge_pass else "Full SQUARETune did not beat the no-merge ablation on merge-required task.",
            ["synthetic_llm_merge_required"],
            "Mark merge_reintegration not supported.",
        )
    )

    memory_rows = metrics[metrics["dataset_key"].isin(["synthetic_llm_repeated_regression_memory", "synthetic_llm_curriculum_order"])]
    full_memory = _mean(memory_rows, "square_tune_full", "preserved_known_good_score")
    no_memory_score = _mean(memory_rows, "square_tune_no_memory", "preserved_known_good_score")
    repeated_bad_full = _mean(memory_rows, "square_tune_full", "repeated_bad_action_count")
    repeated_bad_no_memory = _mean(memory_rows, "square_tune_no_memory", "repeated_bad_action_count")
    memory_pass = (
        full_memory is not None
        and no_memory_score is not None
        and full_memory >= no_memory_score
        and repeated_bad_full is not None
        and repeated_bad_no_memory is not None
        and repeated_bad_full < repeated_bad_no_memory
    )
    gates.append(
        CalibrationGateResult(
            "memory_required",
            "passed" if memory_pass else "failed",
            {
                "full_preserved_known_good": full_memory,
                "no_memory_preserved_known_good": no_memory_score,
                "full_repeated_bad": repeated_bad_full,
                "no_memory_repeated_bad": repeated_bad_no_memory,
            },
            None if memory_pass else "Memory-enabled optimizer did not clearly reduce repeated damaging actions.",
            ["synthetic_llm_repeated_regression_memory", "synthetic_llm_curriculum_order"],
            "Mark memory_preservation not supported.",
        )
    )

    regression_rows = metrics[metrics["dataset_key"].isin(["synthetic_llm_regression_veto", "synthetic_llm_data_poison_regression", "synthetic_llm_prompt_regression"])]
    full_reg = _mean(regression_rows, "square_tune_full", "protected_utility")
    no_reg = _mean(regression_rows, "square_tune_no_regression_sensor", "protected_utility")
    full_reg_count = _mean(regression_rows, "square_tune_full", "regression_count")
    no_reg_count = _mean(regression_rows, "square_tune_no_regression_sensor", "regression_count")
    regression_pass = full_reg is not None and no_reg is not None and full_reg > no_reg + 0.005 and (full_reg_count or 0) <= (no_reg_count or 0)
    gates.append(
        CalibrationGateResult(
            "regression_awareness",
            "passed" if regression_pass else "failed",
            {
                "full_protected_utility": full_reg,
                "no_regression_sensor_protected_utility": no_reg,
                "full_regressions": full_reg_count,
                "no_regression_sensor_regressions": no_reg_count,
            },
            None if regression_pass else "Regression-aware optimizer did not beat no-regression-sensor on protected utility.",
            ["synthetic_llm_regression_veto", "synthetic_llm_data_poison_regression", "synthetic_llm_prompt_regression"],
            "Mark regression_awareness not supported and flag commercial regression risk.",
        )
    )

    cost_rows = metrics[metrics["dataset_key"].isin(["synthetic_llm_cost_tradeoff", "synthetic_llm_adapter_tradeoff", "synthetic_llm_rag_policy_conflict"])]
    full_cost = _mean(cost_rows, "square_tune_full", "cost_adjusted_improvement")
    no_cost = _mean(cost_rows, "square_tune_no_cost_sensor", "cost_adjusted_improvement")
    cost_pass = full_cost is not None and no_cost is not None and full_cost > no_cost + 0.005
    gates.append(
        CalibrationGateResult(
            "cost_awareness",
            "passed" if cost_pass else "failed",
            {"full_cost_adjusted": full_cost, "no_cost_sensor_cost_adjusted": no_cost},
            None if cost_pass else "Cost-aware optimizer did not beat no-cost-sensor on cost-adjusted utility.",
            ["synthetic_llm_cost_tradeoff", "synthetic_llm_adapter_tradeoff", "synthetic_llm_rag_policy_conflict"],
            "Mark cost_awareness not supported.",
        )
    )

    budget_cols = ["response_surface_evaluations", "candidate_actions_scored"]
    budget_pass = all(col in metrics.columns for col in budget_cols)
    budget_evidence: dict[str, Any] = {"required_columns_present": budget_pass}
    if budget_pass and not metrics.empty:
        grouped = metrics.groupby("optimizer_name")[budget_cols].mean()
        budget_evidence["mean_usage_by_optimizer"] = grouped.round(4).to_dict(orient="index")
        spread = float(grouped["response_surface_evaluations"].max() - grouped["response_surface_evaluations"].min())
        budget_evidence["response_eval_spread"] = spread
        budget_pass = spread <= max(1.0, 0.1 * float(grouped["response_surface_evaluations"].max()))
    gates.append(
        CalibrationGateResult(
            "budget_parity",
            "passed" if budget_pass else "failed",
            budget_evidence,
            None if budget_pass else "Optimizer budget usage is unequal or not recorded.",
            datasets,
            "Downgrade affected certificates to Budget confounded.",
        )
    )

    leakage_pass = True
    leakage_evidence = {"latent_columns_used": []}
    if "latent_columns_used" in metrics.columns:
        used = sorted(
            {str(v) for v in metrics["latent_columns_used"].dropna() if str(v) not in {"", "[]"}}
        )
        leakage_evidence["latent_columns_used"] = used
        leakage_pass = not used
    gates.append(
        CalibrationGateResult(
            "oracle_leakage",
            "passed" if leakage_pass else "failed",
            leakage_evidence,
            None if leakage_pass else "Fair optimizer used latent generator columns or oracle internals.",
            datasets,
            "Mark affected certificates Oracle/leakage invalid.",
        )
    )

    gate_dicts = [gate.to_dict() for gate in gates]
    required_failed = [gate for gate in gate_dicts if gate["status"] == "failed"]
    global_status = "passed" if not required_failed else "failed"
    return {
        "global_status": global_status,
        "gates": gate_dicts,
        "failed_gates": [gate["gate_name"] for gate in required_failed],
    }


def write_calibration_reports(output_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_payload = evaluate_calibration_gates(metrics)
    write_json(output_dir / "calibration_gate_report.json", gate_payload)
    lines = [f"# SQUARETune Calibration Gate Report: {experiment_id}", "", f"Global status: **{gate_payload['global_status']}**", ""]
    lines.extend(["| Gate | Status | Evidence | Required action |", "|---|---|---|---|"])
    for gate in gate_payload["gates"]:
        lines.append(
            f"| {gate['gate_name']} | {gate['status']} | `{gate['evidence']}` | {gate['required_action']} |"
        )
    write_text(output_dir / "calibration_gate_report.md", "\n".join(lines) + "\n")

    if not metrics.empty:
        metric_cols = [
            "dataset_key",
            "optimizer_name",
            "seed",
            "final_utility",
            "protected_utility",
            "cost_adjusted_improvement",
            "regression_count",
            "response_surface_evaluations",
            "candidate_actions_scored",
            "simulated_gpu_hours",
        ]
        available = [col for col in metric_cols if col in metrics.columns]
        metrics[available].to_parquet(output_dir / "budget_parity_table.parquet", index=False)
        control_datasets = [
            "synthetic_llm_random_label",
            "synthetic_llm_linear_control",
            "synthetic_llm_merge_required",
            "synthetic_llm_repeated_regression_memory",
            "synthetic_llm_regression_veto",
            "synthetic_llm_cost_tradeoff",
        ]
        controls = metrics[metrics["dataset_key"].isin(control_datasets)]
        if not controls.empty:
            control_summary = (
                controls.groupby(["dataset_key", "optimizer_name"], as_index=False)
                .agg(
                    final_utility=("final_utility", "mean"),
                    protected_utility=("protected_utility", "mean"),
                    cost_adjusted_improvement=("cost_adjusted_improvement", "mean"),
                    regression_count=("regression_count", "mean"),
                )
                .sort_values(["dataset_key", "final_utility"], ascending=[True, False])
            )
            control_summary.to_parquet(output_dir / "control_task_report.parquet", index=False)
            control_summary.to_json(output_dir / "control_task_report.json", orient="records", indent=2)
            control_lines = [
                f"# SQUARETune Calibration Control Task Report: {experiment_id}",
                "",
                "| Dataset | Optimizer | Final Utility | Protected Utility | Cost-Adjusted | Regressions |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for row in control_summary.to_dict(orient="records"):
                control_lines.append(
                    f"| {row['dataset_key']} | {row['optimizer_name']} | {row['final_utility']:.4f} | "
                    f"{row['protected_utility']:.4f} | {row['cost_adjusted_improvement']:.4f} | "
                    f"{row['regression_count']:.2f} |"
                )
            write_text(output_dir / "control_task_report.md", "\n".join(control_lines) + "\n")
        ablation_pairs = [
            ("square_tune_full", "square_tune_no_snapshot"),
            ("square_tune_full", "square_tune_no_fork"),
            ("square_tune_full", "square_tune_linear_rollout"),
            ("square_tune_full", "square_tune_no_merge"),
            ("square_tune_full", "square_tune_no_memory"),
            ("square_tune_full", "square_tune_no_regression_sensor"),
            ("square_tune_full", "square_tune_no_cost_sensor"),
        ]
        rows: list[dict[str, Any]] = []
        for dataset_key, group in metrics.groupby("dataset_key"):
            for model_a, model_b in ablation_pairs:
                score_a = _mean(group, model_a, "final_utility")
                score_b = _mean(group, model_b, "final_utility")
                if score_a is None or score_b is None:
                    continue
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "model_a": model_a,
                        "model_b": model_b,
                        "final_utility_a": score_a,
                        "final_utility_b": score_b,
                        "delta": score_a - score_b,
                        "status": "supports_a" if score_a > score_b + 0.005 else "inconclusive_or_supports_b",
                    }
                )
        if rows:
            ablation_df = pd.DataFrame(rows)
            ablation_df.to_parquet(output_dir / "ablation_load_bearing_report.parquet", index=False)
            ablation_df.to_json(output_dir / "ablation_load_bearing_report.json", orient="records", indent=2)
            lines_ab = [
                f"# SQUARETune Ablation Load-Bearing Report: {experiment_id}",
                "",
                "| Dataset | Comparison | Delta | Status |",
                "|---|---|---:|---|",
            ]
            for row in rows:
                lines_ab.append(
                    f"| {row['dataset_key']} | {row['model_a']} vs {row['model_b']} | "
                    f"{row['delta']:.4f} | {row['status']} |"
                )
            write_text(output_dir / "ablation_load_bearing_report.md", "\n".join(lines_ab) + "\n")
    summary = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "Calibration passed" if gate_payload["global_status"] == "passed" else "Calibration failed",
        "external_peft_status": "blocked" if gate_payload["global_status"] != "passed" else "allowed for engineering smoke tests with no scientific claims until external benchmarks pass",
        "failed_gates": gate_payload["failed_gates"],
    }
    write_json(output_dir / "calibration_v2_summary.json", summary)
    write_text(
        output_dir / "calibration_v2_summary.md",
        "\n".join(
            [
                f"# SQUARETune Calibration v2 Summary: {experiment_id}",
                "",
                f"Overall status: **{summary['overall_status']}**",
                f"External/PEFT status: **{summary['external_peft_status']}**",
                "",
                "Failed gates:",
                *(f"- `{gate}`" for gate in summary["failed_gates"]),
            ]
        )
        + "\n",
    )
    budget_lines = [f"# SQUARETune Budget Parity Report: {experiment_id}", ""]
    if not metrics.empty and all(
        col in metrics.columns for col in ["optimizer_name", "response_surface_evaluations"]
    ):
        if "actual_response_surface_evaluations" not in metrics.columns:
            metrics = metrics.copy()
            metrics["actual_response_surface_evaluations"] = metrics[
                "response_surface_evaluations"
            ]
        agg_spec = {
            "response_surface_evaluations": ("response_surface_evaluations", "mean"),
            "actual_response_surface_evaluations": (
                "actual_response_surface_evaluations",
                "mean",
            ),
            "final_utility": ("final_utility", "mean"),
        }
        if "simulated_gpu_hours" in metrics.columns:
            agg_spec["simulated_gpu_hours"] = ("simulated_gpu_hours", "mean")
        grouped = metrics.groupby("optimizer_name", as_index=False).agg(**agg_spec)
        if "simulated_gpu_hours" not in grouped.columns:
            grouped["simulated_gpu_hours"] = 0.0
        grouped = grouped.sort_values("optimizer_name")
        budget_lines.extend(["| Optimizer | Normalized Evals | Actual Evals | GPU-Hour Proxy | Utility |", "|---|---:|---:|---:|---:|"])
        for row in grouped.to_dict(orient="records"):
            budget_lines.append(
                f"| {row['optimizer_name']} | {row['response_surface_evaluations']:.2f} | "
                f"{row['actual_response_surface_evaluations']:.2f} | {row['simulated_gpu_hours']:.4f} | "
                f"{row['final_utility']:.4f} |"
            )
    else:
        budget_lines.append("No budget metrics were available.")
    write_text(output_dir / "budget_parity_report.md", "\n".join(budget_lines) + "\n")
    return {"experiment_id": experiment_id, "output_dir": str(output_dir), **summary}


def summarize_prior_experiment(experiment_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if not experiment_path.exists():
        payload = {
            "status": "missing",
            "experiment_path": str(experiment_path),
            "recommended_calibration_plan": [
                "Run Calibration v2 controls before external or PEFT validation.",
                "Require random-label and linear-control gates to pass.",
            ],
        }
    else:
        metrics_path = experiment_path / "metrics.parquet"
        summary_path = experiment_path / "experiment_summary.json"
        metrics = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
        gate_payload = evaluate_calibration_gates(metrics) if not metrics.empty else {"global_status": "warning", "gates": [], "failed_gates": []}
        summary = read_json(summary_path) if summary_path.exists() else {}
        compact_summary = {
            key: value
            for key, value in summary.items()
            if key not in {"results"}
        }
        payload = {
            "status": "ok",
            "experiment_path": str(experiment_path),
            "summary": compact_summary,
            "mechanisms_tested": sorted(metrics["dataset_key"].unique()) if not metrics.empty else [],
            "seeds": sorted(int(v) for v in metrics["seed"].unique()) if "seed" in metrics else [],
            "optimizers": sorted(metrics["optimizer_name"].unique()) if "optimizer_name" in metrics else [],
            "calibration_gates": gate_payload,
            "recommended_calibration_plan": [
                "Regenerate Calibration v2 synthetic controls.",
                "Run controls-only matrix and stop if required gates fail.",
                "Only run full calibration after gates pass or with explicit diagnostic intent.",
            ],
        }
    write_json(output_dir / f"prior_result_review_{timestamp}.json", payload)
    lines = [f"# Prior SQUARETune Result Review: {timestamp}", "", f"Status: `{payload['status']}`", ""]
    if payload["status"] == "ok":
        lines.append(f"Experiment path: `{payload['experiment_path']}`")
        lines.append("")
        lines.append("Failed gates:")
        lines.extend(f"- `{gate}`" for gate in payload["calibration_gates"].get("failed_gates", []))
    else:
        lines.append(f"Prior artifacts were not found at `{experiment_path}`.")
    lines.append("")
    lines.append("Recommended calibration plan:")
    lines.extend(f"- {item}" for item in payload["recommended_calibration_plan"])
    write_text(output_dir / f"prior_result_review_{timestamp}.md", "\n".join(lines) + "\n")
    return payload
