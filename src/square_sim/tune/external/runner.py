from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from square_sim.config import Settings
from square_sim.tune.config import DEFAULT_OBJECTIVE_WEIGHTS, TuneBudget
from square_sim.tune.external.certificate import write_external_certificates
from square_sim.tune.external.paths import (
    external_certificates_root,
    external_reports_root,
    external_root,
    external_runs_root,
)
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.tune.external.reporting import (
    write_external_transfer_report,
    write_no_overwrite_audit,
)
from square_sim.tune.external.scenarios import compile_scenarios
from square_sim.tune.external.transfer_simulator import run_external_transfer_optimizer
from square_sim.tune.simulator.adaptive_compute import summarize_compute_gate
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager, unique_id


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw


def _expand_env_path(value: str | Path | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(os.path.expandvars(str(value))).expanduser()


def _scenario_manifests(root: Path, families: list[str]) -> list[Path]:
    manifests: list[Path] = []
    for family in families:
        candidates = sorted((root / family).glob("*/scenario_manifest.json"))
        if not candidates:
            continue
        latest = max(
            candidates,
            key=lambda path: str(read_json(path).get("generated_at_utc", "")),
        )
        manifests.append(latest)
    return manifests


def _completed_index(settings: Settings) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest_path in external_runs_root(settings).glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("status") == "succeeded" and manifest.get("run_fingerprint"):
            index[str(manifest["run_fingerprint"])] = {
                "run_id": manifest.get("run_id"),
                "run_manifest_path": str(manifest_path),
            }
    return index


def _budget_from_config(raw: dict[str, Any]) -> TuneBudget:
    return TuneBudget.from_config(raw.get("square_tune") or raw.get("budget"))


def _metrics_for_experiment(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows = []
    for manifest_path in external_runs_root(settings).glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "succeeded":
            continue
        metrics_path = Path(str(manifest.get("metrics_path")))
        if metrics_path.exists():
            metric = read_json(metrics_path)
            if isinstance(metric.get("budget_consumed"), dict):
                metric["budget_consumed_json"] = json.dumps(metric["budget_consumed"], sort_keys=True)
                metric.pop("budget_consumed", None)
            metric.update(
                {
                    "run_id": manifest.get("run_id"),
                    "scenario_family": manifest.get("scenario_family"),
                    "scenario_id": manifest.get("scenario_id"),
                    "source_datasets": ",".join(manifest.get("source_datasets", [])),
                    "source_license_status": manifest.get("source_license_status"),
                    "source_appropriate": manifest.get("source_appropriate"),
                    "scenario_warnings": ",".join(manifest.get("warnings", [])),
                    "optimizer_name": manifest.get("optimizer"),
                    "seed": manifest.get("seed"),
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def load_external_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    return _metrics_for_experiment(settings, experiment_id)


def _run_single(
    settings: Settings,
    *,
    experiment_id: str,
    config_path: Path,
    scenario_manifest_path: Path,
    seed: int,
    optimizer: str,
    budget: TuneBudget,
    objective_weights: dict[str, float],
    device: str,
    completed: dict[str, dict[str, Any]],
    resume: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    manifest = read_json(scenario_manifest_path)
    split_manifest_path = scenario_manifest_path.parent / "splits" / "split_manifest.json"
    train_path = scenario_manifest_path.parent / "splits" / "train.parquet"
    scenario_family = str(manifest["scenario_family"])
    scenario_id = str(manifest["scenario_id"])
    fingerprint = stable_hash(
        {
            "experiment_id": experiment_id,
            "scenario_id": scenario_id,
            "scenario_family": scenario_family,
            "seed": seed,
            "optimizer": optimizer,
            "budget": budget.to_dict(),
            "config_hash": sha256_file(config_path),
        },
        16,
    )
    if resume or skip_completed:
        existing = completed.get(fingerprint)
        if existing:
            return {"status": "skipped", "reason": "completed fingerprint exists", **existing}
    date = datetime.now(timezone.utc)
    run_id = f"{date:%Y%m%d-%H%M%S}-{scenario_family[:18]}-{optimizer[:18]}-{fingerprint[:8]}"
    output_dir = external_runs_root(settings) / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}" / run_id
    if output_dir.exists():
        raise FileExistsError(f"Run id collision: {run_id}")
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(output_dir, registry.protected_paths())
    manager.ensure_writable_path(output_dir)
    output_dir.mkdir(parents=True)
    result = run_external_transfer_optimizer(
        optimizer_name=optimizer,
        train_path=train_path,
        scenario_family=scenario_family,
        seed=seed,
        budget=budget,
        objective_weights=objective_weights,
    )
    metrics_path = output_dir / "metrics.json"
    trajectory_path = output_dir / "trajectory.parquet"
    branch_path = output_dir / "branch_diagnostics.parquet"
    final_policy_path = output_dir / "final_policy.json"
    result["trajectory"].to_parquet(trajectory_path, index=False)
    result["branch_diagnostics"].to_parquet(branch_path, index=False)
    adaptive_diagnostics = result.get("adaptive_diagnostics")
    adaptive_dir = output_dir / "adaptive_compute_diagnostics"
    adaptive_path = None
    adaptive_summary_path = None
    if isinstance(adaptive_diagnostics, pd.DataFrame) and not adaptive_diagnostics.empty:
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        adaptive_path = adaptive_dir / "compute_gate_decisions.parquet"
        adaptive_summary_path = adaptive_dir / "compute_gate_summary.json"
        adaptive_diagnostics.to_parquet(adaptive_path, index=False)
        summary_payload = summarize_compute_gate(adaptive_diagnostics.to_dict(orient="records"))
        write_json(adaptive_summary_path, summary_payload)
        decision_counts = adaptive_diagnostics["decision"].value_counts(normalize=True).to_dict()
        write_json(adaptive_dir / "decision_distribution.json", decision_counts)
        adaptive_diagnostics[adaptive_diagnostics["fork_invoked"]].to_parquet(
            adaptive_dir / "invoked_fork_cases.parquet", index=False
        )
        adaptive_diagnostics[~adaptive_diagnostics["fork_invoked"]].to_parquet(
            adaptive_dir / "suppressed_fork_cases.parquet", index=False
        )
        adaptive_diagnostics[["round_idx", "decision", "fork_roi", "realized_cost", "realized_utility_gain"]].to_parquet(
            adaptive_dir / "fork_roi.parquet", index=False
        )
        adaptive_diagnostics[["round_idx", "decision", "merge_roi", "realized_cost", "realized_utility_gain"]].to_parquet(
            adaptive_dir / "merge_roi.parquet", index=False
        )
        adaptive_diagnostics.to_parquet(adaptive_dir / "per_round_adaptive_trace.parquet", index=False)
        write_json(
            adaptive_dir / "budget_savings.json",
            {
                "simulated_gpu_hours": result["metrics"].get("simulated_gpu_hours"),
                "cost_saved_vs_full": summary_payload.get("cost_saved_vs_full"),
                "fork_invocation_rate": summary_payload.get("fork_invocation_rate"),
                "merge_invocation_rate": summary_payload.get("merge_invocation_rate"),
            },
        )
        write_text(
            adaptive_dir / "adaptive_compute_summary.md",
            "\n".join(
                [
                    f"# Adaptive Compute Diagnostics: {run_id}",
                    "",
                    "This run used conditional compute gating inside the SQUARETune external-transfer simulator.",
                    "",
                    f"- Fork invocation rate: `{summary_payload.get('fork_invocation_rate')}`",
                    f"- Merge invocation rate: `{summary_payload.get('merge_invocation_rate')}`",
                    f"- Positive fork ROI rate: `{summary_payload.get('positive_fork_roi_rate')}`",
                    f"- Degenerate behavior flag: `{summary_payload.get('degenerate_behavior_flag')}`",
                    "",
                    "Caveat: this is external-transfer simulation, not physical hardware validation or real fine-tuning proof.",
                ]
            )
            + "\n",
        )
    write_json(metrics_path, result["metrics"])
    write_json(final_policy_path, result["final_policy"])
    calibration_cert = settings.project_root / "certificates" / "square_tune" / "calibration" / "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd" / "certificate_index.json"
    run_manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_type": "square_tune_external_transfer_v1",
        "scenario_family": scenario_family,
        "source_datasets": manifest.get("source_datasets", []),
        "source_dataset_versions": manifest.get("source_dataset_versions", []),
        "source_license_status": ",".join(sorted(manifest.get("license_summary", {}).keys())) or "unknown",
        "source_appropriate": bool(manifest.get("source_appropriate", False)) and not manifest.get("warnings"),
        "scenario_warnings": manifest.get("warnings", []),
        "scenario_id": scenario_id,
        "split_id": f"{scenario_id}_seed_{manifest.get('split_seed')}",
        "split_manifest_path": str(split_manifest_path),
        "seed": seed,
        "optimizer": optimizer,
        "model_or_optimizer_name": optimizer,
        "ablation_flags": {"is_square_tune": optimizer.startswith("square_tune"), "variant": optimizer},
        "budget_config": budget.to_dict(),
        "budget_consumed": result["metrics"].get("budget_consumed", {}),
        "metrics_path": str(metrics_path),
        "trajectory_path": str(trajectory_path),
        "diagnostics_path": str(branch_path),
        "branch_diagnostics_path": str(branch_path),
        "adaptive_diagnostics_path": str(adaptive_path) if adaptive_path else None,
        "adaptive_compute_summary_path": str(adaptive_summary_path) if adaptive_summary_path else None,
        "final_policy_path": str(final_policy_path),
        "calibration_reference": {
            "calibration_experiment_id": "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd",
            "calibration_certificate_path": str(calibration_cert),
            "calibration_global_gate_status": "passed" if calibration_cert.exists() else "missing",
        },
        "protected_results_checked": True,
        "write_root": str(output_dir),
        "no_overwrite_status": "append_only",
        "code_git_commit": _git_commit(),
        "node_hostname": socket.gethostname(),
        "device": device,
        "started_at_utc": date.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "run_fingerprint": fingerprint,
        "caveats": [
            "External-transfer simulation over open/external examples.",
            "Not physical hardware validation or real PEFT validation.",
        ],
    }
    explanation = [
        f"# SQUARETune External Run: {run_id}",
        "",
        "This is an external-transfer simulation over open/external datasets.",
        "It does not prove SQUARE hardware, real fine-tuning performance, or commercial ROI.",
        "",
        f"- Scenario family: `{scenario_family}`",
        f"- Scenario id: `{scenario_id}`",
        f"- Optimizer: `{optimizer}`",
        f"- Seed: `{seed}`",
        f"- Final utility: `{result['metrics'].get('final_utility')}`",
        f"- Cost-adjusted improvement: `{result['metrics'].get('cost_adjusted_improvement')}`",
    ]
    if adaptive_summary_path:
        explanation.extend(
            [
                f"- Adaptive diagnostics: `{adaptive_summary_path}`",
                f"- Fork invocation rate: `{result['metrics'].get('fork_invocation_rate')}`",
                f"- Merge invocation rate: `{result['metrics'].get('merge_invocation_rate')}`",
            ]
        )
    explanation_path = output_dir / "explanation.md"
    write_text(explanation_path, "\n".join(explanation) + "\n")
    run_manifest["report_path"] = str(explanation_path)
    write_json(output_dir / "run_manifest.json", run_manifest)
    return {"status": "succeeded", "run_id": run_id, "run_manifest_path": str(output_dir / "run_manifest.json")}


def run_external_matrix(
    settings: Settings,
    config_path: Path,
    *,
    device: str = "cpu",
    resume: bool = True,
    skip_completed: bool = True,
    smoke: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    raw = _load_config(config_path)
    scenario_root = _expand_env_path(raw.get("scenario_root"), external_root(settings) / "scenarios")
    families = list(raw.get("scenario_families", []))
    seeds = [int(seed) for seed in raw.get("seeds", [101])]
    optimizers = list(raw.get("optimizers", []))
    if smoke:
        seeds = seeds[:1]
        optimizers = optimizers[: max(1, min(len(optimizers), 6))]
    scenario_paths = _scenario_manifests(scenario_root, families)
    prefix = f"square_tune_external_v1_{'smoke' if smoke else str(raw.get('experiment_name', 'minimal'))}"
    experiment_id = unique_id(prefix, {"config": str(config_path), "families": families, "smoke": smoke})
    reports_dir = external_reports_root(settings) / experiment_id
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(external_reports_root(settings), registry.protected_paths())
    manager.ensure_writable_path(reports_dir)
    reports_dir.mkdir(parents=True)
    budget = _budget_from_config(raw)
    objective_weights = dict(raw.get("square_tune", {}).get("objective_weights", DEFAULT_OBJECTIVE_WEIGHTS))
    planned = [(path, seed, opt) for path in scenario_paths for seed in seeds for opt in optimizers]
    if max_runs is not None:
        planned = planned[:max_runs]
    completed = _completed_index(settings)
    results = []
    if not scenario_paths:
        results.append({"status": "failed", "error": f"No scenarios found under {scenario_root}. Run compile-scenarios first."})
    for scenario_manifest_path, seed, optimizer in planned:
        try:
            results.append(
                _run_single(
                    settings,
                    experiment_id=experiment_id,
                    config_path=config_path,
                    scenario_manifest_path=scenario_manifest_path,
                    seed=seed,
                    optimizer=optimizer,
                    budget=budget,
                    objective_weights=objective_weights,
                    device=device,
                    completed=completed,
                    resume=resume,
                    skip_completed=skip_completed,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "scenario_manifest_path": str(scenario_manifest_path),
                    "seed": seed,
                    "optimizer": optimizer,
                    "error": str(exc),
                }
            )
    metrics = _metrics_for_experiment(settings, experiment_id)
    if not metrics.empty:
        metrics.to_parquet(reports_dir / "metrics.parquet", index=False)
        metrics.to_csv(reports_dir / "metrics.csv", index=False)
    summary = {
        "experiment_id": experiment_id,
        "experiment_type": "square_tune_external_transfer_v1",
        "config_path": str(config_path),
        "scenario_root": str(scenario_root),
        "total_planned": len(planned),
        "succeeded": sum(row["status"] == "succeeded" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
        "reports_dir": str(reports_dir),
    }
    write_json(reports_dir / "experiment_summary.json", summary)
    write_text(
        reports_dir / "smoke_report.md" if smoke else reports_dir / "run_report.md",
        "\n".join(
            [
                f"# SQUARETune External {'Smoke' if smoke else 'Run'}: {experiment_id}",
                "",
                f"- Planned: `{summary['total_planned']}`",
                f"- Succeeded: `{summary['succeeded']}`",
                f"- Skipped: `{summary['skipped']}`",
                f"- Failed: `{summary['failed']}`",
            ]
        )
        + "\n",
    )
    no_overwrite = write_no_overwrite_audit(
        reports_dir,
        experiment_id=experiment_id,
        protected_paths=[str(path) for path in registry.protected_paths()],
        write_roots=[str(reports_dir), str(external_runs_root(settings)), str(external_certificates_root(settings))],
    )
    write_external_transfer_report(reports_dir, experiment_id=experiment_id, metrics=metrics, summary=summary)
    return {**summary, "no_overwrite_audit": no_overwrite}


def generate_external_report(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = _metrics_for_experiment(settings, experiment_id)
    output_dir = external_reports_root(settings) / experiment_id
    summary_path = output_dir / "experiment_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"experiment_id": experiment_id}
    return write_external_transfer_report(output_dir, experiment_id=experiment_id, metrics=metrics, summary=summary)


def generate_external_certificate(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = _metrics_for_experiment(settings, experiment_id)
    return write_external_certificates(settings.project_root, experiment_id, metrics)


def run_all_external_v1(
    settings: Settings,
    *,
    protect_prior: bool,
    acquire: bool,
    compile: bool,
    run_smoke_first: bool,
    run_full_if_smoke_passes: bool,
    generate_reports_flag: bool,
    generate_certificates: bool,
    resume: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    from square_sim.tune.external.acquire import acquire_external

    registry = ProtectedResultsRegistry(settings)
    protection = registry.protect_defaults() if protect_prior else registry.load()
    acquisition = (
        acquire_external(
            Path("configs/tune/external_transfer/datasets_minimal_v1.yaml"),
            output_root=external_root(settings),
        )
        if acquire
        else None
    )
    scenario_compilation = (
        compile_scenarios(Path("configs/tune/external_transfer/external_transfer_v1_minimal.yaml"))
        if compile
        else None
    )
    smoke_summary = (
        run_external_matrix(
            settings,
            Path("configs/tune/external_transfer/external_transfer_v1_smoke.yaml"),
            resume=resume,
            skip_completed=skip_completed,
            smoke=True,
        )
        if run_smoke_first
        else None
    )
    full_summary = None
    smoke_passed = not smoke_summary or int(smoke_summary.get("failed", 0)) == 0
    if run_full_if_smoke_passes and smoke_passed:
        full_summary = run_external_matrix(
            settings,
            Path("configs/tune/external_transfer/external_transfer_v1_minimal.yaml"),
            resume=resume,
            skip_completed=skip_completed,
        )
    if generate_reports_flag:
        for summary in [smoke_summary, full_summary]:
            if summary:
                generate_external_report(settings, str(summary["experiment_id"]))
    if generate_certificates:
        for summary in [smoke_summary, full_summary]:
            if summary:
                generate_external_certificate(settings, str(summary["experiment_id"]))
    return {
        "status": "completed",
        "protection": protection,
        "acquisition": acquisition,
        "scenario_compilation": scenario_compilation,
        "smoke_summary": smoke_summary,
        "full_summary": full_summary,
    }
