from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.adaptive_arch.certificate import write_certificates
from square_sim.adaptive_arch.config import (
    SYSTEM_TO_OPTIMIZER,
    TASK_TO_MECHANISM,
    AdaptiveArchConfig,
)
from square_sim.adaptive_arch.diagnostics import diagnose_experiment, write_run_diagnostics
from square_sim.adaptive_arch.generators import generate_suite
from square_sim.adaptive_arch.metrics import architecture_metrics
from square_sim.adaptive_arch.reporting import write_no_overwrite_audit, write_report
from square_sim.config import Settings
from square_sim.tune.config import DEFAULT_OBJECTIVE_WEIGHTS, TuneBudget
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.tune.simulator.square_tune_optimizer import run_optimizer
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager, unique_id


def reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "square_adaptive_arch" / "v1"


def runs_root(settings: Settings) -> Path:
    return settings.project_root / "square_adaptive_arch_runs"


def certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "square_adaptive_arch" / "v1"


def default_dataset_root(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "synthetic" / "square_adaptive_arch_v1"


def _expand_path(value: str | None, fallback: Path) -> Path:
    return Path(os.path.expandvars(value)).expanduser() if value else fallback


def _latest_dataset(root: Path, task: str, seed: int) -> Path | None:
    candidates = []
    for manifest in (root / task).glob("*/generator_manifest.json"):
        payload = read_json(manifest)
        if int(payload.get("seed", -1)) == int(seed):
            candidates.append(manifest.parent)
    return max(candidates, key=lambda p: p.name) if candidates else None


def _completed_index(settings: Settings) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest_path in runs_root(settings).glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("status") == "succeeded" and manifest.get("run_fingerprint"):
            index[str(manifest["run_fingerprint"])] = {"run_id": manifest.get("run_id"), "run_manifest_path": str(manifest_path)}
    return index


def _budget(cfg: AdaptiveArchConfig) -> TuneBudget:
    return TuneBudget(
        max_rounds=cfg.max_rounds,
        num_branches=cfg.num_branches,
        rollout_steps=cfg.rollout_steps,
        max_response_surface_evaluations=cfg.max_response_surface_evaluations,
        max_candidate_actions=cfg.max_candidate_actions,
        simulated_gpu_hour_budget=cfg.simulated_gpu_hour_budget,
        budget_ledger_enabled=True,
    )


def _optimizer_for(system: str) -> str:
    return SYSTEM_TO_OPTIMIZER.get(system, system)


def _relative_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frames = []
    for task, group in metrics.groupby("task"):
        group = group.copy()
        static = group[group["system"].isin(["static_policy", "linear_static_baseline"])]
        no_fork = group[group["system"].isin(["square_adaptive_arch_no_fork", "square_tune_no_fork"])]
        full = group[group["system"].isin(["square_adaptive_arch_full"])]
        static_raw = float(static["final_utility"].mean()) if not static.empty else 0.0
        static_cost = float(static["cost_adjusted_utility"].mean()) if not static.empty else 0.0
        no_fork_raw = float(no_fork["final_utility"].mean()) if not no_fork.empty else 0.0
        no_fork_cost = float(no_fork["cost_adjusted_utility"].mean()) if not no_fork.empty else 0.0
        full_raw = float(full["final_utility"].mean()) if not full.empty else 0.0
        full_cost = float(full["cost_adjusted_utility"].mean()) if not full.empty else 0.0
        group["raw_utility_delta_vs_static"] = group["final_utility"] - static_raw
        group["raw_utility_delta_vs_no_fork"] = group["final_utility"] - no_fork_raw
        group["raw_utility_delta_vs_full"] = group["final_utility"] - full_raw
        group["cost_adjusted_delta_vs_static"] = group["cost_adjusted_utility"] - static_cost
        group["cost_adjusted_delta_vs_no_fork"] = group["cost_adjusted_utility"] - no_fork_cost
        group["cost_adjusted_delta_vs_full"] = group["cost_adjusted_utility"] - full_cost
        group["budget_saved_vs_full"] = full_cost - group["cost_adjusted_utility"]
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def _run_single(
    settings: Settings,
    *,
    experiment_id: str,
    config_path: Path,
    task: str,
    seed: int,
    system: str,
    dataset_root: Path,
    budget: TuneBudget,
    completed: dict[str, dict[str, Any]],
    resume: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    dataset_dir = _latest_dataset(dataset_root, task, seed)
    if dataset_dir is None:
        raise FileNotFoundError(f"No generated dataset found for task={task} seed={seed} under {dataset_root}")
    fingerprint = stable_hash(
        {
            "experiment_id": experiment_id,
            "task": task,
            "seed": seed,
            "system": system,
            "config": sha256_file(config_path),
            "dataset": str(dataset_dir),
            "budget": budget.to_dict(),
        },
        16,
    )
    if resume or skip_completed:
        existing = completed.get(fingerprint)
        if existing:
            return {"status": "skipped", "reason": "completed fingerprint exists", **existing}
    date = datetime.now(timezone.utc)
    run_id = f"{date:%Y%m%d-%H%M%S}-{task[:18]}-{system[:18]}-{fingerprint[:8]}"
    output_dir = runs_root(settings) / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}" / run_id
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(output_dir, registry.protected_paths())
    manager.ensure_writable_path(output_dir)
    output_dir.mkdir(parents=True)
    df = pd.read_parquet(dataset_dir / "train.parquet")
    optimizer = _optimizer_for(system)
    result = run_optimizer(
        optimizer,
        df,
        mechanism_name=TASK_TO_MECHANISM.get(task, task),
        seed=seed,
        budget=budget,
        objective_weights=DEFAULT_OBJECTIVE_WEIGHTS,
    )
    metrics = architecture_metrics(system, task, result.metrics, result.adaptive_diagnostics)
    metrics.update(
        {
            "optimizer_name": optimizer,
            "system": system,
            "task": task,
            "seed": seed,
            "dataset_path": str(dataset_dir),
            "experiment_id": experiment_id,
        }
    )
    metrics_path = output_dir / "metrics.json"
    trajectory_path = output_dir / "trajectory.parquet"
    arch_branch_path = output_dir / "branch_diagnostics.parquet"
    result.trajectory.to_parquet(trajectory_path, index=False)
    result.branch_diagnostics.to_parquet(arch_branch_path, index=False)
    write_json(metrics_path, metrics)
    diagnostics = write_run_diagnostics(
        output_dir,
        run_id=run_id,
        task=task,
        seed=seed,
        system=system,
        trajectory=result.trajectory,
        adaptive=result.adaptive_diagnostics,
        metrics=metrics,
    )
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_type": "square_adaptive_arch_v1",
        "task": task,
        "seed": seed,
        "system": system,
        "optimizer_name": optimizer,
        "dataset_path": str(dataset_dir),
        "budget_config": budget.to_dict(),
        "metrics_path": str(metrics_path),
        "trajectory_path": str(trajectory_path),
        "diagnostics_path": diagnostics["diagnostics_dir"],
        "architecture_trace_path": diagnostics["trace_path"],
        "protected_results_checked": True,
        "write_root": str(output_dir),
        "no_overwrite_status": "append_only",
        "node_hostname": socket.gethostname(),
        "started_at_utc": date.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "run_fingerprint": fingerprint,
        "caveats": ["Adaptive architecture software benchmark; not physical SQUARE hardware validation."],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_text(output_dir / "explanation.md", f"# Adaptive Architecture Run\n\n- Task: `{task}`\n- System: `{system}`\n- Cost-adjusted utility: `{metrics.get('cost_adjusted_utility')}`\n")
    return {"status": "succeeded", "run_id": run_id, "run_manifest_path": str(output_dir / "run_manifest.json")}


def load_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows = []
    for manifest_path in runs_root(settings).glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "succeeded":
            continue
        metric_path = Path(str(manifest.get("metrics_path")))
        if metric_path.exists():
            rows.append(read_json(metric_path))
    return _relative_deltas(pd.DataFrame(rows))


def run_benchmark(
    settings: Settings,
    config_path: Path,
    *,
    resume: bool = True,
    skip_completed: bool = True,
    max_runs: int | None = None,
) -> dict[str, Any]:
    cfg = AdaptiveArchConfig.from_path(config_path)
    dataset_root = _expand_path(cfg.dataset_root, default_dataset_root(settings))
    prefix = f"square_adaptive_arch_v1_{cfg.experiment_name}"
    experiment_id = unique_id(prefix, {"config": str(config_path), "tasks": cfg.tasks, "systems": cfg.systems})
    report_dir = reports_root(settings) / experiment_id
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(reports_root(settings), registry.protected_paths())
    manager.ensure_writable_path(report_dir)
    report_dir.mkdir(parents=True)
    planned = [(task, seed, system) for task in cfg.tasks for seed in cfg.seeds for system in cfg.systems]
    if max_runs is not None:
        planned = planned[:max_runs]
    completed = _completed_index(settings)
    budget = _budget(cfg)
    results = []
    for task, seed, system in planned:
        try:
            results.append(
                _run_single(
                    settings,
                    experiment_id=experiment_id,
                    config_path=config_path,
                    task=task,
                    seed=seed,
                    system=system,
                    dataset_root=dataset_root,
                    budget=budget,
                    completed=completed,
                    resume=resume,
                    skip_completed=skip_completed,
                )
            )
        except Exception as exc:
            results.append({"status": "failed", "task": task, "seed": seed, "system": system, "error": str(exc)})
    metrics = load_metrics(settings, experiment_id)
    if not metrics.empty:
        metrics.to_parquet(report_dir / "metrics.parquet", index=False)
        metrics.to_csv(report_dir / "metrics.csv", index=False)
    summary = {
        "experiment_id": experiment_id,
        "experiment_type": "square_adaptive_arch_v1",
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "total_planned": len(planned),
        "succeeded": sum(row["status"] == "succeeded" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
        "reports_dir": str(report_dir),
    }
    write_json(report_dir / "experiment_summary.json", summary)
    no_overwrite = write_no_overwrite_audit(
        report_dir,
        experiment_id=experiment_id,
        protected_paths=[str(path) for path in registry.protected_paths()],
        write_roots=[str(report_dir), str(runs_root(settings)), str(certificates_root(settings))],
    )
    write_report(report_dir, experiment_id=experiment_id, summary=summary, metrics=metrics)
    return {**summary, "no_overwrite_audit": no_overwrite}


def generate_report(settings: Settings, experiment_id: str) -> dict[str, Any]:
    report_dir = reports_root(settings) / experiment_id
    summary = read_json(report_dir / "experiment_summary.json") if (report_dir / "experiment_summary.json").exists() else {"experiment_id": experiment_id}
    return write_report(report_dir, experiment_id=experiment_id, summary=summary, metrics=load_metrics(settings, experiment_id))


def generate_certificate(settings: Settings, experiment_id: str) -> dict[str, Any]:
    return write_certificates(settings.project_root, experiment_id, load_metrics(settings, experiment_id))


def run_all_v1(
    settings: Settings,
    *,
    protect_prior: bool,
    generate_synthetic: bool,
    run_smoke_first: bool,
    run_synthetic_matrix: bool,
    run_external_proxy: bool,
    generate_diagnostics: bool,
    generate_reports: bool,
    generate_certificates: bool,
    resume: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    registry = ProtectedResultsRegistry(settings)
    result: dict[str, Any] = {}
    if protect_prior:
        result["protection"] = registry.protect_defaults()
    if generate_synthetic:
        result["generation"] = generate_suite(default_dataset_root(settings), rows=50_000, seeds=[101, 202, 303, 404, 505])
    summaries = []
    if run_smoke_first:
        summaries.append(run_benchmark(settings, Path("configs/adaptive_arch/square_adaptive_arch_v1_smoke.yaml"), resume=resume, skip_completed=skip_completed))
    if run_synthetic_matrix:
        summaries.append(run_benchmark(settings, Path("configs/adaptive_arch/square_adaptive_arch_v1_synthetic_matrix.yaml"), resume=resume, skip_completed=skip_completed))
    if run_external_proxy:
        summaries.append(run_benchmark(settings, Path("configs/adaptive_arch/square_adaptive_arch_v1_external_proxy.yaml"), resume=resume, skip_completed=skip_completed))
    result["runs"] = summaries
    for summary in summaries:
        exp = str(summary["experiment_id"])
        if generate_diagnostics:
            result[f"{exp}_diagnostics"] = diagnose_experiment(settings, exp)
        if generate_reports:
            result[f"{exp}_report"] = generate_report(settings, exp)
        if generate_certificates:
            result[f"{exp}_certificate"] = generate_certificate(settings, exp)
    return result

