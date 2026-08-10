from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from square_sim.config import Settings
from square_sim.tune.config import TuneExperimentConfig, experiment_id_for
from square_sim.tune.reporting.commercial_value_report import write_commercial_value_report
from square_sim.tune.reporting.mechanism_report import write_mechanism_report
from square_sim.tune.reporting.run_report import write_run_explanation
from square_sim.tune.simulator.square_tune_optimizer import run_optimizer
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _dataset_dir(root: Path, dataset_key: str, seed: int) -> Path:
    return root / dataset_key / f"seed_{seed}"


def _default_dataset_root(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "synthetic" / "square_tune"


def _load_metrics_for_experiment(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted((settings.project_root / "tune_runs").glob("*/*/*/*/run_manifest.json")):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "succeeded":
            continue
        metrics_path = Path(str(manifest.get("metrics_path")))
        if metrics_path.exists():
            metric = read_json(metrics_path)
            metric.update(
                {
                    "dataset_key": manifest.get("dataset_key"),
                    "optimizer_name": manifest.get("model_or_optimizer_name"),
                    "seed": manifest.get("seed"),
                    "control_type": manifest.get("control_type"),
                    "run_id": manifest.get("run_id"),
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def _completed_fingerprint_index(settings: Settings) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for manifest_path in (settings.project_root / "tune_runs").glob("*/*/*/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        fingerprint = manifest.get("run_fingerprint")
        if fingerprint and manifest.get("status") == "succeeded":
            completed[str(fingerprint)] = {
                "run_id": manifest.get("run_id"),
                "manifest_path": str(manifest_path),
            }
    return completed


def run_tune_single(
    settings: Settings,
    *,
    experiment_id: str,
    dataset_root: Path,
    dataset_key: str,
    seed: int,
    optimizer_name: str,
    cfg: TuneExperimentConfig,
    config_path: Path,
    device: str = "cpu",
    skip_completed: bool = True,
    completed_fingerprints: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dataset_dir = _dataset_dir(dataset_root, dataset_key, seed)
    manifest_path = dataset_dir / "generator_manifest.json"
    expected_path = dataset_dir / "expected_outcomes.json"
    protocol_path = Path(cfg.protocol_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing generated dataset at {dataset_dir}. Run `square-sim tune synthetic generate` first."
        )
    manifest = read_json(manifest_path)
    expected = read_json(expected_path)
    version_id = str(manifest.get("dataset_version_id", f"{dataset_key}-{seed}"))
    fingerprint = stable_hash(
        {
            "experiment_id": experiment_id,
            "dataset_key": dataset_key,
            "dataset_version_id": version_id,
            "seed": seed,
            "optimizer": optimizer_name,
            "budget": cfg.budget.to_dict(),
            "objective_weights": cfg.objective_weights,
            "protocol": str(protocol_path),
        },
        16,
    )
    run_id = f"{_now_id()}-{dataset_key.replace('synthetic_llm_', '')[:16]}-{optimizer_name[:18]}-{fingerprint[:8]}"
    date = datetime.now(timezone.utc)
    output_dir = settings.project_root / "tune_runs" / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}" / run_id
    if skip_completed:
        completed = completed_fingerprints
        if completed is None:
            completed = _completed_fingerprint_index(settings)
        existing = completed.get(fingerprint)
        if existing:
            return {
                "status": "skipped",
                "run_id": existing.get("run_id"),
                "reason": "completed fingerprint exists",
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_dir / "train.parquet")
    result = run_optimizer(
        optimizer_name,
        df,
        mechanism_name=str(manifest["mechanism_name"]),
        seed=seed,
        budget=cfg.budget,
        objective_weights=cfg.objective_weights,
    )
    metrics_path = output_dir / "metrics.json"
    trajectory_path = output_dir / "trajectory.parquet"
    branch_path = output_dir / "branch_diagnostics.parquet"
    final_policy_path = output_dir / "final_policy.json"
    config_copy_path = output_dir / "config.yaml"
    result.trajectory.to_parquet(trajectory_path, index=False)
    result.branch_diagnostics.to_parquet(branch_path, index=False)
    write_json(metrics_path, result.metrics)
    write_json(final_policy_path, result.final_policy)
    write_text(config_copy_path, yaml.safe_dump(cfg.to_dict(), sort_keys=True))
    protocol_hash = sha256_file(protocol_path) if protocol_path.exists() else stable_hash({"missing": str(protocol_path)}, 16)
    run_manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "run_type": "synthetic",
        "dataset_key": dataset_key,
        "dataset_version_id": version_id,
        "generator_manifest_path": str(manifest_path),
        "generator_manifest_hash": sha256_file(manifest_path),
        "protocol_path": str(protocol_path),
        "protocol_hash": protocol_hash,
        "seed": seed,
        "model_or_optimizer_name": optimizer_name,
        "ablation_flags": {
            "is_square_tune": optimizer_name.startswith("square_tune"),
            "variant": optimizer_name,
        },
        "budget_config": cfg.budget.to_dict(),
        "initial_state_summary": result.initial_state.to_dict(),
        "final_state_summary": result.final_state.to_dict(),
        "metrics_path": str(metrics_path),
        "trajectory_path": str(trajectory_path),
        "branch_diagnostics_path": str(branch_path),
        "memory_diagnostics_path": str(final_policy_path),
        "final_policy_path": str(final_policy_path),
        "comparison_group": f"{dataset_key}:{seed}",
        "code_git_commit": _git_commit(),
        "node_hostname": socket.gethostname(),
        "device": device,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "caveats": [
            "Synthetic mechanism diagnostic only.",
            "Not physical hardware validation.",
        ],
        "control_type": expected.get("control_type"),
        "run_fingerprint": fingerprint,
    }
    explanation_path = output_dir / "explanation.md"
    run_manifest["explanation_path"] = str(explanation_path)
    write_run_explanation(explanation_path, run_manifest, result.metrics)
    write_json(output_dir / "run_manifest.json", run_manifest)
    return {"status": "succeeded", "run_id": run_id, "metrics_path": str(metrics_path), "run_manifest_path": str(output_dir / "run_manifest.json")}


def run_tune_matrix(
    settings: Settings,
    config_path: Path,
    *,
    device: str = "cpu",
    resume: bool = True,
    skip_completed: bool = True,
    max_runs: int | None = None,
) -> dict[str, Any]:
    cfg = TuneExperimentConfig.from_path(config_path)
    experiment_id = experiment_id_for(config_path, cfg)
    dataset_root = Path(cfg.dataset_root).expanduser() if cfg.dataset_root else _default_dataset_root(settings)
    results = []
    planned = [(d, seed, opt) for d in cfg.datasets for seed in cfg.seeds for opt in cfg.optimizers]
    if max_runs is not None:
        planned = planned[:max_runs]
    completed_fingerprints = _completed_fingerprint_index(settings) if skip_completed or resume else {}
    for dataset_key, seed, optimizer_name in planned:
        try:
            result = run_tune_single(
                settings,
                experiment_id=experiment_id,
                dataset_root=dataset_root,
                dataset_key=dataset_key,
                seed=seed,
                optimizer_name=optimizer_name,
                cfg=cfg,
                config_path=config_path,
                device=device,
                skip_completed=skip_completed or resume,
                completed_fingerprints=completed_fingerprints,
            )
            results.append(result)
        except Exception as exc:
            results.append({"status": "failed", "dataset_key": dataset_key, "seed": seed, "optimizer_name": optimizer_name, "error": str(exc)})
    output_dir = settings.project_root / "reports" / "square_tune" / "experiments" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics_for_experiment(settings, experiment_id)
    if not metrics.empty:
        metrics.to_parquet(output_dir / "metrics.parquet", index=False)
        metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_mechanism_report(settings.project_root / "reports" / "square_tune" / "mechanisms" / experiment_id, experiment_id, metrics)
    write_commercial_value_report(settings.project_root / "reports" / "square_tune" / "commercial" / experiment_id, experiment_id, metrics)
    summary = {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "dataset_root": str(dataset_root),
        "total_planned": len(planned),
        "completed": sum(1 for row in results if row["status"] == "succeeded"),
        "skipped": sum(1 for row in results if row["status"] == "skipped"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "results": results,
        "reports_dir": str(output_dir),
    }
    write_json(output_dir / "experiment_summary.json", summary)
    lines = [f"# SQUARETune Experiment Summary: {experiment_id}", "", f"- Total planned: `{summary['total_planned']}`", f"- Completed: `{summary['completed']}`", f"- Skipped: `{summary['skipped']}`", f"- Failed: `{summary['failed']}`"]
    write_text(output_dir / "experiment_summary.md", "\n".join(lines) + "\n")
    return summary


def generate_reports(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = _load_metrics_for_experiment(settings, experiment_id)
    mechanism = write_mechanism_report(settings.project_root / "reports" / "square_tune" / "mechanisms" / experiment_id, experiment_id, metrics)
    commercial = write_commercial_value_report(settings.project_root / "reports" / "square_tune" / "commercial" / experiment_id, experiment_id, metrics)
    return {"experiment_id": experiment_id, "mechanism_report": mechanism, "commercial_value_report": commercial}
