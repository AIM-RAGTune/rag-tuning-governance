from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.config import Settings
from square_sim.square_core.common.diagnostics import no_overwrite_audit, write_run_diagnostics
from square_sim.square_core.config import CoreConfig
from square_sim.square_core.matrix.certificates import write_core_certificates
from square_sim.square_core.matrix.plan import plan_matrix
from square_sim.square_core.matrix.registry import TRACK_RUNNERS
from square_sim.square_core.matrix.reports import write_core_reports
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager


def reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "square_core" / "v1"


def artifacts_root(settings: Settings) -> Path:
    return settings.project_root / "artifacts" / "square_core" / "v1"


def certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "square_core" / "v1"


def logs_root(settings: Settings) -> Path:
    return settings.project_root / "logs" / "square_core" / "v1"


def _load_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows = []
    for manifest_path in artifacts_root(settings).glob("*/runs/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") == experiment_id and manifest.get("status") == "succeeded":
            rows.append(read_json(Path(str(manifest["metrics_path"]))))
    return pd.DataFrame(rows)


def _completed_index(settings: Settings) -> dict[str, dict[str, str]]:
    index = {}
    for manifest_path in artifacts_root(settings).glob("*/runs/*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("status") == "succeeded" and manifest.get("run_fingerprint"):
            index[str(manifest["run_fingerprint"])] = {"run_id": str(manifest["run_id"]), "run_manifest_path": str(manifest_path)}
    return index


def _run_one(
    settings: Settings,
    *,
    experiment_id: str,
    cfg: CoreConfig,
    config_path: Path,
    run_dir: Path,
    track: str,
    task: str,
    system: str,
    seed: int,
    completed: dict[str, dict[str, str]],
    resume: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    fingerprint = stable_hash(
        {
            "experiment_id": experiment_id,
            "config": sha256_file(config_path),
            "track": track,
            "task": task,
            "system": system,
            "seed": seed,
            "grid_size": cfg.grid_size,
            "emitter_count": cfg.emitter_count,
            "steps": cfg.steps,
        },
        16,
    )
    if resume or skip_completed:
        existing = completed.get(fingerprint)
        if existing:
            return {"status": "skipped", "reason": "completed fingerprint exists", **existing}
    runner = TRACK_RUNNERS[track]
    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%d-%H%M%S}-{track[:14]}-{task[:18]}-{system[:18]}-{fingerprint[:8]}"
    out = run_dir / run_id
    manager = WriteOncePathManager(out, ProtectedResultsRegistry(settings).protected_paths())
    manager.ensure_writable_path(out)
    out.mkdir(parents=True)
    try:
        metrics, trace = runner(task, system, seed, grid_size=cfg.grid_size, emitter_count=cfg.emitter_count, steps=cfg.steps)
        status = "succeeded"
        errors: list[str] = []
    except Exception as exc:  # pragma: no cover - exercised through integration failures.
        metrics, trace = {"final_utility": 0.0, "cost_adjusted_utility": 0.0, "numerical_instability": True}, []
        status = "failed"
        errors = [str(exc)]
        if not cfg.continue_on_failure:
            raise
    metrics.update({"track": track, "task": task, "system": system, "seed": seed, "experiment_id": experiment_id})
    metrics_path = out / "metrics.json"
    write_json(metrics_path, metrics)
    diagnostics = write_run_diagnostics(out, metrics, trace)
    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_type": "square_core_validation_v1",
        "track": track,
        "task": task,
        "system": system,
        "seed": seed,
        "device": cfg.device,
        "precision": cfg.precision,
        "budget_config": {"grid_size": cfg.grid_size, "emitter_count": cfg.emitter_count, "steps": cfg.steps},
        "metrics_path": str(metrics_path),
        "diagnostics_path": diagnostics,
        "protected_results_checked": True,
        "write_root": str(out),
        "node_hostname": socket.gethostname(),
        "started_at_utc": started.isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors": errors,
        "run_fingerprint": fingerprint,
        "caveats": ["SQUARE Core Validation Matrix software simulation; not physical hardware validation."],
    }
    write_json(out / "run_manifest.json", manifest)
    write_text(out / "config.yaml", config_path.read_text(encoding="utf-8"))
    write_text(out / "logs" / "run.log", f"{status}: {track}/{task}/{system}/{seed}\n")
    return {"status": status, "run_id": run_id, "run_manifest_path": str(out / "run_manifest.json"), "errors": errors}


def run_core_matrix(settings: Settings, config_path: Path, *, resume: bool = True, skip_completed: bool = True) -> dict[str, Any]:
    cfg = CoreConfig.from_path(config_path)
    registry = ProtectedResultsRegistry(settings)
    report_root = reports_root(settings)
    manager = WriteOncePathManager(report_root, registry.protected_paths())
    experiment_id, report_dir = manager.create_experiment_dir(
        f"square_core_validation_v1_{cfg.matrix_name}",
        {"config": str(config_path), "tracks": cfg.tracks, "seeds": cfg.seeds},
    )
    artifact_dir = artifacts_root(settings) / experiment_id
    certificate_dir = certificates_root(settings) / experiment_id
    log_dir = logs_root(settings) / experiment_id
    for path in [artifact_dir, certificate_dir, log_dir]:
        WriteOncePathManager(path, registry.protected_paths()).ensure_writable_path(path)
        path.mkdir(parents=True)
    plan = plan_matrix(config_path)
    write_json(report_dir / "experiment_manifest.json", {"experiment_id": experiment_id, **plan})
    completed = _completed_index(settings)
    run_dir = artifact_dir / "runs"
    run_dir.mkdir(parents=True)
    results = []
    for row in plan["runs"]:
        results.append(
            _run_one(
                settings,
                experiment_id=experiment_id,
                cfg=cfg,
                config_path=config_path,
                run_dir=run_dir,
                track=row["track"],
                task=row["task"],
                system=row["system"],
                seed=int(row["seed"]),
                completed=completed,
                resume=resume,
                skip_completed=skip_completed,
            )
        )
    metrics = _load_metrics(settings, experiment_id)
    summary = {
        "experiment_id": experiment_id,
        "matrix_name": cfg.matrix_name,
        "config_path": str(config_path),
        "tracks": cfg.tracks,
        "total_planned": len(plan["runs"]),
        "succeeded": sum(row["status"] == "succeeded" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "results": results,
    }
    audit = no_overwrite_audit(experiment_id, registry.protected_paths(), [report_dir, artifact_dir, certificate_dir, log_dir])
    write_json(report_dir / "no_overwrite_audit.json", audit)
    write_text(report_dir / "no_overwrite_audit.md", f"# No Overwrite Audit\n\nStatus: `{audit['status']}`\n")
    report_paths = write_core_reports(report_dir, experiment_id, {**summary, "no_overwrite_audit": audit}, metrics)
    cert_index = write_core_certificates(certificate_dir, experiment_id, metrics)
    write_json(report_dir / "certificate_index.json", cert_index)
    return {**summary, "reports_dir": str(report_dir), "artifacts_dir": str(artifact_dir), "certificates_dir": str(certificate_dir), "no_overwrite_audit": audit, "report_paths": report_paths}


def diagnose(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = _load_metrics(settings, experiment_id)
    from square_sim.square_core.matrix.aggregate import summarize_metrics

    return {"experiment_id": experiment_id, **summarize_metrics(metrics)}


def report(settings: Settings, experiment_id: str) -> dict[str, Any]:
    metrics = _load_metrics(settings, experiment_id)
    report_dir = reports_root(settings) / experiment_id
    summary_path = report_dir / "experiment_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"experiment_id": experiment_id}
    return write_core_reports(report_dir, experiment_id, summary, metrics)


def certificate(settings: Settings, experiment_id: str) -> dict[str, Any]:
    return write_core_certificates(certificates_root(settings) / experiment_id, experiment_id, _load_metrics(settings, experiment_id))
