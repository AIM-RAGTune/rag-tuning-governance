from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_generalized.config import GeneralizedConfig
from square_sim.square_tune_generalized.reporting.certificates import write_generalized_certificates
from square_sim.square_tune_generalized.reporting.no_overwrite_audit import no_overwrite_audit
from square_sim.square_tune_generalized.reporting.reports import write_generalized_reports
from square_sim.square_tune_generalized.simulation.response_surfaces import (
    simulate_generalized_system,
)
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager


def reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "generalized"


def artifacts_root(settings: Settings) -> Path:
    return settings.project_root / "artifacts" / "generalized"


def certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "generalized"


def dataset_root(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "generalized" / "v1"


def scenario_root(settings: Settings) -> Path:
    return settings.project_root / "scenarios" / "generalized" / "v1"


def _latest_scenario_manifest(root: Path, track: str, scenario: str) -> Path:
    manifests = sorted((root / track / scenario).glob("*/scenario_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No compiled scenario found for {track}/{scenario} under {root}")
    return max(manifests, key=lambda path: str(read_json(path).get("generated_at_utc", "")))


def _load_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in (artifacts_root(settings) / experiment_id / "runs").glob("*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("status") == "succeeded":
            rows.append(read_json(Path(str(manifest["metrics_path"]))))
    return pd.DataFrame(rows)


def run_generalized_matrix(
    settings: Settings,
    config_path: Path,
    *,
    resume: bool = True,
    skip_completed: bool = True,
) -> dict[str, Any]:
    cfg = GeneralizedConfig.from_path(config_path)
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(reports_root(settings), registry.protected_paths())
    experiment_id, report_dir = manager.create_experiment_dir(
        f"square_tune_generalized_v1_{cfg.matrix_name}",
        {"config": str(config_path), "tracks": cfg.tracks, "seeds": cfg.seeds},
    )
    artifact_dir = artifacts_root(settings) / experiment_id
    certificate_dir = certificates_root(settings) / experiment_id
    for path in [artifact_dir, certificate_dir]:
        WriteOncePathManager(path, registry.protected_paths()).ensure_writable_path(path)
        path.mkdir(parents=True)
    run_dir = artifact_dir / "runs"
    run_dir.mkdir()
    results = []
    completed: set[str] = set()
    for row in cfg.planned_runs():
        scenario_manifest = _latest_scenario_manifest(scenario_root(settings), row["track"], row["scenario"])
        fingerprint = stable_hash(
            {
                "experiment_id": experiment_id,
                "config_hash": sha256_file(config_path),
                "scenario": str(scenario_manifest),
                "system": row["system"],
                "seed": row["seed"],
                "stress_profile": row.get("stress_profile", "nominal"),
            },
            16,
        )
        if (resume or skip_completed) and fingerprint in completed:
            results.append({"status": "skipped", "reason": "completed fingerprint exists", "run_fingerprint": fingerprint})
            continue
        date = datetime.now(timezone.utc)
        profile_slug = str(row.get("stress_profile", "nominal"))[:14]
        run_id = f"{date:%Y%m%d-%H%M%S}-{row['track'][:12]}-{row['scenario'][:18]}-{profile_slug}-{row['system'][:18]}-{fingerprint[:8]}"
        out = run_dir / run_id
        if out.exists():
            raise FileExistsError(f"Run id collision: {run_id}")
        out.mkdir()
        try:
            scenario = read_json(scenario_manifest)
            train = pd.read_parquet(scenario_manifest.parent / "splits" / "train.parquet")
            metrics, trace = simulate_generalized_system(
                row["track"],
                row["scenario"],
                row["system"],
                int(row["seed"]),
                train,
                stress_profile=str(row.get("stress_profile", "nominal")),
            )
            status = "succeeded"
            errors: list[str] = []
        except Exception as exc:
            metrics = {
                "track": row["track"],
                "scenario": row["scenario"],
                "system": row["system"],
                "seed": row["seed"],
                "stress_profile": row.get("stress_profile", "nominal"),
                "final_utility": 0.0,
                "cost_adjusted_utility": 0.0,
                "budget_parity_ok": False,
                "error": str(exc),
            }
            trace = pd.DataFrame()
            scenario = {}
            status = "failed"
            errors = [str(exc)]
            if not cfg.continue_on_failure:
                raise
        metrics_path = out / "metrics.json"
        trace_path = out / "trajectory.parquet"
        trace.to_parquet(trace_path, index=False)
        write_json(metrics_path, metrics)
        manifest = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_type": "square_tune_generalized_v1",
            "track": row["track"],
            "scenario": row["scenario"],
            "system": row["system"],
            "seed": row["seed"],
            "stress_profile": row.get("stress_profile", "nominal"),
            "scenario_id": scenario.get("scenario_id"),
            "scenario_manifest_path": str(scenario_manifest),
            "source_datasets": scenario.get("source_datasets", []),
            "license_status": metrics.get("license_status", "captured"),
            "budget_config": {"budget_parity_required": True},
            "metrics_path": str(metrics_path),
            "trajectory_path": str(trace_path),
            "protected_results_checked": True,
            "write_root": str(out),
            "node_hostname": socket.gethostname(),
            "started_at_utc": date.isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "errors": errors,
            "run_fingerprint": fingerprint,
            "caveats": [
                "Generalized SQUARETune software benchmark; not hardware validation.",
                "Healthcare tracks are operations proxies only and not clinical recommendations.",
            ],
        }
        write_json(out / "run_manifest.json", manifest)
        results.append({"status": status, "run_id": run_id, "run_manifest_path": str(out / "run_manifest.json"), "errors": errors})
        completed.add(fingerprint)
    metrics_df = _load_metrics(settings, experiment_id)
    summary = {
        "experiment_id": experiment_id,
        "matrix_name": cfg.matrix_name,
        "config_path": str(config_path),
        "tracks": cfg.tracks,
        "stress_profiles": cfg.stress_profiles or ["nominal"],
        "total_planned": len(cfg.planned_runs()),
        "succeeded": sum(row["status"] == "succeeded" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "results": results,
    }
    audit = no_overwrite_audit(experiment_id, registry.protected_paths(), [report_dir, artifact_dir, certificate_dir])
    report_paths = write_generalized_reports(report_dir, experiment_id, {**summary, "no_overwrite_audit": audit}, metrics_df)
    cert_index = write_generalized_certificates(certificate_dir, experiment_id, metrics_df)
    write_json(report_dir / "certificate_index.json", cert_index)
    write_json(report_dir / "no_overwrite_audit.json", audit)
    write_text(report_dir / "no_overwrite_audit.md", f"# No Overwrite Audit\n\nStatus: `{audit['status']}`\n")
    return {**summary, "reports_dir": str(report_dir), "artifacts_dir": str(artifact_dir), "certificates_dir": str(certificate_dir), "no_overwrite_audit": audit, "report_paths": report_paths}


def load_generalized_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    return _load_metrics(settings, experiment_id)
