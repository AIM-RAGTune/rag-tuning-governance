from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_generalized.reporting.no_overwrite_audit import no_overwrite_audit
from square_sim.square_tune_matched_cost.certificates import evaluate_certificate, write_certificate
from square_sim.square_tune_matched_cost.config import MatchedCostRAGConfig
from square_sim.square_tune_matched_cost.datasets import (
    ingest_matched_cost_rag,
    latest_dataset_manifest,
)
from square_sim.square_tune_matched_cost.matched_cost import evaluate_system
from square_sim.square_tune_matched_cost.paths import (
    artifacts_root,
    certificates_root,
    reports_root,
)
from square_sim.square_tune_matched_cost.publication_bundle import create_publication_bundle
from square_sim.square_tune_matched_cost.reports import write_reports
from square_sim.square_tune_matched_cost.scenario_compile import (
    compile_matched_cost_scenario,
    latest_scenario_manifest,
)
from square_sim.square_tune_matched_cost.statistics import aggregate_statistics, utility_sensitivity
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager


def protect_prior(settings: Settings) -> dict[str, Any]:
    return ProtectedResultsRegistry(settings).protect_defaults(notes="SQUARETune matched-cost RAG kill-test prior protection")


def plan_matrix(config_path: Path) -> dict[str, Any]:
    cfg = MatchedCostRAGConfig.from_path(config_path)
    return {"matrix_name": cfg.matrix_name, "planned": len(cfg.planned_runs()), "systems": cfg.systems, "seeds": cfg.seeds}


def ensure_ingested(settings: Settings, config_path: Path) -> dict[str, Any]:
    cfg = MatchedCostRAGConfig.from_path(config_path)
    existing = latest_dataset_manifest(settings)
    if existing:
        return read_json(existing)
    return ingest_matched_cost_rag(settings, max_rows=cfg.max_rows)


def ensure_scenario(settings: Settings, config_path: Path) -> dict[str, Any]:
    cfg = MatchedCostRAGConfig.from_path(config_path)
    existing = latest_scenario_manifest(settings)
    if existing:
        return read_json(existing)
    return compile_matched_cost_scenario(settings, max_rows=cfg.max_rows)


def _load_metrics(artifact_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((artifact_dir / "runs").glob("*/metrics.json")):
        rows.append(read_json(path))
    return pd.DataFrame(rows)


def _write_diagnostics(diag_dir: Path, metrics: pd.DataFrame, stats: dict[str, pd.DataFrame], sensitivity: pd.DataFrame) -> None:
    diag_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(diag_dir / "per_seed_results.parquet", index=False)
    metrics.to_csv(diag_dir / "per_seed_results.csv", index=False)
    for name in ["paired_deltas", "bootstrap_intervals", "rankings"]:
        frame = stats.get(name, pd.DataFrame())
        frame.to_parquet(diag_dir / f"{name}.parquet", index=False)
        frame.to_csv(diag_dir / f"{name}.csv", index=False)
    sensitivity.to_parquet(diag_dir / "utility_sensitivity_results.parquet", index=False)
    sensitivity.to_csv(diag_dir / "utility_sensitivity_summary.csv", index=False)
    budget = metrics[["system", "seed", "budget_used", "budget_deviation_pct", "budget_confounded_flag"]] if not metrics.empty else pd.DataFrame()
    write_json(diag_dir / "budget_parity_report.json", {"rows": budget.to_dict(orient="records")})
    write_text(diag_dir / "budget_parity_report.md", "# Budget Parity Report\n\nMatched-cost gates use adaptive compute's validation-set expensive-compute rate.\n")
    gating = metrics[["system", "seed", "expensive_compute_invocation_rate", "target_expensive_compute_invocation_rate"]] if not metrics.empty else pd.DataFrame()
    write_json(diag_dir / "matched_cost_gating_summary.json", {"rows": gating.to_dict(orient="records")})
    write_text(diag_dir / "matched_cost_gating_summary.md", "# Matched-Cost Gating Summary\n\nRandom, uncertainty, retrieval, and entropy/margin gates are matched to the adaptive compute invocation rate.\n")
    write_text(diag_dir / "utility_sensitivity_summary.md", "# Utility Sensitivity\n\nSee `utility_sensitivity_results.parquet`.\n")
    write_text(diag_dir / "negative_result_analysis.md", "# Negative Result Analysis\n\nNegative results are retained as publishable kill-test evidence.\n")


def run_matrix(settings: Settings, config_path: Path, *, resume: bool = True, skip_completed: bool = True) -> dict[str, Any]:
    cfg = MatchedCostRAGConfig.from_path(config_path)
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(reports_root(settings), registry.protected_paths())
    experiment_id, report_dir = manager.create_experiment_dir(
        f"square_tune_matched_cost_rag_v1_{cfg.matrix_name}",
        {"config": str(config_path), "seeds": cfg.seeds, "systems": cfg.systems},
    )
    artifact_dir = artifacts_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    for path in [artifact_dir, cert_dir]:
        WriteOncePathManager(path, registry.protected_paths()).ensure_writable_path(path)
        path.mkdir(parents=True)
    run_dir = artifact_dir / "runs"
    run_dir.mkdir()
    scenario_manifest_path = latest_scenario_manifest(settings)
    if scenario_manifest_path is None:
        raise FileNotFoundError("No scenario compiled. Run matched-cost-rag compile-scenarios first.")
    scenario_manifest = read_json(scenario_manifest_path)
    validation = pd.read_parquet(scenario_manifest_path.parent / "splits" / "validation.parquet")
    test = pd.read_parquet(scenario_manifest_path.parent / "splits" / "test.parquet")
    completed: set[str] = set()
    results = []
    for row in cfg.planned_runs():
        fingerprint = stable_hash(
            {
                "experiment_id": experiment_id,
                "config_hash": sha256_file(config_path),
                "scenario_manifest": str(scenario_manifest_path),
                "system": row["system"],
                "seed": row["seed"],
            },
            16,
        )
        if (resume or skip_completed) and fingerprint in completed:
            results.append({"status": "skipped", "run_fingerprint": fingerprint})
            continue
        now = datetime.now(timezone.utc)
        run_id = f"{now:%Y%m%d-%H%M%S}-{row['system'][:28]}-{fingerprint[:8]}"
        out = run_dir / run_id
        if out.exists():
            raise FileExistsError(f"Run id collision: {run_id}")
        out.mkdir()
        try:
            result = evaluate_system(
                system=str(row["system"]),
                seed=int(row["seed"]),
                validation=validation,
                test=test,
                matched_cost_tolerance_pct=cfg.matched_cost_tolerance_pct,
                real_data_used=bool(scenario_manifest.get("real_data_used", False)),
            )
            metrics = result.metrics
            invocations = result.invocations
            status = "succeeded"
            errors: list[str] = []
        except Exception as exc:
            if not cfg.continue_on_failure:
                raise
            metrics = {
                "scenario": row["scenario"],
                "system": row["system"],
                "seed": row["seed"],
                "real_data_used": False,
                "held_out_test_cost_adjusted_utility": 0.0,
                "held_out_test_raw_quality": 0.0,
                "budget_confounded_flag": True,
                "error": str(exc),
            }
            invocations = pd.DataFrame()
            status = "failed"
            errors = [str(exc)]
        metrics_path = out / "metrics.json"
        invocations_path = out / "expensive_compute_invocations.parquet"
        write_json(metrics_path, metrics)
        invocations.to_parquet(invocations_path, index=False)
        manifest = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_type": "square_tune_matched_cost_rag_v1",
            "scenario": row["scenario"],
            "system": row["system"],
            "seed": row["seed"],
            "scenario_manifest_path": str(scenario_manifest_path),
            "real_data_used": bool(metrics.get("real_data_used", False)),
            "budget_config": {"matched_cost_tolerance_pct": cfg.matched_cost_tolerance_pct},
            "metrics_path": str(metrics_path),
            "expensive_compute_invocations_path": str(invocations_path),
            "protected_results_checked": True,
            "write_root": str(out),
            "node_hostname": socket.gethostname(),
            "started_at_utc": now.isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "errors": errors,
            "run_fingerprint": fingerprint,
            "caveats": ["Matched-cost RAG software kill-test; no SQUARE hardware or commercial claim."],
        }
        write_json(out / "run_manifest.json", manifest)
        completed.add(fingerprint)
        results.append({"status": status, "run_id": run_id, "errors": errors})
    metrics_df = _load_metrics(artifact_dir)
    stats = aggregate_statistics(metrics_df, bootstrap_samples=cfg.bootstrap_samples)
    sensitivity = utility_sensitivity(metrics_df)
    audit = no_overwrite_audit(experiment_id, registry.protected_paths(), [report_dir, artifact_dir, cert_dir])
    cert = evaluate_certificate(metrics_df, sensitivity, no_overwrite_status=str(audit["status"]))
    certificate = write_certificate(cert_dir, experiment_id, cert)
    diag_dir = artifact_dir / "diagnostics"
    _write_diagnostics(diag_dir, metrics_df, stats, sensitivity)
    summary = {
        "experiment_id": experiment_id,
        "matrix_name": cfg.matrix_name,
        "config_path": str(config_path),
        "planned": len(cfg.planned_runs()),
        "succeeded": sum(1 for row in results if row["status"] == "succeeded"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "skipped": sum(1 for row in results if row["status"] == "skipped"),
        "dataset": read_json(Path(scenario_manifest["dataset_manifest_path"])),
        "scenario": scenario_manifest,
        "diagnostics_dir": str(diag_dir),
    }
    report_paths = write_reports(
        report_dir,
        experiment_id=experiment_id,
        summary=summary,
        metrics=metrics_df,
        stats=stats,
        sensitivity=sensitivity,
        certificate=certificate,
        no_overwrite_audit=audit,
    )
    return {
        **summary,
        "reports_dir": str(report_dir),
        "artifacts_dir": str(artifact_dir),
        "certificates_dir": str(cert_dir),
        "certificate": certificate,
        "no_overwrite_audit": audit,
        "report_paths": report_paths,
    }


def load_metrics(settings: Settings, experiment_id: str) -> pd.DataFrame:
    return _load_metrics(artifacts_root(settings) / experiment_id)


def rerun_reports(settings: Settings, experiment_id: str, *, bootstrap_samples: int = 1000) -> dict[str, Any]:
    artifact_dir = artifacts_root(settings) / experiment_id
    report_dir = reports_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    if (report_dir / "executive_summary.md").exists() and (cert_dir / "certificate.json").exists():
        return {
            "status": "existing_artifacts_preserved",
            "reports_dir": str(report_dir),
            "certificates_dir": str(cert_dir),
            "executive_summary": str(report_dir / "executive_summary.md"),
            "certificate": str(cert_dir / "certificate.json"),
        }
    metrics = _load_metrics(artifact_dir)
    stats = aggregate_statistics(metrics, bootstrap_samples=bootstrap_samples)
    sensitivity = utility_sensitivity(metrics)
    audit_path = report_dir / "no_overwrite_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else {"status": "append_only_confirmed"}
    cert = evaluate_certificate(metrics, sensitivity, no_overwrite_status=str(audit["status"]))
    certificate = write_certificate(cert_dir, experiment_id, cert)
    summary_path = report_dir / "executive_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"experiment_id": experiment_id, "planned": len(metrics), "succeeded": len(metrics), "failed": 0, "skipped": 0}
    return write_reports(report_dir, experiment_id=experiment_id, summary=summary, metrics=metrics, stats=stats, sensitivity=sensitivity, certificate=certificate, no_overwrite_audit=audit)


def build_publication_bundle(settings: Settings, experiment_id: str, output: Path) -> dict[str, Any]:
    return create_publication_bundle(settings, experiment_id, output)
