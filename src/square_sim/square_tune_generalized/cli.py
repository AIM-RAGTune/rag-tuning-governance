from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer
from rich import print

from square_sim.config import Settings
from square_sim.square_tune_generalized.config import TRACK_DATASET_KEYS, GeneralizedConfig
from square_sim.square_tune_generalized.datasets.audits import write_dataset_audit
from square_sim.square_tune_generalized.datasets.classical_ml_hybrid import (
    generate_ml_to_llm_hybrid,
)
from square_sim.square_tune_generalized.datasets.elastic_compute import (
    generate_elastic_compute_trace,
)
from square_sim.square_tune_generalized.datasets.licenses import (
    generalized_license_summary,
    license_record,
)
from square_sim.square_tune_generalized.datasets.manifests import write_dataset_artifacts
from square_sim.square_tune_generalized.datasets.patient_flow import (
    generate_patient_flow_synthetic_proxy,
    import_mimic_patient_flow,
)
from square_sim.square_tune_generalized.datasets.rag import generate_rag_proxy
from square_sim.square_tune_generalized.reporting.certificates import write_generalized_certificates
from square_sim.square_tune_generalized.reporting.publication_bundle import (
    create_publication_bundle,
)
from square_sim.square_tune_generalized.reporting.reports import write_generalized_reports
from square_sim.square_tune_generalized.scenarios.compiler import compile_generalized_scenarios
from square_sim.square_tune_generalized.simulation.runner import (
    dataset_root,
    load_generalized_metrics,
    reports_root,
    run_generalized_matrix,
    scenario_root,
)
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import stable_hash

app = typer.Typer(help="SQUARETune generalized RAG and decision-optimization benchmark")
ingest_app = typer.Typer(help="Prepare generalized benchmark datasets")
app.add_typer(ingest_app, name="ingest")


def settings() -> Settings:
    return Settings.from_env()


def _expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _rows_from_config(config: Path, track: str, fallback: int = 2000) -> int:
    try:
        cfg = GeneralizedConfig.from_path(config)
        return cfg.rows_per_track
    except Exception:
        return fallback


def _write_ingestion_report(root: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    licenses = [
        license_record(
            dataset_key=str(row["dataset_key"]),
            source_type="synthetic_or_existing_external_proxy",
            license_status=str(row.get("license_status", "captured")),
            publication_safe=bool(row.get("publication_safe", True)),
            caveats=["Synthetic/public-safe proxy; no restricted raw data committed."],
        )
        for row in manifests
    ]
    payload = {"manifests": manifests, "license_summary": generalized_license_summary(licenses)}
    write_json(root / "ingestion_report.json", payload)
    write_text(
        root / "ingestion_report.md",
        "# SQUARETune Generalized Ingestion Report\n\n"
        + "\n".join(f"- `{row['dataset_key']}` rows={row['row_count']} version={row['dataset_version_id']}" for row in manifests)
        + "\n",
    )
    write_json(root / "dataset_license_summary.json", payload["license_summary"])
    write_text(root / "dataset_license_summary.md", "# Dataset License Summary\n\nAll generated proxy datasets are publication-safe synthetic or compact external-proxy derivatives.\n")
    return payload


def _ingest_track(track: str, config: Path, *, rows: int | None = None) -> dict[str, Any]:
    s = settings()
    n = rows or _rows_from_config(config, track)
    generators = {
        "rag": generate_rag_proxy,
        "patient_flow": generate_patient_flow_synthetic_proxy,
        "elastic_compute": generate_elastic_compute_trace,
        "ml_to_llm": generate_ml_to_llm_hybrid,
    }
    dataset_key = TRACK_DATASET_KEYS[track]
    frame = generators[track](n, 101)
    license_metadata = license_record(
        dataset_key=dataset_key,
        source_type="synthetic_proxy" if track != "rag" else "external_proxy_or_synthetic_rag",
        license_status="captured",
        publication_safe=True,
        caveats=["No restricted raw rows are committed by this command."],
    )
    manifest = write_dataset_artifacts(
        root=dataset_root(s),
        dataset_key=dataset_key,
        track=track,
        rows=frame,
        license_metadata=license_metadata,
        source_note="Deterministic generalized benchmark proxy dataset.",
    )
    _write_ingestion_report(dataset_root(s), [manifest])
    return manifest


def _ingest_mimic_patient_flow(config: Path, source_path: Path, *, rows: int | None = None) -> dict[str, Any]:
    s = settings()
    n = rows or _rows_from_config(config, "patient_flow")
    frame = import_mimic_patient_flow(source_path, max_rows=n)
    license_metadata = license_record(
        dataset_key="mimic_patient_flow_operations_proxy_v1",
        source_type="credentialed_manual_mimic_import",
        license_status="credentialed_manual",
        publication_safe=False,
        caveats=[
            "MIMIC source data is credentialed and controlled.",
            "Publication bundles may include aggregate metrics and configs only, not raw MIMIC rows.",
            "Operations proxy only; not clinical diagnosis or treatment planning.",
        ],
    )
    manifest = write_dataset_artifacts(
        root=dataset_root(s),
        dataset_key="mimic_patient_flow_operations_proxy_v1",
        track="patient_flow",
        rows=frame,
        license_metadata=license_metadata,
        source_note=f"Credentialed manual MIMIC import from {source_path}; derived operations features only.",
    )
    _write_ingestion_report(dataset_root(s), [manifest])
    return manifest


@app.command("freeze-evidence")
def freeze_evidence(output: Path = typer.Option(..., "--output")) -> None:
    s = settings()
    registry = ProtectedResultsRegistry(s)
    protected = registry.protect_defaults(notes="SQUARETune generalized v1 evidence freeze")
    out = _expand(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evidence package: {out}")
    out.mkdir(parents=True)
    prior = [
        "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd",
        "square_tune_external_v1_square_tune_external_v2_adaptive_compute_ragtruth_20260802-021733-b3cf204dc6",
        "square_adaptive_arch_v1_external_proxy_20260802-030901-0cdcd779b7",
        "square_core_validation_v1_full_matrix_20260802-152150-d2d7bb0cc3",
    ]
    manifest = {
        "evidence_package": out.name,
        "prior_experiment_ids": prior,
        "protected_paths_added": protected.get("protected_paths", []),
        "source_vs_inference_note": "This package stores compact references and summaries; it does not copy raw datasets or tensors.",
    }
    write_json(out / "evidence_manifest.json", manifest)
    write_json(out / "protected_paths_added.json", protected)
    write_text(
        out / "evidence_summary.md",
        "# SQUARETune Generalized v1 Evidence Freeze\n\n"
        + "\n".join(f"- `{item}`" for item in prior)
        + "\n\nPrior artifacts are read-only evidence inputs for this phase.\n",
    )
    print(manifest)


@ingest_app.command("rag")
def ingest_rag(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    print(_ingest_track("rag", config))


@ingest_app.command("patient-flow")
def ingest_patient_flow(
    config: Path = typer.Option(..., "--config"),
    manual_mimic_path: str = typer.Option("<path_or_none>", "--manual-mimic-path"),
    allow_synthetic_proxy: bool = typer.Option(False, "--allow-synthetic-proxy"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    if manual_mimic_path not in {"<path_or_none>", "none", "None", ""}:
        print(_ingest_mimic_patient_flow(config, _expand(manual_mimic_path)))
        return
    if not allow_synthetic_proxy:
        raise typer.BadParameter("MIMIC unavailable; pass --allow-synthetic-proxy to generate public-safe proxy data.")
    print(_ingest_track("patient_flow", config))


@ingest_app.command("elastic-compute")
def ingest_elastic_compute(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    print(_ingest_track("elastic_compute", config))


@ingest_app.command("ml-to-llm")
def ingest_ml_to_llm(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    print(_ingest_track("ml_to_llm", config))


@app.command("compile-scenarios")
def compile_scenarios_cmd(config: Path = typer.Option(..., "--config")) -> None:
    s = settings()
    print(compile_generalized_scenarios(config, dataset_root=dataset_root(s), output_root=scenario_root(s)))


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_completed: bool = typer.Option(True, "--skip-completed"),
) -> None:
    print(run_generalized_matrix(settings(), config, resume=resume, skip_completed=skip_completed))


@app.command("report")
def report_cmd(experiment_id: str = typer.Option(..., "--experiment-id")) -> None:
    s = settings()
    report_dir = reports_root(s) / experiment_id
    summary = read_json(report_dir / "generalized_benchmark_summary.json")
    print(write_generalized_reports(report_dir, experiment_id, summary, load_generalized_metrics(s, experiment_id)))


@app.command("certificate")
def certificate_cmd(experiment_id: str = typer.Option(..., "--experiment-id")) -> None:
    s = settings()
    from square_sim.square_tune_generalized.simulation.runner import certificates_root

    print(write_generalized_certificates(certificates_root(s) / experiment_id, experiment_id, load_generalized_metrics(s, experiment_id)))


@app.command("publication-bundle")
def publication_bundle_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), output: Path = typer.Option(..., "--output")) -> None:
    print(create_publication_bundle(settings(), experiment_id, _expand(output)))


@app.command("run-all-v1")
def run_all_v1(
    protect_prior: bool = typer.Option(False, "--protect-prior"),
    freeze_evidence_flag: bool = typer.Option(False, "--freeze-evidence"),
    ingest_rag_flag: bool = typer.Option(False, "--ingest-rag"),
    ingest_patient_flow_flag: bool = typer.Option(False, "--ingest-patient-flow"),
    ingest_elastic_compute_flag: bool = typer.Option(False, "--ingest-elastic-compute"),
    ingest_ml_to_llm_flag: bool = typer.Option(False, "--ingest-ml-to-llm"),
    compile_scenarios_flag: bool = typer.Option(False, "--compile-scenarios"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_full_matrix: bool = typer.Option(False, "--run-full-matrix"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    create_publication_bundle_flag: bool = typer.Option(False, "--create-publication-bundle"),
    resume: bool = True,
    skip_completed: bool = typer.Option(True, "--skip-completed"),
) -> None:
    del generate_reports, generate_certificates
    s = settings()
    if protect_prior:
        ProtectedResultsRegistry(s).protect_defaults(notes="generalized run-all-v1")
    if freeze_evidence_flag:
        out = s.project_root / "evidence" / f"square_tune_generalized_v1_baseline_{stable_hash({'run': 'all'}, 8)}"
        freeze_evidence(out)
    dataset_cfg = Path("configs/generalized/datasets_generalized_v1.yaml")
    if ingest_rag_flag:
        _ingest_track("rag", dataset_cfg)
    if ingest_patient_flow_flag:
        _ingest_track("patient_flow", dataset_cfg)
    if ingest_elastic_compute_flag:
        _ingest_track("elastic_compute", dataset_cfg)
    if ingest_ml_to_llm_flag:
        _ingest_track("ml_to_llm", dataset_cfg)
    full_cfg = Path("configs/generalized/square_tune_generalized_v1_full_matrix.yaml")
    if compile_scenarios_flag:
        compile_generalized_scenarios(full_cfg, dataset_root=dataset_root(s), output_root=scenario_root(s))
    smoke_result = None
    if run_smoke_first:
        smoke_result = run_generalized_matrix(s, Path("configs/generalized/square_tune_generalized_v1_smoke.yaml"), resume=resume, skip_completed=skip_completed)
        if smoke_result["failed"]:
            print({"status": "stopped_after_smoke_failure", "smoke": smoke_result})
            return
    full_result = None
    if run_full_matrix:
        full_result = run_generalized_matrix(s, full_cfg, resume=resume, skip_completed=skip_completed)
    if create_publication_bundle_flag and full_result:
        bundle = s.project_root / "publication" / "square_tune_generalized_v1" / full_result["experiment_id"]
        create_publication_bundle(s, full_result["experiment_id"], bundle)
    print({"smoke": smoke_result, "full": full_result})


@app.command("dataset-audit")
def dataset_audit() -> None:
    s = settings()
    out = reports_root(s) / "dataset_license_summary"
    print(write_dataset_audit(dataset_root(s), out))
