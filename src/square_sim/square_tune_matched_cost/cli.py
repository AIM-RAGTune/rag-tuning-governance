from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import print

from square_sim.config import Settings
from square_sim.square_tune_matched_cost.config import MatchedCostRAGConfig
from square_sim.square_tune_matched_cost.datasets import ingest_matched_cost_rag
from square_sim.square_tune_matched_cost.runner import (
    build_publication_bundle,
    ensure_ingested,
    ensure_scenario,
    plan_matrix,
    protect_prior,
    rerun_reports,
    run_matrix,
)
from square_sim.square_tune_matched_cost.scenario_compile import compile_matched_cost_scenario

app = typer.Typer(help="SQUARETune matched-cost real-RAG kill-test")


def settings() -> Settings:
    return Settings.from_env()


def _expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


@app.command("protect-prior")
def protect_prior_cmd() -> None:
    print(protect_prior(settings()))


@app.command("ingest")
def ingest_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    cfg = MatchedCostRAGConfig.from_path(config)
    print(ingest_matched_cost_rag(settings(), max_rows=cfg.max_rows))


@app.command("compile-scenarios")
def compile_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_existing: bool = typer.Option(True, "--skip-existing"),
) -> None:
    del resume, skip_existing
    cfg = MatchedCostRAGConfig.from_path(config)
    print(compile_matched_cost_scenario(settings(), max_rows=cfg.max_rows))


@app.command("plan")
def plan_cmd(config: Path = typer.Option(..., "--config")) -> None:
    print(plan_matrix(config))


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = True,
    skip_completed: bool = typer.Option(True, "--skip-completed"),
) -> None:
    print(run_matrix(settings(), config, resume=resume, skip_completed=skip_completed))


@app.command("sensitivity")
def sensitivity_cmd(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    config: Path = typer.Option(..., "--config"),
) -> None:
    cfg = MatchedCostRAGConfig.from_path(config)
    print(rerun_reports(settings(), experiment_id, bootstrap_samples=cfg.bootstrap_samples))


@app.command("diagnose")
def diagnose_cmd(experiment_id: str = typer.Option(..., "--experiment-id")) -> None:
    print(rerun_reports(settings(), experiment_id))


@app.command("report")
def report_cmd(experiment_id: str = typer.Option(..., "--experiment-id")) -> None:
    print(rerun_reports(settings(), experiment_id))


@app.command("certificate")
def certificate_cmd(experiment_id: str = typer.Option(..., "--experiment-id")) -> None:
    print(rerun_reports(settings(), experiment_id))


@app.command("publication-bundle")
def publication_bundle_cmd(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    print(build_publication_bundle(settings(), experiment_id, _expand(output)))


@app.command("run-all-v1")
def run_all_v1(
    protect_prior_flag: bool = typer.Option(False, "--protect-prior"),
    ingest_flag: bool = typer.Option(False, "--ingest"),
    compile_scenarios_flag: bool = typer.Option(False, "--compile-scenarios"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_full_matrix: bool = typer.Option(False, "--run-full-matrix"),
    run_sensitivity: bool = typer.Option(False, "--run-sensitivity"),
    generate_diagnostics: bool = typer.Option(False, "--generate-diagnostics"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    create_publication_bundle_flag: bool = typer.Option(False, "--create-publication-bundle"),
    resume: bool = True,
    skip_completed: bool = typer.Option(True, "--skip-completed"),
) -> None:
    del generate_diagnostics, generate_reports, generate_certificates
    s = settings()
    if protect_prior_flag:
        protect_prior(s)
    smoke_config = Path("configs/matched_cost_rag/square_tune_matched_cost_rag_v1_smoke.yaml")
    full_config = Path("configs/matched_cost_rag/square_tune_matched_cost_rag_v1_full_matrix.yaml")
    if ingest_flag:
        ensure_ingested(s, full_config)
    if compile_scenarios_flag:
        ensure_scenario(s, full_config)
    smoke = run_matrix(s, smoke_config, resume=resume, skip_completed=skip_completed) if run_smoke_first else None
    full = None
    if run_full_matrix:
        full = run_matrix(s, full_config, resume=resume, skip_completed=skip_completed)
        if run_sensitivity:
            rerun_reports(s, str(full["experiment_id"]))
        if create_publication_bundle_flag:
            build_publication_bundle(
                s,
                str(full["experiment_id"]),
                s.project_root / "publication" / "square_tune_matched_cost_rag" / "v1" / str(full["experiment_id"]),
            )
    print({"smoke": smoke, "full": full})

