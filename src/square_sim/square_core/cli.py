from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from square_sim.config import Settings
from square_sim.square_core.matrix.plan import plan_matrix
from square_sim.square_core.matrix.runner import certificate, diagnose, report, run_core_matrix
from square_sim.tune.external.protection import ProtectedResultsRegistry

app = typer.Typer(help="SQUARE Core Validation Matrix v1")


def _settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


@app.command("protect-prior")
def protect_prior(nas_root: Path | None = None) -> None:
    print(ProtectedResultsRegistry(_settings(nas_root)).protect_defaults(notes="SQUARE Core Validation Matrix prior protection"))


@app.command("plan")
def plan(config: Path = typer.Option(..., "--config")) -> None:
    print(plan_matrix(config))


@app.command("run")
def run(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    nas_root: Path | None = None,
) -> None:
    print(run_core_matrix(_settings(nas_root), config, resume=resume, skip_completed=skip_completed))


@app.command("diagnose")
def diagnose_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(diagnose(_settings(nas_root), experiment_id))


@app.command("report")
def report_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(report(_settings(nas_root), experiment_id))


@app.command("certificate")
def certificate_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(certificate(_settings(nas_root), experiment_id))


@app.command("run-all-v1")
def run_all_v1(
    protect_prior_flag: bool = typer.Option(False, "--protect-prior"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_adaptive_arch: bool = typer.Option(False, "--run-adaptive-arch"),
    run_field_substrate: bool = typer.Option(False, "--run-field-substrate"),
    run_closed_loop: bool = typer.Option(False, "--run-closed-loop"),
    run_quantum_coupling_toy: bool = typer.Option(False, "--run-quantum-coupling-toy"),
    run_soliton: bool = typer.Option(False, "--run-soliton"),
    generate_diagnostics: bool = typer.Option(False, "--generate-diagnostics"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    nas_root: Path | None = None,
) -> None:
    del generate_diagnostics, generate_reports, generate_certificates
    settings = _settings(nas_root)
    if protect_prior_flag:
        ProtectedResultsRegistry(settings).protect_defaults(notes="SQUARE Core Validation Matrix prior protection")
    configs = []
    base = Path("configs/square_core")
    if run_smoke_first:
        configs.append(base / "square_core_validation_v1_smoke.yaml")
    if run_adaptive_arch:
        configs.append(base / "square_core_validation_v1_adaptive_arch_full.yaml")
    if run_field_substrate:
        configs.append(base / "square_core_validation_v1_field_substrate.yaml")
    if run_closed_loop:
        configs.append(base / "square_core_validation_v1_closed_loop.yaml")
    if run_quantum_coupling_toy:
        configs.append(base / "square_core_validation_v1_quantum_coupling_toy.yaml")
    if run_soliton:
        configs.append(base / "square_core_validation_v1_soliton.yaml")
    print([run_core_matrix(settings, config, resume=resume, skip_completed=skip_completed) for config in configs])
