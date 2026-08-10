from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from square_sim.adaptive_arch.diagnostics import diagnose_experiment
from square_sim.adaptive_arch.generators import generate_suite, validate_suite
from square_sim.adaptive_arch.runner import (
    generate_certificate,
    generate_report,
    run_all_v1,
    run_benchmark,
)
from square_sim.config import Settings

app = typer.Typer(help="SQUARE adaptive architecture benchmark")


def _settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


@app.command("generate")
def generate(
    suite: str = typer.Option("square_adaptive_arch_v1", "--suite"),
    rows: int = typer.Option(50_000, "--rows"),
    seeds: str = typer.Option("101,202,303,404,505", "--seeds"),
    output: Path = typer.Option(..., "--output"),
    noise_level: float = typer.Option(0.04, "--noise-level"),
    difficulty: str = typer.Option("mixed", "--difficulty"),
) -> None:
    parsed = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    print(generate_suite(output, suite=suite, rows=rows, seeds=parsed, noise_level=noise_level, difficulty=difficulty))


@app.command("validate")
def validate(root: Path = typer.Option(..., "--root")) -> None:
    print(validate_suite(root))


@app.command("run")
def run(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    max_runs: int | None = typer.Option(None, "--max-runs"),
    nas_root: Path | None = None,
) -> None:
    print(run_benchmark(_settings(nas_root), config, resume=resume, skip_completed=skip_completed, max_runs=max_runs))


@app.command("diagnose")
def diagnose(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(diagnose_experiment(_settings(nas_root), experiment_id))


@app.command("report")
def report(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_report(_settings(nas_root), experiment_id))


@app.command("certificate")
def certificate(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_certificate(_settings(nas_root), experiment_id))


@app.command("run-all-v1")
def run_all(
    protect_prior: bool = typer.Option(False, "--protect-prior"),
    generate_synthetic: bool = typer.Option(False, "--generate-synthetic"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_synthetic_matrix: bool = typer.Option(False, "--run-synthetic-matrix"),
    run_external_proxy: bool = typer.Option(False, "--run-external-proxy"),
    generate_diagnostics: bool = typer.Option(False, "--generate-diagnostics"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    nas_root: Path | None = None,
) -> None:
    print(
        run_all_v1(
            _settings(nas_root),
            protect_prior=protect_prior,
            generate_synthetic=generate_synthetic,
            run_smoke_first=run_smoke_first,
            run_synthetic_matrix=run_synthetic_matrix,
            run_external_proxy=run_external_proxy,
            generate_diagnostics=generate_diagnostics,
            generate_reports=generate_reports,
            generate_certificates=generate_certificates,
            resume=resume,
            skip_completed=skip_completed,
        )
    )

