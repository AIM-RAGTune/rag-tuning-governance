from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from ragtune.config import SuiteConfig
from ragtune.experiments.runner import run_suite
from ragtune.phase2 import data_acquire, data_normalize, data_verify
from ragtune.validation_phase3 import run_public_corpus_acquisition

app = typer.Typer(help="RAGTune validation CLI")
data_app = typer.Typer(help="RAGTune public data commands")
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Run RAGTune validation suites."""


@app.command("run-suite")
def run_suite_cmd(
    suite: str = typer.Option(..., "--suite"),
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(Path("artifacts/runs"), "--output-dir"),
    run_id: str = typer.Option("auto", "--run-id"),
    resume: bool = typer.Option(False, "--resume"),
    force_new_run_id: bool = typer.Option(False, "--force-new-run-id"),
) -> None:
    print(
        json.dumps(
            run_suite(
                suite=suite,
                config_path=config,
                output_dir=output_dir,
                run_id=run_id,
                resume=resume,
                force_new_run_id=force_new_run_id,
            ),
            indent=2,
            default=str,
        )
    )


@data_app.command("acquire")
def data_acquire_cmd(
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(Path("artifacts/datasets/raw"), "--output-dir"),
) -> None:
    print(json.dumps(data_acquire(config, output_dir), indent=2, default=str))


@data_app.command("verify")
def data_verify_cmd(manifest: Path = typer.Option(..., "--manifest")) -> None:
    print(json.dumps(data_verify(manifest), indent=2, default=str))


@data_app.command("normalize")
def data_normalize_cmd(
    manifest: Path = typer.Option(..., "--manifest"),
    output_dir: Path = typer.Option(Path("artifacts/datasets/normalized"), "--output-dir"),
) -> None:
    print(json.dumps(data_normalize(manifest, output_dir), indent=2, default=str))


@data_app.command("manifest")
def data_manifest_cmd(manifest: Path = typer.Option(..., "--manifest")) -> None:
    print(json.dumps(data_verify(manifest), indent=2, default=str))


@data_app.command("availability-report")
def data_availability_cmd(manifest: Path = typer.Option(..., "--manifest")) -> None:
    print(json.dumps(data_verify(manifest), indent=2, default=str))


@data_app.command("acquire-public-corpus")
def data_acquire_public_corpus_cmd(
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(Path("artifacts/datasets"), "--output-dir"),
    run_id: str = typer.Option("auto", "--run-id"),
) -> None:
    cfg = SuiteConfig.from_path(config)
    print(
        json.dumps(
            run_public_corpus_acquisition(
                cfg,
                config,
                output_dir,
                run_id,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    app()
