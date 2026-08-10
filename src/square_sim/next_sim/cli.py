from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from square_sim.config import Settings
from square_sim.next_sim.protected_results import protect_prior
from square_sim.next_sim.runner import (
    build_publication_bundle,
    plan_matrix,
    rerun_reports,
    run_matrix,
)
from square_sim.next_sim.runner import diagnose as diagnose_run
from square_sim.tune.external.protection import ProtectedResultsRegistry

app = typer.Typer(help="SQUARE Next Simulation Package v1")


def settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


@app.command("protect-prior")
def protect_prior_cmd(nas_root: Path | None = None) -> None:
    print(json.dumps(protect_prior(settings(nas_root)), indent=2))


@app.command("protection-status")
def protection_status(nas_root: Path | None = None) -> None:
    print(json.dumps(ProtectedResultsRegistry(settings(nas_root)).load(), indent=2))


@app.command("plan")
def plan_cmd(config: Path = typer.Option(..., "--config")) -> None:
    print(json.dumps(plan_matrix(config), indent=2))


@app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(False, "--resume"),
    skip_completed: bool = typer.Option(False, "--skip-completed"),
    nas_root: Path | None = None,
) -> None:
    print(json.dumps(run_matrix(settings(nas_root), config, resume=resume, skip_completed=skip_completed), indent=2, default=str))


@app.command("diagnose")
def diagnose_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(json.dumps(diagnose_run(settings(nas_root), experiment_id), indent=2))


@app.command("report")
def report_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(json.dumps(rerun_reports(settings(nas_root), experiment_id), indent=2, default=str))


@app.command("certificate")
def certificate_cmd(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(json.dumps(rerun_reports(settings(nas_root), experiment_id)["certificate"], indent=2, default=str))


@app.command("publication-bundle")
def publication_bundle_cmd(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    output: Path = typer.Option(..., "--output"),
    nas_root: Path | None = None,
) -> None:
    print(json.dumps(build_publication_bundle(settings(nas_root), experiment_id, output), indent=2, default=str))


@app.command("run-all-v1")
def run_all_v1(
    protect_prior_flag: bool = typer.Option(False, "--protect-prior"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_rag_hard_subset: bool = typer.Option(False, "--run-rag-hard-subset"),
    run_no_fork_robustness: bool = typer.Option(False, "--run-no-fork-robustness"),
    run_adaptive_escalation: bool = typer.Option(False, "--run-adaptive-escalation"),
    run_claim_faithfulness: bool = typer.Option(False, "--run-claim-faithfulness"),
    run_elastic_compute: bool = typer.Option(False, "--run-elastic-compute"),
    run_square_core_v2_targeted: bool = typer.Option(False, "--run-square-core-v2-targeted"),
    generate_diagnostics: bool = typer.Option(False, "--generate-diagnostics"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    create_publication_bundle_flag: bool = typer.Option(False, "--create-publication-bundle"),
    resume: bool = typer.Option(False, "--resume"),
    skip_completed: bool = typer.Option(False, "--skip-completed"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    if protect_prior_flag:
        protect_prior(s)
    configs: list[Path] = []
    if run_smoke_first:
        configs.append(Path("configs/next_sim/square_next_sim_v1_smoke.yaml"))
    if run_rag_hard_subset:
        configs.append(Path("configs/next_sim/rag_hard_subset_v1_full.yaml"))
    if run_no_fork_robustness:
        configs.append(Path("configs/next_sim/no_fork_robustness_v1_full.yaml"))
    if run_adaptive_escalation:
        configs.append(Path("configs/next_sim/adaptive_escalation_v2_full.yaml"))
    if run_claim_faithfulness:
        configs.append(Path("configs/next_sim/claim_level_faithfulness_v1_full.yaml"))
    if run_elastic_compute:
        configs.append(Path("configs/next_sim/elastic_compute_policy_v1_full.yaml"))
    if run_square_core_v2_targeted:
        configs.append(Path("configs/next_sim/square_core_v2_targeted_full.yaml"))
    results = [run_matrix(s, config, resume=resume, skip_completed=skip_completed) for config in configs]
    if results and (generate_diagnostics or generate_reports or generate_certificates or create_publication_bundle_flag):
        for result in results:
            experiment_id = str(result["experiment_id"])
            if generate_diagnostics:
                diagnose_run(s, experiment_id)
            if generate_reports or generate_certificates:
                rerun_reports(s, experiment_id)
            if create_publication_bundle_flag:
                build_publication_bundle(s, experiment_id)
    print(json.dumps({"runs": results}, indent=2, default=str))
