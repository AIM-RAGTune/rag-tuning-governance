from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from square_sim.config import Settings
from square_sim.tune.cli_calibration import app as calibration_app
from square_sim.tune.config import parse_csv_ints
from square_sim.tune.experiments.comparisons import load_experiment_metrics
from square_sim.tune.experiments.matrix import plan_tune_matrix
from square_sim.tune.experiments.runner import generate_reports, run_tune_matrix
from square_sim.tune.external.acquire import acquire_external
from square_sim.tune.external.audit import data_audit, scenario_audit
from square_sim.tune.external.catalog import (
    list_external_catalog,
    refresh_catalog,
    show_external_dataset,
)
from square_sim.tune.external.diagnostics import diagnose_adaptive_compute, diagnose_cost
from square_sim.tune.external.manual_import import manual_import_dataset
from square_sim.tune.external.paths import external_root
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.tune.external.runner import (
    generate_external_certificate,
    generate_external_report,
    run_all_external_v1,
    run_external_matrix,
)
from square_sim.tune.external.scenarios import compile_scenarios
from square_sim.tune.peft_optional.local_lora_smoke import run_peft_smoke
from square_sim.tune.reporting.certificate import write_certificates
from square_sim.tune.synthetic.generators import (
    describe_mechanism,
    generate_suite,
    list_generated_datasets,
)
from square_sim.tune.synthetic.validators import validate_suite

app = typer.Typer(help="SQUARETune LLM adaptation simulation")
synthetic_app = typer.Typer(help="Synthetic SQUARETune datasets")
external_app = typer.Typer(help="Optional external dataset acquisition")
adaptive_compute_app = typer.Typer(help="Adaptive-compute external-transfer experiments")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(external_app, name="external")
app.add_typer(calibration_app, name="calibration")
external_app.add_typer(adaptive_compute_app, name="adaptive-compute")


def _settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


@synthetic_app.command("generate")
def synthetic_generate(
    suite: str = typer.Option("llm_tuning_v1", "--suite"),
    rows: int = typer.Option(50_000, "--rows"),
    seeds: str = typer.Option("101,202,303,404,505", "--seeds"),
    output: Path = typer.Option(..., "--output"),
    noise_level: float = typer.Option(0.05, "--noise-level"),
    difficulty: str = typer.Option("mixed", "--difficulty"),
    write_latent: bool = typer.Option(False, "--write-latent"),
) -> None:
    print(
        generate_suite(
            output,
            suite=suite,
            rows=rows,
            seeds=parse_csv_ints(seeds),
            noise_level=noise_level,
            difficulty=difficulty,
            write_latent=write_latent,
        )
    )


@synthetic_app.command("list")
def synthetic_list(root: Path = typer.Option(Path("./tmp/square_tune_synthetic"), "--root")) -> None:
    print(list_generated_datasets(root))


@synthetic_app.command("describe")
def synthetic_describe(dataset: str = typer.Option(..., "--dataset")) -> None:
    print(describe_mechanism(dataset))


@synthetic_app.command("validate")
def synthetic_validate(
    all_datasets: bool = typer.Option(False, "--all"),
    root: Path = typer.Option(Path("./tmp/square_tune_synthetic"), "--root"),
) -> None:
    if not all_datasets:
        raise typer.BadParameter("Use --all to validate a generated suite root.")
    print(validate_suite(root))


@app.command("plan")
def tune_plan(config: Path = typer.Option(..., "--config")) -> None:
    print(plan_tune_matrix(config))


@app.command("run")
def tune_run(config: Path = typer.Option(..., "--config"), device: str = typer.Option("cpu", "--device"), nas_root: Path | None = None) -> None:
    print(run_tune_matrix(_settings(nas_root), config, device=device, max_runs=None))


@app.command("run-matrix")
def tune_run_matrix(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    max_runs: int | None = typer.Option(None, "--max-runs"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    print(
        run_tune_matrix(
            _settings(nas_root),
            config,
            device=device,
            resume=resume,
            skip_completed=skip_completed,
            max_runs=max_runs,
        )
    )


@app.command("report")
def tune_report(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_reports(_settings(nas_root), experiment_id))


@app.command("certificate")
def tune_certificate(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    s = _settings(nas_root)
    metrics = load_experiment_metrics(s.project_root, experiment_id)
    print(write_certificates(s.project_root / "certificates" / "square_tune" / experiment_id, experiment_id, metrics))


@external_app.command("acquire")
def external_acquire(
    config: Path = typer.Option(..., "--config"),
    allow_unknown_license: bool = typer.Option(False, "--allow-unknown-license"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    allow_optional: bool = typer.Option(False, "--allow-optional"),
    require_all: bool = typer.Option(False, "--require-all"),
    max_rows: int | None = typer.Option(None, "--max-rows"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing"),
    nas_root: Path | None = None,
) -> None:
    _ = resume
    print(
        acquire_external(
            config,
            allow_unknown_license=allow_unknown_license,
            output_root=output_root,
            allow_optional=allow_optional,
            require_all=require_all,
            max_rows=max_rows,
            settings=_settings(nas_root),
            skip_existing=skip_existing,
        )
    )


@external_app.command("protect-prior")
def external_protect_prior(nas_root: Path | None = None) -> None:
    print(ProtectedResultsRegistry(_settings(nas_root)).protect_defaults())


@external_app.command("protection-status")
def external_protection_status(nas_root: Path | None = None) -> None:
    print(ProtectedResultsRegistry(_settings(nas_root)).load())


@external_app.command("import-manual")
def external_import_manual(
    dataset: str = typer.Option(..., "--dataset"),
    path: Path = typer.Option(..., "--path"),
    license_note: Path | None = typer.Option(None, "--license-note"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    scenario_families: str = typer.Option("external_transfer", "--scenario-families"),
    allow_unknown_license: bool = typer.Option(False, "--allow-unknown-license"),
    max_rows: int | None = typer.Option(None, "--max-rows"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    print(
        manual_import_dataset(
            dataset,
            path,
            output_root=output_root or external_root(s),
            license_note=license_note,
            scenario_families=[item.strip() for item in scenario_families.split(",") if item.strip()],
            allow_unknown_license=allow_unknown_license,
            max_rows=max_rows,
        )
    )


@external_app.command("catalog")
def external_catalog(
    action: str = typer.Argument("list"),
    dataset: str | None = typer.Option(None, "--dataset"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    nas_root: Path | None = None,
) -> None:
    root = output_root or external_root(_settings(nas_root))
    if action == "refresh":
        print(refresh_catalog(root))
    elif action == "show":
        if not dataset:
            raise typer.BadParameter("Use --dataset with catalog show.")
        print(show_external_dataset(root, dataset))
    elif action == "list":
        print(list_external_catalog(root))
    else:
        raise typer.BadParameter("Use catalog action: list, show, or refresh.")


@external_app.command("compile-scenarios")
def external_compile_scenarios(
    config: Path = typer.Option(..., "--config"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    seed: int = typer.Option(101, "--seed"),
    source_appropriate: bool = typer.Option(False, "--source-appropriate"),
) -> None:
    print(compile_scenarios(config, output_root=output_root, seed=seed, source_appropriate=source_appropriate))


@external_app.command("data-audit")
def external_data_audit(
    root: Path = typer.Option(..., "--root"),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    print(data_audit(root, strict=strict))


@external_app.command("scenario-audit")
def external_scenario_audit(
    scenario_root: Path = typer.Option(..., "--scenario-root"),
    strict: bool = typer.Option(False, "--strict"),
) -> None:
    print(scenario_audit(scenario_root, strict=strict))


@external_app.command("run-smoke")
def external_run_smoke(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    print(run_external_matrix(_settings(nas_root), config, device=device, resume=resume, skip_completed=skip_completed, smoke=True))


@external_app.command("run-v1")
def external_run_v1(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    allow_optional: bool = typer.Option(False, "--allow-optional"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    _ = allow_optional
    print(run_external_matrix(_settings(nas_root), config, device=device, resume=resume, skip_completed=skip_completed))


@external_app.command("report")
def external_report(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_external_report(_settings(nas_root), experiment_id))


@external_app.command("certificate")
def external_certificate(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_external_certificate(_settings(nas_root), experiment_id))


@external_app.command("diagnose-cost")
def external_diagnose_cost(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(diagnose_cost(_settings(nas_root), experiment_id))


@adaptive_compute_app.command("smoke")
def adaptive_compute_smoke(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    print(run_external_matrix(_settings(nas_root), config, device=device, resume=resume, skip_completed=skip_completed, smoke=True))


@adaptive_compute_app.command("run")
def adaptive_compute_run(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    print(run_external_matrix(_settings(nas_root), config, device=device, resume=resume, skip_completed=skip_completed))


@adaptive_compute_app.command("diagnose")
def adaptive_compute_diagnose(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(diagnose_adaptive_compute(_settings(nas_root), experiment_id))


@adaptive_compute_app.command("report")
def adaptive_compute_report(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    s = _settings(nas_root)
    report = generate_external_report(s, experiment_id)
    diagnostics = diagnose_adaptive_compute(s, experiment_id)
    print({"report": report, "adaptive_diagnostics": diagnostics})


@adaptive_compute_app.command("certificate")
def adaptive_compute_certificate(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    print(generate_external_certificate(_settings(nas_root), experiment_id))


@adaptive_compute_app.command("run-all-v2")
def adaptive_compute_run_all_v2(
    protect_prior: bool = typer.Option(False, "--protect-prior"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_rag_policy: bool = typer.Option(False, "--run-rag-policy"),
    run_faithfulness: bool = typer.Option(False, "--run-faithfulness"),
    run_full: bool = typer.Option(False, "--run-full"),
    generate_diagnostics: bool = typer.Option(False, "--generate-diagnostics"),
    generate_reports_flag: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    result: dict[str, object] = {}
    summaries = []
    if protect_prior:
        result["protection"] = ProtectedResultsRegistry(s).protect_defaults()
    if run_smoke_first:
        summary = run_external_matrix(
            s,
            Path("configs/tune/external_transfer/external_transfer_v2_adaptive_compute_smoke.yaml"),
            device=device,
            resume=resume,
            skip_completed=skip_completed,
            smoke=True,
        )
        summaries.append(summary)
        result["smoke"] = summary
    if run_rag_policy:
        summary = run_external_matrix(
            s,
            Path("configs/tune/external_transfer/external_transfer_v2_adaptive_compute_rag_policy_only.yaml"),
            device=device,
            resume=resume,
            skip_completed=skip_completed,
        )
        summaries.append(summary)
        result["rag_policy"] = summary
    if run_faithfulness:
        summary = run_external_matrix(
            s,
            Path("configs/tune/external_transfer/external_transfer_v2_adaptive_compute_faithfulness_only.yaml"),
            device=device,
            resume=resume,
            skip_completed=skip_completed,
        )
        summaries.append(summary)
        result["faithfulness"] = summary
    if run_full:
        summary = run_external_matrix(
            s,
            Path("configs/tune/external_transfer/external_transfer_v2_adaptive_compute_ragtruth.yaml"),
            device=device,
            resume=resume,
            skip_completed=skip_completed,
        )
        summaries.append(summary)
        result["full"] = summary
    for summary in summaries:
        experiment_id = str(summary.get("experiment_id"))
        if generate_diagnostics:
            result[f"{experiment_id}_diagnostics"] = diagnose_adaptive_compute(s, experiment_id)
        if generate_reports_flag:
            result[f"{experiment_id}_report"] = generate_external_report(s, experiment_id)
        if generate_certificates:
            result[f"{experiment_id}_certificate"] = generate_external_certificate(s, experiment_id)
    print(result)


@external_app.command("run-all-v1")
def external_run_all_v1(
    protect_prior: bool = typer.Option(False, "--protect-prior"),
    acquire: bool = typer.Option(False, "--acquire"),
    compile_scenarios_flag: bool = typer.Option(False, "--compile-scenarios"),
    run_smoke_first: bool = typer.Option(False, "--run-smoke-first"),
    run_full_if_smoke_passes: bool = typer.Option(False, "--run-full-if-smoke-passes"),
    generate_reports_flag: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    nas_root: Path | None = None,
) -> None:
    print(
        run_all_external_v1(
            _settings(nas_root),
            protect_prior=protect_prior,
            acquire=acquire,
            compile=compile_scenarios_flag,
            run_smoke_first=run_smoke_first,
            run_full_if_smoke_passes=run_full_if_smoke_passes,
            generate_reports_flag=generate_reports_flag,
            generate_certificates=generate_certificates,
            resume=resume,
            skip_completed=skip_completed,
        )
    )


@external_app.command("run-full-v1")
def external_run_full_v1(
    protect_prior: bool = typer.Option(False, "--protect-prior"),
    acquire_full: bool = typer.Option(False, "--acquire-full"),
    compile_source_appropriate_scenarios: bool = typer.Option(False, "--compile-source-appropriate-scenarios"),
    audit_data: bool = typer.Option(False, "--audit-data"),
    audit_scenarios: bool = typer.Option(False, "--audit-scenarios"),
    run_minimal_expanded: bool = typer.Option(False, "--run-minimal-expanded"),
    diagnose_cost_flag: bool = typer.Option(False, "--diagnose-cost"),
    generate_reports_flag: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    full_root = s.project_root / "datasets" / "external" / "square_tune_v1_full"
    result: dict[str, object] = {}
    if protect_prior:
        result["protection"] = ProtectedResultsRegistry(s).protect_defaults()
    if acquire_full:
        result["acquisition"] = acquire_external(
            Path("configs/tune/external_transfer/datasets_full_v1.yaml"),
            output_root=full_root,
            allow_optional=True,
            settings=s,
            skip_existing=True,
        )
    if audit_data:
        result["data_audit"] = data_audit(full_root)
    if compile_source_appropriate_scenarios:
        result["scenario_compilation"] = compile_scenarios(
            Path("configs/tune/external_transfer/external_transfer_v1_full.yaml"),
            output_root=full_root / "scenarios",
            source_appropriate=True,
        )
    if audit_scenarios:
        result["scenario_audit"] = scenario_audit(full_root / "scenarios")
    run_summary = None
    if run_minimal_expanded:
        run_summary = run_external_matrix(
            s,
            Path("configs/tune/external_transfer/external_transfer_v1_minimal_expanded.yaml"),
            resume=resume,
            skip_completed=skip_completed,
        )
        result["run_summary"] = run_summary
    if run_summary and diagnose_cost_flag:
        result["cost_diagnostics"] = diagnose_cost(s, str(run_summary["experiment_id"]))
    if run_summary and generate_reports_flag:
        result["report"] = generate_external_report(s, str(run_summary["experiment_id"]))
    if run_summary and generate_certificates:
        result["certificate"] = generate_external_certificate(s, str(run_summary["experiment_id"]))
    print(result)


@app.command("peft-smoke")
def peft_smoke(config: Path = typer.Option(..., "--config"), device: str = typer.Option("cuda:0", "--device")) -> None:
    print(run_peft_smoke(config, device=device))


@app.command("demo")
def tune_demo(
    rows: int = typer.Option(1000, "--rows"),
    seed: int = typer.Option(101, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    root = s.project_root / "datasets" / "synthetic" / "square_tune_demo"
    generate_suite(
        root,
        rows=rows,
        seeds=[seed],
        datasets=[
            "synthetic_llm_linear_control",
            "synthetic_llm_random_label",
            "synthetic_llm_failure_cluster_routing",
        ],
    )
    config_path = s.project_root / "protocols" / "square_tune" / "demo_runtime_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""experiment_name: square_tune_demo
dataset_root: {root}
datasets:
  - synthetic_llm_linear_control
  - synthetic_llm_random_label
  - synthetic_llm_failure_cluster_routing
seeds: [{seed}]
optimizers:
  - random_search
  - greedy_eval_improvement
  - square_tune_full
  - square_tune_no_fork
  - square_tune_linear_rollout
  - square_tune_no_merge
square_tune:
  max_rounds: 4
  num_branches: 4
  rollout_steps: 2
""",
        encoding="utf-8",
    )
    print(run_tune_matrix(s, config_path, device=device, skip_completed=False))
