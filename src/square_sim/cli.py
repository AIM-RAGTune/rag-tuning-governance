from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from square_sim.adaptive_arch.cli import app as adaptive_arch_app
from square_sim.config import Settings
from square_sim.data.acquire import acquire_dataset
from square_sim.data.catalog import load_dataset_configs, show_catalog
from square_sim.data.ensure_splits import build_ensure_requests, ensure_splits
from square_sim.data.normalize import normalize_dataset
from square_sim.data.split import create_split
from square_sim.data.validate import validate_processed_schema
from square_sim.next_sim.cli import app as next_sim_app
from square_sim.orchestration.jobs import create_job, get_job, list_jobs, set_job_status
from square_sim.orchestration.matrix import expand_matrix, run_matrix, write_plan
from square_sim.orchestration.matrix import preflight as run_preflight
from square_sim.orchestration.worker import run_worker
from square_sim.paths import LabPaths
from square_sim.registry.repositories import RunRepository
from square_sim.reporting.bootstrap_compare import generate_bootstrap_comparisons
from square_sim.reporting.certificate import (
    generate_certificate_report,
    generate_certificates_for_experiment,
)
from square_sim.square_core.cli import app as square_core_app
from square_sim.square_tune_generalized.cli import app as generalized_app
from square_sim.square_tune_matched_cost.cli import app as matched_cost_rag_app
from square_sim.system.environment_snapshot import write_environment_snapshot
from square_sim.system.gpu import gpu_info
from square_sim.system.health import health, path_check
from square_sim.training.train import run_experiment_config, run_single_model
from square_sim.tune.cli import app as tune_app

app = typer.Typer(help="AIM SQUARESim Lab CLI")
data_app = typer.Typer(help="Dataset acquisition and preparation")
catalog_app = typer.Typer(help="Dataset catalog")
report_app = typer.Typer(help="Reports and comparisons")
compare_app = typer.Typer(help="Paired comparisons")
certificate_app = typer.Typer(help="Certificate generation")
jobs_app = typer.Typer(help="Job lifecycle")
smoke_app = typer.Typer(help="Smoke validation")
synthetic_app = typer.Typer(help="Synthetic diagnostic datasets")
app.add_typer(data_app, name="data")
data_app.add_typer(catalog_app, name="catalog")
app.add_typer(report_app, name="report")
app.add_typer(compare_app, name="compare")
app.add_typer(certificate_app, name="certificate")
app.add_typer(jobs_app, name="jobs")
app.add_typer(smoke_app, name="smoke")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(tune_app, name="tune")
app.add_typer(adaptive_arch_app, name="adaptive-arch")
app.add_typer(square_core_app, name="square-core")
app.add_typer(generalized_app, name="generalized")
app.add_typer(matched_cost_rag_app, name="matched-cost-rag")
app.add_typer(next_sim_app, name="next-sim")


def settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


@app.command("health")
def health_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    """Report system status."""
    payload = health(settings())
    print(json.dumps(payload, indent=2) if json_output else payload)


@app.command("gpu-info")
def gpu_info_cmd() -> None:
    print(json.dumps(gpu_info(), indent=2))


@app.command("path-check")
def path_check_cmd(nas_root: Path | None = None, create: bool = True) -> None:
    s = settings(nas_root)
    if create:
        LabPaths.from_settings(s).ensure_layout()
    print(json.dumps(path_check(s), indent=2))


@app.command("snapshot-env")
def snapshot_env(output: Path = Path("environment.json")) -> None:
    print(write_environment_snapshot(output, Path.cwd()))


@data_app.command("acquire")
def data_acquire(
    dataset: str | None = typer.Option(None),
    all_datasets: bool = typer.Option(False, "--all"),
    nas_root: Path | None = None,
    force: bool = False,
    offline_zip: Path | None = typer.Option(None, "--offline-zip"),
    no_extract: bool = False,
    verify_only: bool = False,
) -> None:
    s = settings(nas_root)
    names = list(load_dataset_configs()) if all_datasets else [dataset]
    if not names or names == [None]:
        raise typer.BadParameter("Provide --dataset or --all.")
    for name in names:
        print(acquire_dataset(str(name), s, force, offline_zip, no_extract, verify_only))


@data_app.command("validate")
def data_validate(
    dataset: str = typer.Option(..., "--dataset"),
    target: str | None = typer.Option(None, "--target"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    cfg = load_dataset_configs()[dataset]
    root = s.project_root / "datasets" / "processed" / dataset
    versions = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    if not versions:
        raise typer.BadParameter(f"No processed versions found for {dataset}.")
    print(validate_processed_schema(versions[-1] / "schema.json", cfg.expected_targets, target))


@data_app.command("normalize")
def data_normalize(
    dataset: str = typer.Option(..., "--dataset"),
    nas_root: Path | None = None,
    version: str | None = None,
) -> None:
    print(normalize_dataset(dataset, settings(nas_root), version))


@data_app.command("split")
def data_split(
    dataset: str = typer.Option(..., "--dataset"),
    split_id: str = "default",
    seed: int = 42,
    target: str = "target",
    method: str = "random",
    nas_root: Path | None = None,
) -> None:
    print(create_split(dataset, settings(nas_root), split_id, seed, target, method))


@data_app.command("ensure-splits")
def data_ensure_splits(
    dataset: str | None = typer.Option(None, "--dataset"),
    all_datasets: bool = typer.Option(False, "--all-datasets"),
    targets: str = typer.Option("target", "--targets"),
    target: str | None = typer.Option(None, "--target"),
    seed: int = typer.Option(42, "--seed"),
    create: bool = typer.Option(False, "--create"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    require_existing: bool = typer.Option(False, "--require-existing"),
    split_method: str = typer.Option("stratified", "--split-method"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    dataset_names = list(load_dataset_configs()) if all_datasets else ([dataset] if dataset else None)
    if not dataset_names:
        raise typer.BadParameter("Provide --dataset or --all-datasets.")
    target_names = [target] if target else [item.strip() for item in targets.split(",") if item.strip()]
    payload = ensure_splits(
        s,
        build_ensure_requests(
            datasets=[str(name) for name in dataset_names],
            targets=target_names,
            seed=seed,
            split_method=split_method,
        ),
        create=create,
        dry_run=dry_run,
        require_existing=require_existing,
    )
    print(payload)


@data_app.command("profile")
def data_profile(dataset: str = typer.Option(..., "--dataset"), nas_root: Path | None = None) -> None:
    s = settings(nas_root)
    root = s.project_root / "datasets" / "processed" / dataset
    versions = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    if not versions:
        raise typer.BadParameter(f"No processed versions found for {dataset}.")
    print(json.loads((versions[-1] / "profile.json").read_text(encoding="utf-8")))


@catalog_app.command("list")
def catalog_list(nas_root: Path | None = None) -> None:
    s = settings(nas_root)
    print({name: show_catalog(s.project_root, name) for name in load_dataset_configs()})


@catalog_app.command("show")
def catalog_show(dataset: str = typer.Option(..., "--dataset"), nas_root: Path | None = None) -> None:
    print(show_catalog(settings(nas_root).project_root, dataset))


@app.command("run")
def run_cmd(
    config: Path | None = typer.Option(None, "--config"),
    dataset: str | None = typer.Option(None, "--dataset"),
    target: str = typer.Option("target", "--target"),
    model: str = typer.Option("squaresim_full", "--model"),
    device: str = typer.Option("cpu", "--device"),
    split_id: str = typer.Option("default", "--split-id"),
    seed: int = typer.Option(42, "--seed"),
    max_epochs: int = typer.Option(5, "--max-epochs"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    if config:
        print(run_experiment_config(config, s))
    else:
        if not dataset:
            raise typer.BadParameter("Provide --config or --dataset.")
        print(
            run_single_model(
                s,
                dataset,
                target,
                model,
                split_id=split_id,
                seed=seed,
                device=device,
                max_epochs=max_epochs,
            )
        )


@app.command("preflight")
def preflight_cmd(
    config: Path = typer.Option(..., "--config"),
    ensure_missing_splits: bool = typer.Option(False, "--ensure-splits"),
    nas_root: Path | None = None,
) -> None:
    print(run_preflight(settings(nas_root), config, ensure_missing_splits=ensure_missing_splits))


@app.command("plan")
def plan_cmd(
    config: Path = typer.Option(..., "--config"),
    skip_completed: bool = typer.Option(False, "--skip-completed"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    experiment_id, planned = expand_matrix(s, config, skip_completed=skip_completed)
    print(write_plan(s, experiment_id, planned))


def _csv_opt(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


@app.command("run-matrix")
def run_matrix_cmd(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    only_missing: bool = typer.Option(False, "--only-missing"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    max_runs: int | None = typer.Option(None, "--max-runs"),
    datasets: str | None = typer.Option(None, "--datasets"),
    targets: str | None = typer.Option(None, "--targets"),
    models: str | None = typer.Option(None, "--models"),
    device: str | None = typer.Option(None, "--device"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    nas_root: Path | None = None,
) -> None:
    print(
        run_matrix(
            settings(nas_root),
            config,
            resume=resume,
            skip_completed=skip_completed,
            only_missing=only_missing,
            retry_failed=retry_failed,
            max_runs=max_runs,
            datasets_filter=_csv_opt(datasets),
            targets_filter=_csv_opt(targets),
            models_filter=_csv_opt(models),
            device_override=device,
            dry_run=dry_run,
        )
    )


@app.command("run-spectra")
def run_spectra_cmd(
    config: Path = typer.Option(Path("configs/experiments/full_spectra_matrix.yaml"), "--config"),
    ensure_missing_splits: bool = typer.Option(False, "--ensure-splits"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    generate_reports: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    if ensure_missing_splits:
        run_preflight(s, config, ensure_missing_splits=True)
    summary = run_matrix(s, config, resume=resume, skip_completed=skip_completed)
    experiment_id = str(summary["experiment_id"])
    if generate_reports:
        generate_bootstrap_comparisons(s, experiment_id)
    if generate_certificates:
        generate_certificates_for_experiment(s, experiment_id)
    print(summary)


@app.command("ablate")
def ablate(
    dataset: str | None = typer.Option(None, "--dataset"),
    all_datasets: bool = typer.Option(False, "--all-datasets"),
    target: str = typer.Option("target", "--target"),
    device: str = typer.Option("cpu", "--device"),
    nas_root: Path | None = None,
) -> None:
    from square_sim.models.ablations import SQUARESIM_MODELS

    names = list(load_dataset_configs()) if all_datasets else [dataset]
    if not names or names == [None]:
        raise typer.BadParameter("Provide --dataset or --all-datasets.")
    for name in names:
        for model_name in SQUARESIM_MODELS:
            print(run_single_model(settings(nas_root), str(name), target, model_name, device=device))


@report_app.command("run")
def report_run(run_id: str = typer.Option(..., "--run-id"), nas_root: Path | None = None) -> None:
    record = RunRepository(settings(nas_root).database_url).get(run_id)
    if not record:
        raise typer.BadParameter(f"Run not found: {run_id}")
    print(record.get("explanation_path"))


@report_app.command("compare")
def report_compare(
    dataset: str = typer.Option(..., "--dataset"),
    target: str = typer.Option(..., "--target"),
    nas_root: Path | None = None,
) -> None:
    from square_sim.reporting.compare_runs import compare_metrics_rows

    s = settings(nas_root)
    rows = []
    for run in RunRepository(s.database_url).list(dataset, target):
        path = run.get("metrics_path")
        if path and Path(path).exists():
            rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    output = s.project_root / "reports" / "comparison_reports" / f"{dataset}-{target}.md"
    compare_metrics_rows(output, rows)
    print(str(output))


@report_app.command("certificate")
def report_certificate(
    dataset: str = typer.Option(..., "--dataset"),
    target: str = typer.Option(..., "--target"),
    nas_root: Path | None = None,
) -> None:
    s = settings(nas_root)
    rows = []
    for run in RunRepository(s.database_url).list(dataset, target):
        path = run.get("metrics_path")
        if path and Path(path).exists():
            rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    output = s.project_root / "reports" / "certificates" / f"{dataset}-{target}.md"
    print(generate_certificate_report(output, dataset, target, rows))


@report_app.command("all")
def report_all(
    experiment_id: str | None = typer.Option(None, "--experiment-id"),
    nas_root: Path | None = None,
) -> None:
    if experiment_id:
        print(generate_bootstrap_comparisons(settings(nas_root), experiment_id))
    else:
        for dataset_name in load_dataset_configs():
            for target in ["target", "target_real", "in_pocket"]:
                report_certificate(dataset_name, target, nas_root)


@report_app.command("snapshot")
def report_snapshot(run_id: str = typer.Option(..., "--run-id"), nas_root: Path | None = None) -> None:
    s = settings(nas_root)
    for manifest_path in (s.project_root / "runs").glob(f"*/*/*/{run_id}/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            {
                "run_id": run_id,
                "snapshot_diagnostics_path": manifest.get("snapshot_diagnostics_path"),
                "snapshot_diagnostics": manifest.get("snapshot_diagnostics"),
                "explanation_path": manifest.get("explanation_path"),
            }
        )
        return
    raise typer.BadParameter(f"Run not found: {run_id}")


@compare_app.command("bootstrap")
def compare_bootstrap(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    samples: int = typer.Option(1000, "--samples"),
    seed: int = typer.Option(42, "--seed"),
    nas_root: Path | None = None,
) -> None:
    print(generate_bootstrap_comparisons(settings(nas_root), experiment_id, samples=samples, seed=seed))


@compare_app.command("snapshot")
def compare_snapshot(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    samples: int = typer.Option(1000, "--samples"),
    seed: int = typer.Option(42, "--seed"),
    nas_root: Path | None = None,
) -> None:
    print(generate_bootstrap_comparisons(settings(nas_root), experiment_id, samples=samples, seed=seed))


@certificate_app.command("all")
def certificate_all(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    nas_root: Path | None = None,
) -> None:
    print(generate_certificates_for_experiment(settings(nas_root), experiment_id))


@smoke_app.command("gpu")
def smoke_gpu(
    dataset: str = typer.Option("energy", "--dataset"),
    target: str = typer.Option("target", "--target"),
    device: str = typer.Option("cuda:0", "--device"),
    run_config: bool = typer.Option(False, "--run-config"),
    nas_root: Path | None = None,
) -> None:
    import torch

    if not torch.cuda.is_available():
        raise typer.BadParameter(
            f"Requested device {device} but torch.cuda.is_available() is false. "
            "Run this command on the GPU node or set --device cpu for CPU-only checks."
        )
    print({"cuda_available": True, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    from square_sim.models.squaresim.model import SQUARESimModel, config_for_model_name

    model = SQUARESimModel(config_for_model_name("squaresim_full", input_dim=8)).to(device)
    x = torch.randn(4, 8, device=device)
    y = model(x)
    print({"forward_shape": list(y.shape), "dataset": dataset, "target": target})
    if run_config:
        print(
            run_experiment_config(
                Path("configs/experiments/gpu_smoke_energy.yaml"),
                settings(nas_root),
            )
        )


@smoke_app.command("snapshot")
def smoke_snapshot(
    dataset: str = typer.Option("energy", "--dataset"),
    target: str = typer.Option("target", "--target"),
    device: str = typer.Option("cuda:0", "--device"),
    nas_root: Path | None = None,
) -> None:
    import torch

    requested = device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise typer.BadParameter(
            f"Requested device {requested} but torch.cuda.is_available() is false. "
            "Run this command on the GPU node or set --device cpu for CPU-only snapshot checks."
        )
    from square_sim.models.squaresim.model import make_squaresim_model

    model = make_squaresim_model("squaresim_snapshot_rollout", input_dim=8).to(requested)
    x = torch.randn(4, 8, device=requested)
    with torch.no_grad():
        y = model(x)
    print(
        {
            "snapshot_forward_shape": list(y.shape),
            "dataset": dataset,
            "target": target,
            "device": requested,
            "diagnostics": getattr(model, "last_diagnostics", {}),
        }
    )
    try:
        if requested.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


@synthetic_app.command("make-snapshot-diagnostics")
def synthetic_snapshot_diagnostics(
    output: Path = typer.Option(..., "--output"),
    rows: int = typer.Option(10_000, "--rows"),
    seed: int = typer.Option(42, "--seed"),
    nas_root: Path | None = None,
) -> None:
    from square_sim.data.synthetic_snapshot import make_snapshot_diagnostics

    print(make_snapshot_diagnostics(settings(nas_root), output, rows=rows, seed=seed))


@app.command("server")
def server() -> None:
    import uvicorn

    s = settings()
    uvicorn.run("square_sim.orchestration.api:create_app", factory=True, host=s.api_host, port=s.api_port)


@app.command("worker")
def worker(role: str = typer.Option(..., "--role"), queues: str = typer.Option("gpu", "--queues")) -> None:
    run_worker(settings(), role, [q.strip() for q in queues.split(",") if q.strip()])


@app.command("submit")
def submit(config: Path = typer.Option(..., "--config"), queue: str = typer.Option("gpu", "--queue")) -> None:
    job_id = create_job(settings().database_url, queue, {"config": str(config)})
    print({"job_id": job_id, "status": "submitted"})


@jobs_app.command("list")
def jobs_list() -> None:
    print(list_jobs(settings().database_url))


@jobs_app.command("show")
def jobs_show(job_id: str = typer.Option(..., "--job-id")) -> None:
    print(get_job(settings().database_url, job_id))


@jobs_app.command("cancel")
def jobs_cancel(job_id: str = typer.Option(..., "--job-id")) -> None:
    set_job_status(settings().database_url, job_id, "cancelled")
    print({"job_id": job_id, "status": "cancelled"})


if __name__ == "__main__":
    app()
