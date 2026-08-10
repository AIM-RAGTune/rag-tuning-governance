from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print

from square_sim.config import Settings
from square_sim.tune.config import parse_csv_ints
from square_sim.tune.experiments.comparisons import load_experiment_metrics
from square_sim.tune.experiments.runner import generate_reports, run_tune_matrix
from square_sim.tune.reporting.calibration import (
    summarize_prior_experiment,
    write_calibration_reports,
)
from square_sim.tune.reporting.certificate import write_certificates
from square_sim.tune.synthetic.generators import generate_suite

app = typer.Typer(help="SQUARETune Calibration Pass v2")

CALIBRATION_DATASETS = [
    "synthetic_llm_random_label",
    "synthetic_llm_linear_control",
    "synthetic_llm_merge_required",
    "synthetic_llm_repeated_regression_memory",
    "synthetic_llm_regression_veto",
    "synthetic_llm_cost_tradeoff",
    "synthetic_llm_data_poison_regression",
    "synthetic_llm_prompt_regression",
    "synthetic_llm_nonmonotonic_data_mix",
    "synthetic_llm_failure_cluster_routing",
    "synthetic_llm_adapter_tradeoff",
    "synthetic_llm_rag_policy_conflict",
]


def _compact(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in {"results"}}


def _settings(nas_root: Path | None = None) -> Settings:
    return Settings.from_env(nas_root)


def _timestamped_config(settings: Settings, source: Path, prefix: str) -> Path:
    import yaml

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cfg["experiment_name"] = f"{cfg.get('experiment_name', source.stem)}_{ts}"
    cfg["dataset_root"] = str(settings.project_root / "datasets" / "synthetic" / "square_tune_calibration_v2")
    out = settings.project_root / "protocols" / "square_tune" / "active_runs" / f"{prefix}_{ts}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


@app.command("summarize-prior")
def summarize_prior(
    experiment_path: Path = typer.Option(..., "--experiment-path"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    output = s.project_root / "reports" / "square_tune" / "calibration"
    print(summarize_prior_experiment(experiment_path, output))


@app.command("prepare-v2")
def prepare_v2(
    rows: int = typer.Option(50_000, "--rows"),
    seeds: str = typer.Option("101,202,303,404,505,606,707,808,909,1001", "--seeds"),
    output: Path = typer.Option(..., "--output"),
    noise_level: float = typer.Option(0.05, "--noise-level"),
) -> None:
    print(
        _compact(
            generate_suite(
                output,
                suite="llm_tuning_v1",
                rows=rows,
                seeds=parse_csv_ints(seeds),
                noise_level=noise_level,
                datasets=CALIBRATION_DATASETS,
            )
        )
    )


@app.command("run-controls")
def run_controls(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    timestamped: bool = typer.Option(True, "--timestamped/--no-timestamped"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    cfg_path = _timestamped_config(s, config, "square_tune_calibration_v2_controls") if timestamped else config
    print(_compact(run_tune_matrix(s, cfg_path, device=device, resume=resume, skip_completed=skip_completed)))


@app.command("run-v2")
def run_v2(
    config: Path = typer.Option(..., "--config"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    device: str = typer.Option("cpu", "--device"),
    timestamped: bool = typer.Option(True, "--timestamped/--no-timestamped"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    cfg_path = _timestamped_config(s, config, "square_tune_calibration_v2_matrix") if timestamped else config
    print(_compact(run_tune_matrix(s, cfg_path, device=device, resume=resume, skip_completed=skip_completed)))


@app.command("report")
def calibration_report(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    s = _settings(nas_root)
    generate_reports(s, experiment_id)
    metrics = load_experiment_metrics(s.project_root, experiment_id)
    print(write_calibration_reports(s.project_root / "reports" / "square_tune" / "calibration" / experiment_id, experiment_id, metrics))


@app.command("certificate")
def calibration_certificate(experiment_id: str = typer.Option(..., "--experiment-id"), nas_root: Path | None = None) -> None:
    s = _settings(nas_root)
    metrics = load_experiment_metrics(s.project_root, experiment_id)
    output = s.project_root / "certificates" / "square_tune" / "calibration" / experiment_id
    payload = write_certificates(output, experiment_id, metrics)
    print(
        {
            "experiment_id": payload["experiment_id"],
            "certificate_count": payload["certificate_count"],
            "global_status": payload["calibration_gates"].get("global_status"),
            "failed_gates": payload["calibration_gates"].get("failed_gates", []),
            "output_dir": str(output),
        }
    )


@app.command("run-all-v2")
def run_all_v2(
    prepare_data: bool = typer.Option(False, "--prepare-data"),
    run_controls_first: bool = typer.Option(False, "--run-controls-first"),
    stop_if_gates_fail: bool = typer.Option(False, "--stop-if-gates-fail"),
    run_full_if_controls_pass: bool = typer.Option(False, "--run-full-if-controls-pass"),
    generate_reports_flag: bool = typer.Option(False, "--generate-reports"),
    generate_certificates: bool = typer.Option(False, "--generate-certificates"),
    rows: int = typer.Option(50_000, "--rows"),
    seeds: str = typer.Option("101,202,303,404,505,606,707,808,909,1001", "--seeds"),
    nas_root: Path | None = None,
) -> None:
    s = _settings(nas_root)
    output = s.project_root / "datasets" / "synthetic" / "square_tune_calibration_v2"
    if prepare_data:
        generate_suite(
            output,
            suite="llm_tuning_v1",
            rows=rows,
            seeds=parse_csv_ints(seeds),
            datasets=CALIBRATION_DATASETS,
        )
    controls_summary = None
    if run_controls_first:
        controls_cfg = _timestamped_config(
            s,
            Path("configs/tune/square_tune_calibration_v2_controls_only.yaml"),
            "square_tune_calibration_v2_controls",
        )
        controls_summary = run_tune_matrix(s, controls_cfg, resume=True, skip_completed=True)
        controls_exp = str(controls_summary["experiment_id"])
        metrics = load_experiment_metrics(s.project_root, controls_exp)
        gate_report = write_calibration_reports(
            s.project_root / "reports" / "square_tune" / "calibration" / controls_exp,
            controls_exp,
            metrics,
        )
        if generate_reports_flag:
            generate_reports(s, controls_exp)
        if generate_certificates:
            write_certificates(
                s.project_root / "certificates" / "square_tune" / "calibration" / controls_exp,
                controls_exp,
                metrics,
            )
        if stop_if_gates_fail and gate_report["overall_status"] != "Calibration passed":
            print(
                {
                    "status": "stopped_after_controls",
                    "controls_summary": _compact(controls_summary),
                    "gate_report": gate_report,
                }
            )
            return
    full_summary = None
    if run_full_if_controls_pass:
        full_cfg = _timestamped_config(
            s,
            Path("configs/tune/square_tune_calibration_v2_matrix.yaml"),
            "square_tune_calibration_v2_matrix",
        )
        full_summary = run_tune_matrix(s, full_cfg, resume=True, skip_completed=True)
        full_exp = str(full_summary["experiment_id"])
        metrics = load_experiment_metrics(s.project_root, full_exp)
        if generate_reports_flag:
            generate_reports(s, full_exp)
            write_calibration_reports(
                s.project_root / "reports" / "square_tune" / "calibration" / full_exp,
                full_exp,
                metrics,
            )
        if generate_certificates:
            write_certificates(
                s.project_root / "certificates" / "square_tune" / "calibration" / full_exp,
                full_exp,
                metrics,
            )
    print(
        {
            "status": "completed",
            "controls_summary": _compact(controls_summary) if controls_summary else None,
            "full_summary": _compact(full_summary) if full_summary else None,
        }
    )
