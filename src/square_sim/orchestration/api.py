from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.acquire import acquire_dataset
from square_sim.data.catalog import load_dataset_configs, show_catalog
from square_sim.data.normalize import normalize_dataset
from square_sim.data.split import create_split
from square_sim.orchestration.jobs import get_job, list_jobs, set_job_status
from square_sim.orchestration.node_registry import list_nodes
from square_sim.registry.repositories import RunRepository
from square_sim.reporting.certificate import generate_certificate_report
from square_sim.system.health import health
from square_sim.training.train import run_single_model


def create_app(settings: Settings | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for service mode. Run `uv sync`.") from exc
    settings = settings or Settings.from_env()
    app = FastAPI(title="SQUARESim Lab")

    @app.get("/health")
    def api_health() -> dict[str, Any]:
        return health(settings)

    @app.get("/nodes")
    def api_nodes() -> list[dict[str, Any]]:
        return list_nodes(settings)

    @app.get("/datasets")
    def api_datasets() -> dict[str, Any]:
        cfg = load_dataset_configs()
        return {k: v.__dict__ | show_catalog(settings.project_root, k) for k, v in cfg.items()}

    @app.get("/runs")
    def api_runs() -> list[dict[str, Any]]:
        return RunRepository(settings.database_url).list()

    @app.get("/runs/{run_id}")
    def api_run(run_id: str) -> dict[str, Any] | None:
        return RunRepository(settings.database_url).get(run_id)

    @app.post("/data/acquire")
    def api_acquire(payload: dict[str, Any]) -> dict[str, Any]:
        return acquire_dataset(payload["dataset"], settings)

    @app.post("/data/prepare")
    def api_prepare(payload: dict[str, Any]) -> dict[str, Any]:
        result = normalize_dataset(payload["dataset"], settings)
        split = create_split(payload["dataset"], settings, target=payload.get("target", "target"))
        return {"normalize": result, "split": split}

    @app.post("/runs")
    def api_run_submit(payload: dict[str, Any]) -> dict[str, Any]:
        return run_single_model(
            settings,
            dataset=payload["dataset"],
            target=payload["target"],
            model_name=payload["model"],
            split_id=payload.get("split_id", "default"),
            seed=int(payload.get("seed", 42)),
            device=payload.get("device", "cpu"),
        )

    @app.post("/reports/certificate")
    def api_certificate(payload: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for run in RunRepository(settings.database_url).list(payload["dataset"], payload["target"]):
            metrics_path = run.get("metrics_path")
            if metrics_path and Path(metrics_path).exists():
                import json

                rows.append(json.loads(Path(metrics_path).read_text(encoding="utf-8")))
        output = settings.project_root / "reports" / "certificates" / f"{payload['dataset']}-{payload['target']}.md"
        return generate_certificate_report(output, payload["dataset"], payload["target"], rows)

    @app.get("/jobs")
    def api_jobs() -> list[dict[str, Any]]:
        return list_jobs(settings.database_url)

    @app.get("/jobs/{job_id}")
    def api_job(job_id: str) -> dict[str, Any] | None:
        return get_job(settings.database_url, job_id)

    @app.post("/jobs/{job_id}/cancel")
    def api_cancel(job_id: str) -> dict[str, str]:
        set_job_status(settings.database_url, job_id, "cancelled")
        return {"job_id": job_id, "status": "cancelled"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        runs = RunRepository(settings.database_url).list()[:20]
        rows = "".join(
            f"<tr><td>{r['run_id']}</td><td>{r['dataset']}</td><td>{r['target']}</td>"
            f"<td>{r['model']}</td><td>{r['status']}</td></tr>"
            for r in runs
        )
        return f"""
        <html><head><title>SQUARESim Lab</title>
        <style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}
        td,th{{border-bottom:1px solid #ddd;padding:.5rem;text-align:left}}</style></head>
        <body><h1>SQUARESim Lab</h1><p>Dataset status, latest runs, and worker health.</p>
        <h2>Runs</h2><table><tr><th>Run</th><th>Dataset</th><th>Target</th><th>Model</th><th>Status</th></tr>{rows}</table>
        <h2>Workers</h2><pre>{list_nodes(settings)}</pre></body></html>
        """

    return app

