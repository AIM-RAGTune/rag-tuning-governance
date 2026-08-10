from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.catalog import load_dataset_configs
from square_sim.data.resolver import default_split_id, resolve_dataset_version
from square_sim.paths import LabPaths
from square_sim.utils.files import write_json, write_text


@dataclass(frozen=True)
class EnsureSplitRequest:
    dataset_key: str
    target: str
    seed: int
    split_method: str = "stratified"
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _find_square_ingest() -> str | None:
    direct = shutil.which("square-ingest")
    if direct:
        return direct
    sibling = Path.cwd().parent / "squaresim-dataset-ingestor" / ".venv312" / "bin" / "square-ingest"
    if sibling.exists():
        return str(sibling)
    return None


def _valid_split(split_dir: Path, target: str) -> bool:
    if not all((split_dir / name).exists() for name in ["train.parquet", "val.parquet", "test.parquet"]):
        return False
    manifest = split_dir / "split_manifest.json"
    if not manifest.exists():
        return False
    try:
        import json

        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return not payload.get("target") or payload.get("target") == target


def ensure_splits(
    settings: Settings,
    requests: list[EnsureSplitRequest],
    *,
    create: bool = False,
    dry_run: bool = False,
    require_existing: bool = False,
) -> dict[str, Any]:
    lab = LabPaths.from_settings(settings)
    report_dir = settings.project_root / "reports" / "preflight" / _timestamp()
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    square_ingest = _find_square_ingest()

    for request in requests:
        version = resolve_dataset_version(settings, request.dataset_key)
        split_id = default_split_id(request.target, request.seed)
        split_dir = lab.split_dir(request.dataset_key, version.dataset_version_id, split_id)
        command = [
            "square-ingest",
            "split",
            "--dataset",
            request.dataset_key,
            "--version-id",
            version.dataset_version_id,
            "--target",
            request.target,
            "--seed",
            str(request.seed),
            "--method",
            request.split_method,
        ]
        row: dict[str, Any] = {
            "dataset": request.dataset_key,
            "target": request.target,
            "dataset_version_id": version.dataset_version_id,
            "split_id": split_id,
            "split_dir": str(split_dir),
            "command": " ".join(command),
            "status": "missing",
            "log_path": None,
        }
        if _valid_split(split_dir, request.target):
            row["status"] = "exists"
        elif dry_run:
            row["status"] = "would_create" if create else "missing"
        elif create and square_ingest:
            actual_command = [square_ingest, *command[1:]]
            log_path = report_dir / f"{request.dataset_key}_{request.target}_{split_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(actual_command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
            row["log_path"] = str(log_path)
            row["status"] = "created" if proc.returncode == 0 and _valid_split(split_dir, request.target) else "failed"
            row["returncode"] = proc.returncode
        elif create:
            row["status"] = "missing"
            row["message"] = "square-ingest was not found; run the suggested command manually."
        rows.append(row)

    failed = [r for r in rows if r["status"] in {"failed"}]
    missing = [r for r in rows if r["status"] == "missing"]
    if (require_existing and missing) or failed:
        status = "failed"
    else:
        status = "ok"
    payload = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "create": create,
        "dry_run": dry_run,
        "require_existing": require_existing,
        "rows": rows,
        "report_dir": str(report_dir),
    }
    write_json(report_dir / "ensure_splits_report.json", payload)
    lines = ["# Ensure Splits Report", "", f"Status: {status}", ""]
    for row in rows:
        lines.append(
            f"- {row['dataset']} / {row['target']} / {row['dataset_version_id']} / "
            f"{row['split_id']}: {row['status']}"
        )
        if row["status"] in {"missing", "would_create", "failed"}:
            lines.append(f"  Command: `{row['command']}`")
    write_text(report_dir / "ensure_splits_report.md", "\n".join(lines) + "\n")
    return payload


def build_ensure_requests(
    *,
    datasets: list[str] | None,
    targets: list[str],
    seed: int,
    split_method: str = "stratified",
) -> list[EnsureSplitRequest]:
    dataset_names = datasets or list(load_dataset_configs())
    return [
        EnsureSplitRequest(dataset_key=dataset, target=target, seed=seed, split_method=split_method)
        for dataset in dataset_names
        for target in targets
    ]
