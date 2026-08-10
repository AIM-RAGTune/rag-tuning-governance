from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.tune.external.acquire import refresh_catalog
from square_sim.utils.files import read_json


def load_external_catalog(output_root: Path) -> dict[str, Any]:
    path = output_root / "catalog" / "external_dataset_catalog.json"
    if not path.exists():
        return {"datasets": []}
    return read_json(path)


def list_external_catalog(output_root: Path) -> list[dict[str, Any]]:
    return list(load_external_catalog(output_root).get("datasets", []))


def show_external_dataset(output_root: Path, dataset_key: str) -> dict[str, Any]:
    rows = [
        row
        for row in list_external_catalog(output_root)
        if row.get("dataset_key") == dataset_key
    ]
    if not rows:
        return {"status": "missing", "dataset_key": dataset_key}
    rows.sort(key=lambda row: str(row.get("dataset_version_id", "")))
    return {"status": "found", "dataset_key": dataset_key, "versions": rows}


__all__ = [
    "list_external_catalog",
    "load_external_catalog",
    "refresh_catalog",
    "show_external_dataset",
]
