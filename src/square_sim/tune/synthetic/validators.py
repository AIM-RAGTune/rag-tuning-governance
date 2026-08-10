from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.synthetic.schemas import LATENT_COLUMNS, REQUIRED_COLUMNS
from square_sim.utils.files import read_json


def validate_dataset_dir(path: Path) -> dict[str, Any]:
    data_path = path / "data.parquet"
    manifest_path = path / "generator_manifest.json"
    card_path = path / "mechanism_card.md"
    expected_path = path / "expected_outcomes.json"
    missing = [str(p.name) for p in [data_path, manifest_path, card_path, expected_path] if not p.exists()]
    if missing:
        return {"path": str(path), "status": "failed", "errors": [f"Missing files: {', '.join(missing)}"]}
    df = pd.read_parquet(data_path)
    errors: list[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    leaked_latents = [c for c in LATENT_COLUMNS if c in df.columns]
    if leaked_latents:
        errors.append(f"Latent columns exposed by default: {leaked_latents}")
    manifest = read_json(manifest_path)
    if manifest.get("row_count") != len(df):
        errors.append("Manifest row_count does not match data.parquet")
    return {
        "path": str(path),
        "dataset_key": manifest.get("generator_name"),
        "seed": manifest.get("seed"),
        "row_count": len(df),
        "status": "failed" if errors else "ok",
        "errors": errors,
    }


def validate_suite(root: Path) -> dict[str, Any]:
    results = [validate_dataset_dir(path) for path in sorted(root.glob("*/seed_*"))]
    return {
        "root": str(root),
        "dataset_count": len(results),
        "failed": sum(1 for row in results if row["status"] != "ok"),
        "results": results,
    }

