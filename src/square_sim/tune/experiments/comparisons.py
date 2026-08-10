from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import read_json


def load_experiment_metrics(project_root: Path, experiment_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((project_root / "tune_runs").glob("*/*/*/*/run_manifest.json")):
        manifest = read_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "succeeded":
            continue
        metrics_path = Path(str(manifest.get("metrics_path")))
        if metrics_path.exists():
            row = read_json(metrics_path)
            row.update(
                {
                    "dataset_key": manifest.get("dataset_key"),
                    "seed": manifest.get("seed"),
                    "optimizer_name": manifest.get("model_or_optimizer_name"),
                    "control_type": manifest.get("control_type"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)

