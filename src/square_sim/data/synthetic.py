from __future__ import annotations

from square_sim.config import Settings
from square_sim.paths import LabPaths
from square_sim.utils.files import write_json


def make_synthetic_dataset(settings: Settings, dataset: str = "energy", version: str = "synthetic-v1", rows: int = 240) -> dict:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    x0 = rng.normal(size=rows)
    x1 = rng.normal(size=rows)
    phase = np.sin(3 * x0) + np.cos(2 * x1)
    target = (phase + 0.25 * rng.normal(size=rows) > 0).astype(int)
    df = pd.DataFrame(
        {
            "feature_a": x0,
            "feature_b": x1,
            "feature_c": rng.uniform(-1, 1, size=rows),
            "target": target,
            "target_real": (x0 + x1 > 0).astype(int),
            "in_pocket": ((phase > 0.5) & (x0 > -0.5)).astype(int),
        }
    )
    lab = LabPaths.from_settings(settings)
    processed = lab.processed_dir(dataset, version)
    processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed / "data.parquet", index=False)
    write_json(
        processed / "schema.json",
        {
            "dataset": dataset,
            "version": version,
            "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        },
    )
    write_json(processed / "profile.json", {"rows": rows, "columns": len(df.columns)})
    return {"dataset": dataset, "version": version, "path": str(processed)}

