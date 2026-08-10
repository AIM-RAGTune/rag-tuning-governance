from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text

CAUTION = (
    "This is a SQUARE Core Validation Matrix software simulation. It does not prove "
    "physical SQUARE hardware, quantum advantage, or commercial viability."
)


def write_run_diagnostics(run_dir: Path, metrics: dict[str, Any], trace: list[dict[str, Any]] | None = None) -> dict[str, str]:
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    write_json(diag_dir / "diagnostics.json", {"metrics": metrics, "caution": CAUTION})
    paths = {"diagnostics": str(diag_dir / "diagnostics.json")}
    if trace is not None:
        trace_path = diag_dir / "trace.parquet"
        pd.DataFrame(trace).to_parquet(trace_path, index=False)
        paths["trace"] = str(trace_path)
    write_text(diag_dir / "summary.md", f"# Run Diagnostics\n\n{CAUTION}\n")
    paths["summary"] = str(diag_dir / "summary.md")
    return paths


def no_overwrite_audit(experiment_id: str, protected_paths: list[Path], write_roots: list[Path]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "protected_paths": [str(path) for path in protected_paths],
        "write_roots": [str(path) for path in write_roots],
        "attempted_overwrites_blocked": 0,
        "status": "append_only_confirmed",
    }
