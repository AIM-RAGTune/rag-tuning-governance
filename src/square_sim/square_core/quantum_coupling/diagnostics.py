from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.utils.files import write_json


def write_quantum_summary(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return str(path)
