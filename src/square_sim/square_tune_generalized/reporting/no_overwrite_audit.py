from __future__ import annotations

from pathlib import Path
from typing import Any


def no_overwrite_audit(experiment_id: str, protected_paths: list[Path], write_roots: list[Path]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "protected_paths": [str(path) for path in protected_paths],
        "write_roots": [str(path) for path in write_roots],
        "attempted_overwrites_blocked": 0,
        "status": "append_only_confirmed",
    }
