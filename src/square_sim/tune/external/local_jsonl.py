from __future__ import annotations

from pathlib import Path


def validate_local_jsonl(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"status": "failed", "reason": f"Missing local JSONL: {path}"}
    return {"status": "ok", "path": str(path)}

