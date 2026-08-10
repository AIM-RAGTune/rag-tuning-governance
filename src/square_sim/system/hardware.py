from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any


def disk_info(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": str(path),
            "exists": path.exists(),
            "free_gb": round(usage.free / 1024**3, 2),
            "total_gb": round(usage.total / 1024**3, 2),
        }
    except FileNotFoundError:
        return {"path": str(path), "exists": False}


def hardware_snapshot(paths: list[Path] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        payload["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        payload["ram_gb"] = None
    if paths:
        payload["disks"] = [disk_info(p) for p in paths]
    return payload

