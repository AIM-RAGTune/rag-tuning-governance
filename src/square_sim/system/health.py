from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.paths import LAYOUT_DIRS, LabPaths
from square_sim.system.gpu import gpu_info
from square_sim.system.hardware import disk_info, hardware_snapshot


def path_check(settings: Settings) -> dict[str, Any]:
    lab = LabPaths.from_settings(settings)
    paths = {
        "aim_nas_root": settings.aim_nas_root,
        "project_root": settings.project_root,
        "gpu_hot_scratch": settings.gpu_hot_scratch,
        "gpu_warm_scratch": settings.gpu_warm_scratch,
        "processing_scratch": settings.processing_scratch,
    }
    payload = {
        name: {
            **disk_info(Path(path)),
            "writable": os.access(path, os.W_OK) if Path(path).exists() else False,
        }
        for name, path in paths.items()
    }
    payload["layout_dirs"] = [str(lab.root / rel) for rel in LAYOUT_DIRS]
    return payload


def health(settings: Settings) -> dict[str, Any]:
    return {
        "status": "ok",
        "paths": path_check(settings),
        "hardware": hardware_snapshot([settings.aim_nas_root, settings.project_root]),
        "gpu": gpu_info(),
    }
