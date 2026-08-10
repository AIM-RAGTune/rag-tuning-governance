from __future__ import annotations

import shutil
import subprocess
from typing import Any


def torch_gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False, "gpus": []}

    info: dict[str, Any] = {
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def nvidia_smi_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total": parts[2],
                    "driver_version": parts[3],
                }
            )
    return {"available": True, "gpus": gpus}


def gpu_info() -> dict[str, Any]:
    return {"torch": torch_gpu_info(), "nvidia_smi": nvidia_smi_info()}

