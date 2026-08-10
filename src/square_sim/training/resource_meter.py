from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResourceMeter:
    device: str
    start_time: float = field(default_factory=time.perf_counter)
    peak_gpu_memory_mb: float | None = None

    def start(self) -> None:
        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def stop(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.start_time
        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        except Exception:
            pass
        return {"train_seconds": elapsed, "peak_gpu_memory_mb": self.peak_gpu_memory_mb}


def parameter_count(model: Any) -> int | None:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return None

