from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


def _default_nas_root() -> str:
    return "<approved-data-root>" if platform.system() == "Darwin" else "<approved-data-root>"


@dataclass(frozen=True)
class Settings:
    aim_nas_root: Path
    project_root: Path
    gpu_hot_scratch: Path
    gpu_warm_scratch: Path
    processing_scratch: Path
    database_url: str
    redis_url: str
    api_host: str
    api_port: int
    local_llm_endpoint: str | None

    @classmethod
    def from_env(cls, nas_root: str | Path | None = None) -> Settings:
        root = Path(nas_root or os.getenv("AIM_NAS_ROOT", _default_nas_root())).expanduser()
        project = Path(
            os.getenv(
                "SQUARESIM_DATA_ROOT",
                os.getenv("AIM_PROJECT_ROOT", str(root / "SQUARE" / "source-validation-workspace")),
            )
        ).expanduser()
        return cls(
            aim_nas_root=root,
            project_root=project,
            gpu_hot_scratch=Path(
                os.getenv("SQUARESIM_GPU_HOT_SCRATCH", "/srv/aim/squaresim/hot")
            ),
            gpu_warm_scratch=Path(
                os.getenv("SQUARESIM_GPU_WARM_SCRATCH", "/srv/aim/squaresim/warm")
            ),
            processing_scratch=Path(
                os.getenv("SQUARESIM_PROCESSING_SCRATCH", "/srv/aim/squaresim/processing")
            ),
            database_url=os.getenv(
                "SQUARESIM_DATABASE_URL", "sqlite:///./.local/squaresim_registry.sqlite3"
            ),
            redis_url=os.getenv("SQUARESIM_REDIS_URL", "redis://localhost:6379/0"),
            api_host=os.getenv("SQUARESIM_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("SQUARESIM_API_PORT", "8087")),
            local_llm_endpoint=os.getenv("LOCAL_LLM_ENDPOINT"),
        )


def require_package(package: str, purpose: str) -> None:
    try:
        __import__(package)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing optional dependency '{package}' required for {purpose}. "
            "Run `uv sync --extra dev` for CPU workflows or `uv sync --extra gpu --extra dev` "
            "on the GPU node."
        ) from exc
