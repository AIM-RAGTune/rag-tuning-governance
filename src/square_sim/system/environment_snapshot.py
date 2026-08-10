from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "torch",
    "xgboost",
    "lightgbm",
    "duckdb",
    "sqlalchemy",
]


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_snapshot(cwd: Path | None = None) -> dict[str, Any]:
    hf_vars = {k: v for k, v in os.environ.items() if k in {"HF_HOME", "TRANSFORMERS_CACHE"}}
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": git_commit(cwd),
        "packages": package_versions(),
        "hf_cache_env": hf_vars,
    }


def write_environment_snapshot(path: Path, cwd: Path | None = None) -> dict[str, Any]:
    payload = environment_snapshot(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
