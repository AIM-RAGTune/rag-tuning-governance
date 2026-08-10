from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def run_peft_smoke(config_path: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    if not config_path.exists():
        return {"status": "skipped", "reason": f"Missing config: {config_path}"}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not cfg.get("enable_real_peft", False):
        return {"status": "skipped", "reason": "enable_real_peft is false. No model download or adapter training attempted."}
    if not cfg.get("license_acknowledged", False) or not cfg.get("model_id_or_path"):
        return {"status": "blocked", "reason": "Set model_id_or_path and license_acknowledged before PEFT smoke."}
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return {"status": "skipped", "reason": f"Missing optional PEFT dependencies: {exc}"}
    return {
        "status": "not_run",
        "reason": "PEFT dependencies are present, but this scaffold does not start model training without a concrete local-lab config.",
        "device": device,
    }

