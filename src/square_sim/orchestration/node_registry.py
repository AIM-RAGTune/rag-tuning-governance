from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.system.gpu import gpu_info
from square_sim.system.hardware import disk_info
from square_sim.system.node_identity import get_node_identity
from square_sim.utils.files import read_json, write_json


def heartbeat(settings: Settings, role: str, scratch: Path | None = None) -> dict[str, Any]:
    identity = get_node_identity(role)
    payload = {
        "node_id": identity.node_id,
        "hostname": identity.hostname,
        "role": identity.role,
        "gpu": gpu_info(),
        "nas": disk_info(settings.project_root),
        "scratch": disk_info(scratch or settings.processing_scratch),
        "status": "ok",
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
    }
    path = settings.project_root / "registry" / "snapshots" / f"node-{identity.node_id}.json"
    write_json(path, payload)
    return payload


def list_nodes(settings: Settings) -> list[dict[str, Any]]:
    root = settings.project_root / "registry" / "snapshots"
    if not root.exists():
        return []
    return [read_json(p) for p in sorted(root.glob("node-*.json"))]

