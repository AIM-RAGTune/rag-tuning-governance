from __future__ import annotations

import os
import platform
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    hostname: str
    role: str


def get_node_identity(role: str | None = None) -> NodeIdentity:
    hostname = platform.node() or "unknown-host"
    resolved_role = role or os.getenv("SQUARESIM_NODE_ROLE", "local")
    node_id = os.getenv("SQUARESIM_NODE_ID", f"{hostname}-{uuid.getnode():x}")
    return NodeIdentity(node_id=node_id, hostname=hostname, role=resolved_role)

