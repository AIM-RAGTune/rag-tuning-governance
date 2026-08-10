from __future__ import annotations

from pathlib import Path

from square_sim.config import Settings
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.write_once import WriteOncePathManager


def path_manager(settings: Settings, root: Path) -> WriteOncePathManager:
    return WriteOncePathManager(root, ProtectedResultsRegistry(settings).protected_paths())
