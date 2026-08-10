from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.square_core.config import CoreConfig
from square_sim.utils.hashing import sha256_file


def plan_matrix(config_path: Path) -> dict[str, Any]:
    cfg = CoreConfig.from_path(config_path)
    planned = cfg.planned_runs()
    by_track = {track: len([row for row in planned if row["track"] == track]) for track in cfg.tracks}
    return {
        "matrix_name": cfg.matrix_name,
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "tracks": cfg.tracks,
        "seeds": cfg.seeds,
        "total_planned": len(planned),
        "planned_by_track": by_track,
        "runs": planned,
    }
