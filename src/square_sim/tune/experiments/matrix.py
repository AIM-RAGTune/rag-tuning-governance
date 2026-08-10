from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.tune.config import TuneExperimentConfig, experiment_id_for


def plan_tune_matrix(config_path: Path) -> dict[str, Any]:
    cfg = TuneExperimentConfig.from_path(config_path)
    rows = [
        {"dataset_key": dataset, "seed": seed, "optimizer_name": optimizer}
        for dataset in cfg.datasets
        for seed in cfg.seeds
        for optimizer in cfg.optimizers
    ]
    return {
        "experiment_id": experiment_id_for(config_path, cfg),
        "datasets": len(cfg.datasets),
        "seeds": len(cfg.seeds),
        "optimizers": len(cfg.optimizers),
        "planned_runs": len(rows),
        "runs": rows,
    }

