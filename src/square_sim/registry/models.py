from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunRecord:
    run_id: str
    dataset: str
    dataset_version: str
    split_id: str
    target: str
    model: str
    seed: int
    config_hash: str
    status: str
    run_path: str
    metrics_path: str | None = None
    predictions_path: str | None = None
    explanation_path: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

