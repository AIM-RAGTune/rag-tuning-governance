from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationRunSpec:
    track: str
    task_name: str
    system_name: str
    seed: int
    config_hash: str
    budget_config: dict[str, Any]
    output_root: Path
    device: str = "cpu"
    precision: str = "float32"
    notes: str = ""


@dataclass
class SimulationRunResult:
    run_id: str
    experiment_id: str
    status: str
    metrics: dict[str, float | int | str | bool]
    diagnostics_paths: dict[str, str] = field(default_factory=dict)
    plots_paths: dict[str, str] = field(default_factory=dict)
    manifest_path: str | None = None
    certificate_inputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CoreValidationExperiment:
    experiment_id: str
    matrix_name: str
    tracks: list[str]
    tasks: list[str]
    systems: list[str]
    seeds: list[int]
    created_at_utc: str
    output_root: Path
    no_overwrite_status: str
    calibration_references: dict[str, Any]
    git_commit: str | None = None


@dataclass
class CoreCertificate:
    track: str
    task: str
    status: str
    supported_components: list[str]
    refused_components: list[str]
    evidence: dict[str, Any]
    ablation_results: dict[str, Any]
    control_results: dict[str, Any]
    caveats: list[str]
