from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from square_sim.config import Settings

LAYOUT_DIRS = [
    "datasets/raw/kaggle",
    "datasets/staging",
    "datasets/processed",
    "datasets/splits",
    "runs",
    "reports/dataset_reports",
    "reports/comparison_reports",
    "reports/certificates",
    "artifacts/compact",
    "artifacts/exported_tables",
    "artifacts/figures",
    "logs/orchestrator",
    "logs/workers",
    "logs/audit",
    "workflows/submitted_jobs",
    "workflows/completed_jobs",
    "workflows/failed_jobs",
    "registry/exports",
    "registry/snapshots",
]


@dataclass(frozen=True)
class LabPaths:
    root: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> LabPaths:
        return cls(settings.project_root)

    def ensure_layout(self) -> None:
        for rel in LAYOUT_DIRS:
            (self.root / rel).mkdir(parents=True, exist_ok=True)

    def raw_dataset_dir(self, dataset_name: str, version: str) -> Path:
        return self.root / "datasets" / "raw" / "kaggle" / dataset_name / version

    def staging_dir(self, dataset_name: str, version: str) -> Path:
        return self.root / "datasets" / "staging" / dataset_name / version

    def processed_dir(self, dataset_name: str, version: str) -> Path:
        return self.root / "datasets" / "processed" / dataset_name / version

    def split_dir(self, dataset_name: str, version: str, split_id: str) -> Path:
        return self.root / "datasets" / "splits" / dataset_name / version / split_id

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id[0:4] / run_id[4:6] / run_id[6:8] / run_id

    def certificate_dir(self) -> Path:
        return self.root / "reports" / "certificates"

