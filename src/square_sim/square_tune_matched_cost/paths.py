from __future__ import annotations

from pathlib import Path

from square_sim.config import Settings


def dataset_root(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "square_tune" / "matched_cost_rag" / "v1"


def scenario_root(settings: Settings) -> Path:
    return settings.project_root / "scenarios" / "square_tune" / "matched_cost_rag" / "v1"


def reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "square_tune" / "matched_cost_rag" / "v1"


def artifacts_root(settings: Settings) -> Path:
    return settings.project_root / "artifacts" / "square_tune" / "matched_cost_rag" / "v1"


def certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "square_tune" / "matched_cost_rag" / "v1"


def publication_root(settings: Settings) -> Path:
    return settings.project_root / "publication" / "square_tune_matched_cost_rag" / "v1"

