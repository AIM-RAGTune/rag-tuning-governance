from __future__ import annotations

from pathlib import Path

from square_sim.config import Settings


def external_root(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "external" / "square_tune_v1"


def external_reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "square_tune" / "external_transfer"


def external_certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "square_tune" / "external_transfer"


def external_runs_root(settings: Settings) -> Path:
    return settings.project_root / "tune_external_runs"
