from __future__ import annotations

from pathlib import Path

from square_sim.config import Settings


def reports_root(settings: Settings) -> Path:
    return settings.project_root / "reports" / "square_next_sim" / "v1"


def artifacts_root(settings: Settings) -> Path:
    return settings.project_root / "artifacts" / "square_next_sim" / "v1"


def certificates_root(settings: Settings) -> Path:
    return settings.project_root / "certificates" / "square_next_sim" / "v1"


def logs_root(settings: Settings) -> Path:
    return settings.project_root / "logs" / "square_next_sim" / "v1"


def publication_root(settings: Settings) -> Path:
    return settings.project_root / "publication" / "square_next_sim" / "v1"

