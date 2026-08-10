from __future__ import annotations

from pathlib import Path

from square_sim.config import Settings
from square_sim.data.acquire import acquire_dataset


def import_offline_zip(dataset_name: str, zip_path: Path, settings: Settings) -> dict:
    return acquire_dataset(dataset_name, settings, offline_zip=zip_path)

