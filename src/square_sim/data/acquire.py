from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from square_sim.config import Settings
from square_sim.data.catalog import load_dataset_configs
from square_sim.data.kaggle_client import download_dataset
from square_sim.data.manifest import write_source_manifest
from square_sim.paths import LabPaths
from square_sim.utils.files import write_json
from square_sim.utils.hashing import sha256_file


def dataset_storage_name(dataset_name: str) -> str:
    return {
        "energy": "spectra-energy",
        "oilgas": "spectra-oilgas",
        "maintenance": "spectra-maintenance",
        "telecom": "spectra-telecom",
    }[dataset_name]


def acquire_dataset(
    dataset_name: str,
    settings: Settings,
    force: bool = False,
    offline_zip: Path | None = None,
    no_extract: bool = False,
    verify_only: bool = False,
) -> dict:
    datasets = load_dataset_configs()
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {sorted(datasets)}.")
    cfg = datasets[dataset_name]
    lab = LabPaths.from_settings(settings)
    lab.ensure_layout()
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = lab.raw_dataset_dir(dataset_storage_name(dataset_name), version)
    staging_dir = lab.staging_dir(dataset_name, version)
    zip_path = raw_dir / "original.zip"

    if verify_only:
        if not zip_path.exists():
            raise FileNotFoundError(f"No ZIP exists at {zip_path}; cannot verify.")
        return {"dataset": dataset_name, "zip": str(zip_path), "sha256": sha256_file(zip_path)}

    if raw_dir.exists() and not force:
        raise FileExistsError(f"Raw dataset directory already exists: {raw_dir}. Use --force.")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if offline_zip:
        if not offline_zip.exists():
            raise FileNotFoundError(f"Offline ZIP not found: {offline_zip}")
        shutil.copy2(offline_zip, zip_path)
    else:
        download_dataset(cfg.kaggle_slug, zip_path)

    metadata = {
        "slug": cfg.kaggle_slug,
        "source_url": f"https://www.kaggle.com/datasets/{cfg.kaggle_slug}",
        "license": "See Kaggle dataset metadata/page.",
        "zip_sha256": sha256_file(zip_path),
    }
    write_json(raw_dir / "kaggle_metadata.json", metadata)

    if not no_extract:
        staging_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging_dir)
        write_source_manifest(
            staging_dir / "manifest.json",
            dataset=dataset_name,
            slug=cfg.kaggle_slug,
            source_url=metadata["source_url"],
            local_path=staging_dir,
            version=version,
            extra={"raw_zip": str(zip_path), "raw_zip_sha256": metadata["zip_sha256"]},
        )

    manifest = write_source_manifest(
        raw_dir / "manifest.json",
        dataset=dataset_name,
        slug=cfg.kaggle_slug,
        source_url=metadata["source_url"],
        local_path=raw_dir,
        version=version,
        extra={"staging_path": str(staging_dir) if staging_dir.exists() else None},
    )
    return manifest
