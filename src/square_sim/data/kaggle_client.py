from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class KaggleCredentialError(RuntimeError):
    pass


def has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or (
        bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    )


def download_dataset(slug: str, output_zip: Path) -> None:
    if not has_kaggle_credentials():
        raise KaggleCredentialError(
            "Kaggle credentials were not found. Create ~/.kaggle/kaggle.json with chmod 600 "
            "or set KAGGLE_USERNAME and KAGGLE_KEY. Credentials are never committed."
        )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle"):
        subprocess.check_call(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                slug,
                "-p",
                str(output_zip.parent),
                "--force",
            ]
        )
        if not output_zip.exists():
            zips = sorted(output_zip.parent.glob("*.zip"))
            if zips:
                zips[-1].rename(output_zip)
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Install the kaggle CLI/API (`uv add kaggle`) or place a ZIP with --offline-zip."
        ) from exc
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(output_zip.parent), unzip=False, quiet=False)
    zips = sorted(output_zip.parent.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"Kaggle download for {slug} did not produce a ZIP.")
    zips[-1].rename(output_zip)
