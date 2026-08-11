from __future__ import annotations

import os
from pathlib import Path

from ragtune.storage.azure_blob import AzureBlobArtifactSink
from ragtune.storage.base import ArtifactSink, StorageUnavailable
from ragtune.storage.gcs import GCSArtifactSink
from ragtune.storage.local import LocalArtifactSink
from ragtune.storage.s3 import S3ArtifactSink


class DisabledArtifactSink(ArtifactSink):
    mode = "disabled"

    def put_file(self, local_path: Path, artifact_name: str):
        raise StorageUnavailable("artifact storage is disabled")

    def healthcheck(self) -> dict[str, object]:
        return {"mode": self.mode, "available": False, "secrets_logged": False}


def build_storage_sink(mode: str | None = None, output_root: str | Path | None = None) -> ArtifactSink:
    selected = (mode or os.environ.get("RAGTUNE_STORAGE_MODE") or "local").strip().lower()
    root = Path(output_root or os.environ.get("RAGTUNE_OUTPUT_ROOT") or "artifacts/storage")
    if selected == "local":
        return LocalArtifactSink(root)
    if selected == "disabled":
        return DisabledArtifactSink()
    if selected == "azure_blob":
        return AzureBlobArtifactSink(container=os.environ.get("RAGTUNE_AZURE_BLOB_CONTAINER"), prefix=os.environ.get("RAGTUNE_AZURE_BLOB_PREFIX", ""))
    if selected == "s3":
        return S3ArtifactSink(bucket=os.environ.get("RAGTUNE_S3_BUCKET"), prefix=os.environ.get("RAGTUNE_S3_PREFIX", ""))
    if selected == "gcs":
        return GCSArtifactSink(bucket=os.environ.get("RAGTUNE_GCS_BUCKET"), prefix=os.environ.get("RAGTUNE_GCS_PREFIX", ""))
    raise StorageUnavailable(f"unknown storage mode: {selected}")
