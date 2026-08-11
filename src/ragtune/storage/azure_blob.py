from __future__ import annotations

from pathlib import Path

from ragtune.storage.base import ArtifactSink, StorageUnavailable, StoredArtifact


class AzureBlobArtifactSink(ArtifactSink):
    mode = "azure_blob"

    def __init__(self, *, container: str | None, prefix: str = ""):
        if not container:
            raise StorageUnavailable("RAGTUNE_AZURE_BLOB_CONTAINER is required")
        try:
            import azure.storage.blob  # noqa: F401
        except Exception as exc:
            raise StorageUnavailable("azure-storage-blob optional dependency is not installed") from exc
        self.container = container
        self.prefix = prefix.strip("/")

    def put_file(self, local_path: Path, artifact_name: str) -> StoredArtifact:
        raise StorageUnavailable("Azure upload is disabled in the publication test harness")
