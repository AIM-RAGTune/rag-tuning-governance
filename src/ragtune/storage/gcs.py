from __future__ import annotations

from pathlib import Path

from ragtune.storage.base import ArtifactSink, StorageUnavailable, StoredArtifact


class GCSArtifactSink(ArtifactSink):
    mode = "gcs"

    def __init__(self, *, bucket: str | None, prefix: str = ""):
        if not bucket:
            raise StorageUnavailable("RAGTUNE_GCS_BUCKET is required")
        try:
            import google.cloud.storage  # noqa: F401
        except Exception as exc:
            raise StorageUnavailable("google-cloud-storage optional dependency is not installed") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def put_file(self, local_path: Path, artifact_name: str) -> StoredArtifact:
        raise StorageUnavailable("GCS upload is disabled in the publication test harness")
