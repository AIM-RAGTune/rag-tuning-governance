from __future__ import annotations

from pathlib import Path

from ragtune.storage.base import ArtifactSink, StorageUnavailable, StoredArtifact


class S3ArtifactSink(ArtifactSink):
    mode = "s3"

    def __init__(self, *, bucket: str | None, prefix: str = ""):
        if not bucket:
            raise StorageUnavailable("RAGTUNE_S3_BUCKET is required")
        try:
            import boto3  # noqa: F401
        except Exception as exc:
            raise StorageUnavailable("boto3 optional dependency is not installed") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def put_file(self, local_path: Path, artifact_name: str) -> StoredArtifact:
        raise StorageUnavailable("S3 upload is disabled in the publication test harness")
