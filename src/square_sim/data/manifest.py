from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.utils.files import write_json
from square_sim.utils.hashing import sha256_file, write_checksums


def file_inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def write_source_manifest(
    manifest_path: Path,
    dataset: str,
    slug: str,
    source_url: str,
    local_path: Path,
    version: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset": dataset,
        "slug": slug,
        "source_url": source_url,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "local_path": str(local_path),
        "files": file_inventory(local_path),
    }
    if extra:
        payload.update(extra)
    write_json(manifest_path, payload)
    write_checksums([Path(f["path"]) for f in payload["files"]], manifest_path.parent / "checksums.sha256")
    return payload

