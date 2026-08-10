from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any, length: int = 10) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def write_checksums(paths: list[Path], output_path: Path) -> dict[str, str]:
    checksums = {str(p): sha256_file(p) for p in sorted(paths)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(f"{digest}  {path}" for path, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )
    return checksums

