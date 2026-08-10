from __future__ import annotations

from pathlib import Path
from typing import Any


def github_manual_fallback(dataset_key: str, repository: str | None, output_root: Path) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "status": "manual_required",
        "repository": repository,
        "output_root": str(output_root),
        "manual_import": (
            f"square-sim tune external import-manual --dataset {dataset_key} "
            "--path <local_download_or_clone> --license-note <license.txt>"
        ),
        "reason": "GitHub acquisition is intentionally optional for External Transfer v1; use manual import when source format varies.",
    }
