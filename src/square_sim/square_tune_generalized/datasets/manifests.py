from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash


def write_dataset_artifacts(
    *,
    root: Path,
    dataset_key: str,
    track: str,
    rows: pd.DataFrame,
    license_metadata: dict[str, Any],
    source_note: str,
) -> dict[str, Any]:
    version_id = f"{dataset_key}-{stable_hash({'track': track, 'rows': len(rows), 'time': datetime.now(timezone.utc).isoformat()}, 12)}"
    out = root / track / version_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite dataset version: {out}")
    out.mkdir(parents=True)
    data_path = out / "data.parquet"
    rows.to_parquet(data_path, index=False)
    schema = {col: str(dtype) for col, dtype in rows.dtypes.items()}
    profile = {
        "row_count": len(rows),
        "column_count": len(rows.columns),
        "columns": list(rows.columns),
        "track": track,
    }
    manifest = {
        "dataset_key": dataset_key,
        "dataset_version_id": version_id,
        "track": track,
        "source_note": source_note,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "row_count": len(rows),
        "column_count": len(rows.columns),
        "checksum": sha256_file(data_path),
        "license_status": license_metadata.get("license_status", "unknown"),
        "publication_safe": bool(license_metadata.get("publication_safe")),
    }
    write_json(out / "schema.json", schema)
    write_json(out / "profile.json", profile)
    write_json(out / "dataset_manifest.json", manifest)
    write_json(out / "license_summary.json", license_metadata)
    write_text(
        out / "checksums.sha256",
        f"{manifest['checksum']}  data.parquet\n",
    )
    return manifest
