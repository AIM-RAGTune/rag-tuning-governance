from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def validate_columns(columns: Iterable[str], expected_targets: list[str], required_target: str | None = None) -> dict:
    cols = set(columns)
    present = [c for c in expected_targets if c in cols]
    missing = [c for c in expected_targets if c not in cols]
    if required_target and required_target not in cols:
        raise ValueError(
            f"Required target column '{required_target}' is absent. Available columns: {sorted(cols)}"
        )
    if not present:
        raise ValueError(
            "None of the expected SPECTRA target columns were found "
            f"({expected_targets}). Available columns: {sorted(cols)}"
        )
    return {"present_targets": present, "missing_targets": missing, "column_count": len(cols)}


def validate_processed_schema(schema_path: Path, expected_targets: list[str], target: str | None = None) -> dict:
    import json

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    columns = [c["name"] for c in schema["columns"]]
    return validate_columns(columns, expected_targets, target)

