from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.utils.files import read_json, write_json, write_text


def write_dataset_audit(root: Path, output: Path) -> dict[str, Any]:
    manifests = sorted(root.glob("*/*/dataset_manifest.json"))
    rows = [read_json(path) for path in manifests]
    payload = {
        "dataset_count": len(rows),
        "datasets": rows,
        "warnings": [
            "MIMIC and credentialed healthcare datasets are not included unless manually imported by an authorized user.",
            "Synthetic patient-flow proxy contains no PHI and is not a clinical dataset.",
        ],
    }
    write_json(output.with_suffix(".json"), payload)
    lines = ["# Generalized Dataset Audit", ""]
    for row in rows:
        lines.append(f"- `{row['dataset_key']}` rows={row['row_count']} license={row['license_status']}")
    write_text(output.with_suffix(".md"), "\n".join(lines) + "\n")
    return payload
