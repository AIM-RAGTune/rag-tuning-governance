from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.external.scenarios import SOURCE_MAPPING
from square_sim.utils.files import read_json, write_json, write_text

DATASET_THRESHOLDS = {
    "ragbench": 5000,
    "hagrid": 1000,
    "expertqa": 1000,
    "helpsteer2": 5000,
    "dolly15k": 5000,
    "bfcl": 1000,
}


def _project_root_from_external_root(root: Path) -> Path:
    parts = list(root.parts)
    if "datasets" in parts:
        idx = parts.index("datasets")
        return Path(*parts[:idx]) if idx > 0 else Path("/")
    return root.parent


def data_audit(root: Path, *, strict: bool = False) -> dict[str, Any]:
    rows = []
    for manifest_path in sorted((root / "normalized").glob("*/*/external_dataset_manifest.json")):
        manifest = read_json(manifest_path)
        key = str(manifest.get("dataset_key"))
        row_count = int(manifest.get("row_count", 0))
        warnings = list(manifest.get("warnings", []))
        threshold = DATASET_THRESHOLDS.get(key)
        if threshold and row_count < threshold:
            warnings.append(f"row_count below suggested threshold for {key}: {row_count} < {threshold}")
        rows.append(
            {
                "dataset_key": key,
                "dataset_version_id": manifest.get("dataset_version_id"),
                "row_count": row_count,
                "column_count": manifest.get("column_count"),
                "license_status": manifest.get("license_status"),
                "source_url": manifest.get("source_url"),
                "scenario_families": manifest.get("scenario_families", []),
                "warnings": warnings,
                "status": "warning" if warnings else "ok",
                "suitability": _suitability_for_dataset(key),
            }
        )
    missing = [key for key in ["ragbench", "ifeval", "helpsteer2", "dolly15k", "bfcl"] if key not in {r["dataset_key"] for r in rows}]
    payload = {
        "status": "failed" if strict and missing else "completed",
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": rows,
        "missing_required": missing,
    }
    project_root = _project_root_from_external_root(root)
    report_dir = project_root / "reports" / "square_tune" / "external_transfer"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    write_json(report_dir / f"data_audit_{ts}.json", payload)
    lines = ["# SQUARETune External Data Audit", "", f"Root: `{root}`", "", "| Dataset | Version | Rows | License | Status | Warnings |", "|---|---|---:|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['dataset_key']} | {row['dataset_version_id']} | {row['row_count']} | "
            f"{row['license_status']} | {row['status']} | `{row['warnings']}` |"
        )
    if missing:
        lines.extend(["", "## Missing Required", ""])
        lines.extend(f"- `{item}`" for item in missing)
    write_text(report_dir / f"data_audit_{ts}.md", "\n".join(lines) + "\n")
    return payload


def _suitability_for_dataset(dataset_key: str) -> list[str]:
    return [family for family, mapping in SOURCE_MAPPING.items() if dataset_key in mapping["primary"] | mapping["secondary"]]


def scenario_audit(scenario_root: Path, *, strict: bool = False) -> dict[str, Any]:
    rows = []
    for manifest_path in sorted(scenario_root.glob("*/*/scenario_manifest.json")):
        manifest = read_json(manifest_path)
        family = str(manifest.get("scenario_family"))
        scenario_path = manifest_path.parent / "scenario.parquet"
        distribution: dict[str, int] = {}
        if scenario_path.exists():
            frame = pd.read_parquet(scenario_path, columns=["source_dataset"])
            distribution = dict(Counter(frame["source_dataset"].astype(str)))
        allowed = SOURCE_MAPPING.get(family, {"primary": set(), "secondary": set(), "exclude": set()})
        sources = set(distribution)
        inappropriate = sorted(sources & allowed["exclude"])
        lacks_primary = bool(allowed["primary"]) and not bool(sources & allowed["primary"])
        warnings = []
        if inappropriate:
            warnings.append(f"inappropriate sources present: {inappropriate}")
        if lacks_primary:
            warnings.append(f"missing primary source for {family}")
        if int(manifest.get("row_count", 0)) < 500:
            warnings.append("scenario row count below 500")
        rows.append(
            {
                "scenario_family": family,
                "scenario_id": manifest.get("scenario_id"),
                "row_count": manifest.get("row_count"),
                "source_distribution": distribution,
                "source_distribution_percent": {
                    key: value / max(1, sum(distribution.values())) for key, value in distribution.items()
                },
                "license_summary": manifest.get("license_summary", {}),
                "source_mapping_appropriate": not warnings,
                "warnings": warnings,
            }
        )
    payload = {
        "status": "failed" if strict and any(row["warnings"] for row in rows) else "completed",
        "scenario_root": str(scenario_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": rows,
    }
    project_root = _project_root_from_external_root(scenario_root)
    report_dir = project_root / "reports" / "square_tune" / "external_transfer"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    write_json(report_dir / f"scenario_audit_{ts}.json", payload)
    lines = ["# SQUARETune External Scenario Audit", "", f"Root: `{scenario_root}`", "", "| Scenario | Rows | Sources | Appropriate | Warnings |", "|---|---:|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['scenario_family']} | {row['row_count']} | `{row['source_distribution']}` | "
            f"{row['source_mapping_appropriate']} | `{row['warnings']}` |"
        )
    write_text(report_dir / f"scenario_audit_{ts}.md", "\n".join(lines) + "\n")
    return payload
