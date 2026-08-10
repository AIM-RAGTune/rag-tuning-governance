from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from square_sim.config import Settings
from square_sim.tune.external.hf_datasets import availability, write_hf_sample
from square_sim.tune.external.licenses import license_metadata
from square_sim.tune.external.normalize import normalize_dataset_path
from square_sim.tune.external.paths import external_root
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import stable_hash, write_checksums


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict[str, Any] = {"datasets": {}}
    for include in raw.get("include_configs", []):
        child = _load_config(Path(include))
        merged["datasets"].update(child.get("datasets", {}))
    merged["datasets"].update(raw.get("datasets", {}))
    if "external_datasets" in raw:
        for item in raw.get("external_datasets", []):
            merged["datasets"][item["key"]] = item
    return merged


def _download_id(dataset_key: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{dataset_key}-{stable_hash({'dataset': dataset_key, 'ts': ts}, 8)}"


def _copy_manual_source(source: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        dest = raw_dir / "original"
        if dest.exists():
            raise FileExistsError(f"Manual import destination already exists: {dest}")
        shutil.copytree(source, dest)
        return dest
    dest = raw_dir / source.name
    if dest.exists():
        raise FileExistsError(f"Manual import destination already exists: {dest}")
    shutil.copy2(source, dest)
    return dest


def _find_tabular(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() in {".parquet", ".csv", ".jsonl", ".json"}:
        return path
    for pattern in ["*.parquet", "*.jsonl", "*.json", "*.csv"]:
        matches = sorted(path.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _refresh_catalog(root: Path) -> dict[str, Any]:
    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_path in sorted((root / "normalized").glob("*/*/external_dataset_manifest.json")):
        rows.append(read_json(manifest_path))
    write_json(catalog_dir / "external_dataset_catalog.json", {"datasets": rows})
    if rows:
        pd.DataFrame(rows).to_parquet(catalog_dir / "external_dataset_catalog.parquet", index=False)
    return {"status": "ok", "count": len(rows), "catalog_path": str(catalog_dir / "external_dataset_catalog.json")}


def acquire_external(
    config_path: Path,
    *,
    allow_unknown_license: bool = False,
    output_root: Path | None = None,
    allow_optional: bool = False,
    require_all: bool = False,
    max_rows: int | None = None,
    settings: Settings | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if not config_path.exists():
        return {
            "status": "skipped",
            "reason": f"Config not found: {config_path}",
            "required_action": "Provide a local JSONL/CSV or a licensed Hugging Face dataset config.",
        }
    settings = settings or Settings.from_env()
    root = output_root or external_root(settings)
    cfg = _load_config(config_path)
    datasets = [
        {"key": key, **value}
        for key, value in cfg.get("datasets", {}).items()
        if value.get("enabled", True) and (allow_optional or value.get("required_for_minimal", False))
    ]
    if not datasets:
        return {"status": "skipped", "reason": "No enabled external datasets configured."}
    existing_keys = {
        manifest_path.parent.parent.name
        for manifest_path in sorted((root / "normalized").glob("*/*/external_dataset_manifest.json"))
    }
    results = []
    for item in datasets:
        dataset_key = str(item.get("key"))
        if skip_existing and dataset_key in existing_keys:
            results.append(
                {
                    "key": dataset_key,
                    "status": "exists",
                    "reason": "Normalized external dataset already exists; skipped to preserve append-only imports.",
                }
            )
            continue
        did = _download_id(dataset_key)
        raw_dir = root / "raw" / dataset_key / did
        metadata: dict[str, Any] = {}
        source_path: Path | None = None
        source_metadata: dict[str, Any] = {}
        try:
            manual_path = item.get("manual_import_path")
            if manual_path:
                metadata = license_metadata(item, allow_unknown_license=allow_unknown_license)
                if item.get("license_required", True) and metadata["license_status"] == "missing":
                    status = "blocked" if item.get("required_for_minimal") or require_all else "skipped"
                    results.append(
                        {
                            "key": dataset_key,
                            "status": status,
                            "reason": "Missing license metadata. Use --allow-unknown-license only after review.",
                            "manual_import": f"square-sim tune external import-manual --dataset {dataset_key} --path <path> --license-note <license.txt>",
                        }
                    )
                    continue
                copied = _copy_manual_source(Path(str(manual_path)).expanduser(), raw_dir)
                source_path = _find_tabular(copied)
                source_metadata = {"source_type": "manual", "copied_path": str(copied)}
            elif item.get("source_type") == "huggingface":
                hf = availability()
                if hf["status"] != "available":
                    results.append(
                        {
                            "key": dataset_key,
                            "status": "unavailable",
                            "reason": hf["reason"],
                            "manual_import": f"square-sim tune external import-manual --dataset {dataset_key} --path <path> --license-note <license.txt>",
                        }
                    )
                    continue
                source_path = raw_dir / "original" / "hf_sample.parquet"
                source_metadata = write_hf_sample(
                    source_path,
                    item,
                    max_rows=max_rows or item.get("max_rows"),
                )
                item = {**item, "license": item.get("license") or source_metadata.get("license")}
                metadata = license_metadata(item, allow_unknown_license=allow_unknown_license)
                if item.get("license_required", True) and metadata["license_status"] == "missing":
                    status = "blocked" if item.get("required_for_minimal") or require_all else "skipped"
                    results.append(
                        {
                            "key": dataset_key,
                            "status": status,
                            "reason": "Hugging Face metadata did not expose a license. Retry with --allow-unknown-license only after review.",
                            "manual_import": f"square-sim tune external import-manual --dataset {dataset_key} --path <path> --license-note <license.txt>",
                        }
                    )
                    continue
            elif item.get("source_type") == "huggingface_file":
                try:
                    from huggingface_hub import hf_hub_download
                except Exception as exc:  # pragma: no cover - optional dependency path
                    raise RuntimeError("huggingface_hub is not installed; install datasets or use manual import.") from exc
                dataset_id = str(item.get("dataset_id"))
                filename = str(item.get("data_file") or item.get("filename"))
                if not dataset_id or not filename:
                    raise ValueError("huggingface_file acquisition requires dataset_id and data_file.")
                downloaded = Path(
                    hf_hub_download(
                        repo_id=dataset_id,
                        filename=filename,
                        repo_type="dataset",
                        cache_dir=item.get("cache_dir"),
                    )
                )
                copied = _copy_manual_source(downloaded, raw_dir)
                source_path = _find_tabular(copied)
                source_metadata = {
                    "source_type": "huggingface_file",
                    "dataset_id": dataset_id,
                    "filename": filename,
                    "cache_path": str(downloaded),
                }
                metadata = license_metadata(item, allow_unknown_license=allow_unknown_license)
                if item.get("license_required", True) and metadata["license_status"] == "missing":
                    status = "blocked" if item.get("required_for_minimal") or require_all else "skipped"
                    results.append(
                        {
                            "key": dataset_key,
                            "status": status,
                            "reason": "Hugging Face file metadata did not include a reviewed license.",
                            "manual_import": f"square-sim tune external import-manual --dataset {dataset_key} --path <path> --license-note <license.txt>",
                        }
                    )
                    continue
            else:
                results.append(
                    {
                        "key": dataset_key,
                        "status": "unavailable",
                        "reason": f"Automatic acquisition not implemented for source_type={item.get('source_type')}.",
                        "manual_import": f"square-sim tune external import-manual --dataset {dataset_key} --path <path> --license-note <license.txt>",
                    }
                )
                continue
            if source_path is None or not source_path.exists():
                raise FileNotFoundError(f"No tabular source file detected for {dataset_key}.")
            if not metadata:
                metadata = license_metadata(item, allow_unknown_license=allow_unknown_license)
            write_json(raw_dir / "source_metadata.json", {"dataset": item, "source": source_metadata})
            write_json(raw_dir / "license_metadata.json", metadata)
            write_checksums([source_path], raw_dir / "checksums.sha256")
            manifest = normalize_dataset_path(
                source_path,
                root / "normalized",
                dataset_key=dataset_key,
                scenario_families=list(item.get("scenario_families", [])),
                license_status=str(metadata["license_status"]),
                source_url=str(item.get("dataset_id") or item.get("repository") or ""),
                max_rows=max_rows or item.get("max_rows"),
            )
            manifest["download_id"] = did
            manifest["raw_path"] = str(raw_dir)
            write_text(raw_dir / "acquire.log", f"Acquired {dataset_key} at {datetime.now(timezone.utc).isoformat()}\n")
            results.append({"key": dataset_key, "status": "acquired", "manifest": manifest})
        except Exception as exc:
            status = "failed" if item.get("required_for_minimal") or require_all else "skipped"
            results.append({"key": dataset_key, "status": status, "reason": str(exc)})
    catalog = _refresh_catalog(root)
    report_dir = root.parent.parent.parent / "reports" / "square_tune" / "external_transfer" / "acquisition"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "completed", "output_root": str(root), "results": results, "catalog": catalog}
    write_json(report_dir / "acquisition_report.json", payload)
    lines = ["# SQUARETune External Acquisition Report", "", "| Dataset | Status | Reason |", "|---|---|---|"]
    for row in results:
        lines.append(f"| {row['key']} | {row['status']} | {row.get('reason', '')} |")
    write_text(report_dir / "acquisition_report.md", "\n".join(lines) + "\n")
    failed_required = [row for row in results if row["status"] == "failed"]
    return {
        "status": "failed" if failed_required else "completed",
        "output_root": str(root),
        "results": results,
        "catalog": catalog,
    }


def import_manual_dataset(
    dataset_key: str,
    source_path: Path,
    *,
    output_root: Path,
    license_note: Path | None = None,
    scenario_families: list[str] | None = None,
    allow_unknown_license: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    item = {
        "key": dataset_key,
        "source_type": "manual",
        "manual_import_path": str(source_path),
        "license": license_note.read_text(encoding="utf-8").strip() if license_note and license_note.exists() else None,
        "scenario_families": scenario_families or ["external_transfer"],
    }
    did = _download_id(dataset_key)
    raw_dir = output_root / "raw" / dataset_key / did
    metadata = license_metadata(item, allow_unknown_license=allow_unknown_license)
    if metadata["license_status"] == "missing":
        raise RuntimeError("Manual import requires license metadata or --allow-unknown-license.")
    copied = _copy_manual_source(source_path, raw_dir)
    tabular = _find_tabular(copied)
    if tabular is None:
        raise FileNotFoundError(f"No CSV/JSONL/JSON/Parquet found under {copied}")
    write_json(raw_dir / "source_metadata.json", {"dataset": item, "copied_path": str(copied)})
    write_json(raw_dir / "license_metadata.json", metadata)
    if license_note and license_note.exists():
        shutil.copy2(license_note, raw_dir / "license.txt")
    manifest = normalize_dataset_path(
        tabular,
        output_root / "normalized",
        dataset_key=dataset_key,
        scenario_families=item["scenario_families"],
        license_status=str(metadata["license_status"]),
        source_url=str(source_path),
        max_rows=max_rows,
    )
    manifest["download_id"] = did
    manifest["raw_path"] = str(raw_dir)
    write_json(raw_dir / "file_manifest.json", {"source": str(copied), "tabular": str(tabular)})
    write_checksums([tabular], raw_dir / "checksums.sha256")
    _refresh_catalog(output_root)
    return {"status": "imported", "manifest": manifest}


def refresh_catalog(output_root: Path) -> dict[str, Any]:
    return _refresh_catalog(output_root)
