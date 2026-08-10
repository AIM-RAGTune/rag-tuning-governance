from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.catalog import latest_processed_version, load_dataset_configs
from square_sim.paths import LabPaths
from square_sim.utils.files import read_json


@dataclass(frozen=True)
class DatasetVersion:
    dataset_key: str
    display_name: str
    kaggle_slug: str
    dataset_version_id: str
    processed_parquet_path: Path
    schema_path: Path
    profile_path: Path
    validation_report_path: Path | None
    source_manifest_path: Path | None
    row_count: int | None
    column_count: int | None
    target_columns_present: list[str]
    warnings: list[str]
    checksums: dict[str, Any]
    created_at_utc: str | None
    catalog_path: Path | None
    raw_zip_path: Path | None
    download_id: str | None


@dataclass(frozen=True)
class ResolvedDatasetInput:
    dataset_key: str
    target: str
    dataset_version_id: str
    split_id: str
    train_path: Path
    val_path: Path
    test_path: Path
    split_dir: Path
    split_manifest_path: Path
    schema_path: Path
    profile_path: Path
    validation_report_path: Path | None
    source_manifest_path: Path | None
    dataset_catalog_path: Path | None
    processed_parquet_path: Path
    schema: dict[str, Any]
    profile: dict[str, Any]
    validation_report: dict[str, Any]
    split_manifest: dict[str, Any]
    warnings: list[str]
    leakage_warnings: list[str]
    source_metadata: dict[str, Any]


def dataset_catalog_json_path(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "catalog" / "dataset_catalog.json"


def dataset_catalog_parquet_path(settings: Settings) -> Path:
    return settings.project_root / "datasets" / "catalog" / "dataset_catalog.parquet"


def default_split_id(target: str, seed: int) -> str:
    return f"default_target_{target}_seed_{seed}"


def resolved_split_id(split_id: str | None, target: str, seed: int) -> tuple[str, str | None]:
    if split_id in {None, "", "default", "catalog_resolve"}:
        resolved = default_split_id(target, seed)
        warning = (
            f'split_id "{split_id or "default"}" resolved through catalog to {resolved}.'
            if split_id in {"default", None, ""}
            else None
        )
        return resolved, warning
    return split_id, None


def _catalog_entries(settings: Settings) -> list[dict[str, Any]]:
    json_path = dataset_catalog_json_path(settings)
    if json_path.exists():
        data = read_json(json_path)
        if isinstance(data, dict):
            entries = data.get("datasets") or data.get("entries") or data.get("versions") or []
            if isinstance(entries, list):
                return entries
        if isinstance(data, list):
            return data

    parquet_path = dataset_catalog_parquet_path(settings)
    if parquet_path.exists():
        try:
            import pandas as pd

            return pd.read_parquet(parquet_path).to_dict(orient="records")
        except Exception:
            return []
    return []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, tuple | set):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def resolve_dataset_version(
    settings: Settings,
    dataset_key: str,
    dataset_version_id: str | None = None,
    require_ready: bool = True,
) -> DatasetVersion:
    configs = load_dataset_configs()
    entries = [e for e in _catalog_entries(settings) if e.get("dataset_key") == dataset_key or e.get("dataset") == dataset_key]
    cfg = configs.get(dataset_key)
    if require_ready:
        ready = [
            e
            for e in entries
            if str(e.get("status", "ready")).lower() in {"ready", "processed", "succeeded", "complete"}
        ]
        entries = ready or entries
    if dataset_version_id:
        entries = [
            e
            for e in entries
            if e.get("dataset_version_id") == dataset_version_id or e.get("version") == dataset_version_id
        ]

    if entries:
        entries = sorted(entries, key=lambda e: str(e.get("created_at_utc") or e.get("created_at") or ""))
        entry = entries[-1]
        version = str(entry.get("dataset_version_id") or entry.get("version"))
        processed = Path(entry.get("processed_parquet_path") or settings.project_root / "datasets" / "processed" / dataset_key / version / "data.parquet")
        processed_dir = processed.parent
        schema_path = Path(entry.get("schema_path") or processed_dir / "schema.json")
        profile_path = Path(entry.get("profile_path") or processed_dir / "profile.json")
        validation_path = _first_existing(
            Path(str(entry["validation_report_path"])) if entry.get("validation_report_path") else None,
            processed_dir / "validation_report.json",
        )
        source_path = _first_existing(
            Path(str(entry["source_manifest_path"])) if entry.get("source_manifest_path") else None,
            processed_dir / "source_manifest.json",
        )
        return DatasetVersion(
            dataset_key=dataset_key,
            display_name=str(entry.get("display_name") or (cfg.display_name if cfg else dataset_key)),
            kaggle_slug=str(entry.get("kaggle_slug") or (cfg.kaggle_slug if cfg else "synthetic/local")),
            dataset_version_id=version,
            processed_parquet_path=processed,
            schema_path=schema_path,
            profile_path=profile_path,
            validation_report_path=validation_path,
            source_manifest_path=source_path,
            row_count=int(entry["row_count"]) if entry.get("row_count") is not None else None,
            column_count=int(entry["column_count"]) if entry.get("column_count") is not None else None,
            target_columns_present=_string_list(
                entry.get("target_columns_present") or entry.get("target_columns") or entry.get("present_targets")
            ),
            warnings=_string_list(entry.get("warnings")),
            checksums=entry.get("checksums") if isinstance(entry.get("checksums"), dict) else {},
            created_at_utc=entry.get("created_at_utc") or entry.get("created_at"),
            catalog_path=dataset_catalog_json_path(settings) if dataset_catalog_json_path(settings).exists() else None,
            raw_zip_path=Path(str(entry["raw_zip_path"])) if entry.get("raw_zip_path") else None,
            download_id=entry.get("download_id"),
        )

    version = dataset_version_id or latest_processed_version(settings.project_root, dataset_key)
    processed_dir = LabPaths.from_settings(settings).processed_dir(dataset_key, version)
    schema_path = processed_dir / "schema.json"
    schema = read_json(schema_path) if schema_path.exists() else {}
    columns = schema.get("columns", [])
    expected_targets = cfg.expected_targets if cfg else ["target", "target_real", "in_pocket"]
    present_targets = [c for c in expected_targets if c in {col.get("name") for col in columns}]
    return DatasetVersion(
        dataset_key=dataset_key,
        display_name=cfg.display_name if cfg else dataset_key,
        kaggle_slug=cfg.kaggle_slug if cfg else "synthetic/local",
        dataset_version_id=version,
        processed_parquet_path=processed_dir / "data.parquet",
        schema_path=schema_path,
        profile_path=processed_dir / "profile.json",
        validation_report_path=_first_existing(processed_dir / "validation_report.json"),
        source_manifest_path=_first_existing(processed_dir / "source_manifest.json"),
        row_count=schema.get("row_count"),
        column_count=schema.get("column_count"),
        target_columns_present=present_targets,
        warnings=_string_list(schema.get("warnings")),
        checksums={},
        created_at_utc=None,
        catalog_path=None,
        raw_zip_path=None,
        download_id=None,
    )


def _load_optional(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        try:
            return read_json(path)
        except Exception:
            return {"warnings": [f"Could not parse JSON file: {path}"]}
    return {}


def collect_warnings(*payloads: dict[str, Any], catalog_warnings: list[str] | None = None) -> list[str]:
    warnings: list[str] = []
    warnings.extend(catalog_warnings or [])
    for payload in payloads:
        for key in ("warnings", "leakage_warnings", "data_warnings"):
            warnings.extend(_string_list(payload.get(key)))
        for item in payload.get("columns", []) if isinstance(payload.get("columns"), list) else []:
            warnings.extend(_string_list(item.get("warnings") if isinstance(item, dict) else None))
    seen: set[str] = set()
    unique = []
    for warning in warnings:
        if warning and warning not in seen:
            seen.add(warning)
            unique.append(warning)
    return unique


def leakage_warnings_from(warnings: list[str], schema: dict[str, Any]) -> list[str]:
    leakage_terms = {"leakage", "target", "label", "outcome", "churn", "failure", "pocket", "class", "segment"}
    found = [w for w in warnings if any(term in w.lower() for term in leakage_terms)]
    for col in schema.get("columns", []) if isinstance(schema.get("columns"), list) else []:
        name = str(col.get("name", ""))
        role = str(col.get("role", ""))
        if role in {"target", "target_real", "pocket_flag"}:
            continue
        if any(term in name.lower() for term in leakage_terms):
            found.append(f"Potential leakage-like column name: {name}")
    seen: set[str] = set()
    return [w for w in found if not (w in seen or seen.add(w))]


def resolve_dataset_input(
    settings: Settings,
    dataset_key: str,
    target: str,
    seed: int = 42,
    split_id: str | None = "default",
    dataset_version_id: str | None = None,
) -> ResolvedDatasetInput:
    version = resolve_dataset_version(settings, dataset_key, dataset_version_id)
    resolved_id, warning = resolved_split_id(split_id, target, seed)
    split_dir = LabPaths.from_settings(settings).split_dir(dataset_key, version.dataset_version_id, resolved_id)
    manifest_path = split_dir / "split_manifest.json"
    required_paths = {
        "train": split_dir / "train.parquet",
        "val": split_dir / "val.parquet",
        "test": split_dir / "test.parquet",
        "split_manifest": manifest_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        command = (
            f"square-ingest split --dataset {dataset_key} --version-id {version.dataset_version_id} "
            f"--target {target} --seed {seed}"
        )
        raise FileNotFoundError(
            f"Missing split for dataset={dataset_key}, version={version.dataset_version_id}, "
            f"target={target}, seed={seed}. Expected split_id={resolved_id}. Missing: {missing}. "
            f"Run: {command}"
        )

    schema = _load_optional(version.schema_path)
    profile = _load_optional(version.profile_path)
    validation = _load_optional(version.validation_report_path)
    split_manifest = _load_optional(manifest_path)
    if split_manifest.get("target") and split_manifest["target"] != target:
        raise ValueError(
            f"Split target mismatch: requested {target}, but {manifest_path} records {split_manifest['target']}."
        )
    manifest_version = split_manifest.get("source_dataset_version") or split_manifest.get("dataset_version_id")
    if manifest_version and manifest_version != version.dataset_version_id:
        raise ValueError(
            f"Split dataset version mismatch: requested {version.dataset_version_id}, "
            f"but {manifest_path} records {manifest_version}."
        )
    warnings = collect_warnings(
        schema,
        profile,
        validation,
        split_manifest,
        catalog_warnings=version.warnings + ([warning] if warning else []),
    )
    leakage = leakage_warnings_from(warnings, schema)
    return ResolvedDatasetInput(
        dataset_key=dataset_key,
        target=target,
        dataset_version_id=version.dataset_version_id,
        split_id=resolved_id,
        train_path=required_paths["train"],
        val_path=required_paths["val"],
        test_path=required_paths["test"],
        split_dir=split_dir,
        split_manifest_path=manifest_path,
        schema_path=version.schema_path,
        profile_path=version.profile_path,
        validation_report_path=version.validation_report_path,
        source_manifest_path=version.source_manifest_path,
        dataset_catalog_path=version.catalog_path,
        processed_parquet_path=version.processed_parquet_path,
        schema=schema,
        profile=profile,
        validation_report=validation,
        split_manifest=split_manifest,
        warnings=warnings,
        leakage_warnings=leakage,
        source_metadata={
            "display_name": version.display_name,
            "kaggle_slug": version.kaggle_slug,
            "raw_zip_path": str(version.raw_zip_path) if version.raw_zip_path else None,
            "download_id": version.download_id,
            "checksums": version.checksums,
            "created_at_utc": version.created_at_utc,
        },
    )
