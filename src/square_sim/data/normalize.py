from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.catalog import load_dataset_configs
from square_sim.data.manifest import file_inventory
from square_sim.data.schema import TARGET_ROLES
from square_sim.data.validate import validate_columns
from square_sim.paths import LabPaths
from square_sim.utils.files import latest_child, write_json
from square_sim.utils.hashing import write_checksums

TABULAR_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".jsonl"}


def _require_pandas():
    try:
        import pandas as pd

        return pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for normalization. Run `uv sync`.") from exc


def find_tabular_files(staging_dir: Path) -> list[Path]:
    files = [p for p in staging_dir.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No tabular files found under {staging_dir}.")
    return sorted(files)


def read_tabular(path: Path):
    pd = _require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported tabular file type: {path}")


def _role_guess(name: str) -> str:
    lower = name.lower()
    if name in TARGET_ROLES:
        return TARGET_ROLES[name]
    if any(token in lower for token in ["id", "uuid", "timestamp", "date"]):
        return "metadata"
    return "feature"


def data_dictionary(df) -> list[dict[str, Any]]:
    dictionary = []
    for col in df.columns:
        series = df[col]
        item: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "distinct_count": int(series.nunique(dropna=True)),
            "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
            "role_guess": _role_guess(str(col)),
        }
        if hasattr(series, "min") and str(series.dtype) != "object":
            try:
                item["min"] = float(series.min())
                item["max"] = float(series.max())
            except Exception:
                pass
        dictionary.append(item)
    return dictionary


def profile_dataframe(df, expected_targets: list[str]) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "target_validation": validate_columns(df.columns, expected_targets),
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
    }


def latest_staging_version(settings: Settings, dataset_name: str) -> str:
    latest = latest_child(settings.project_root / "datasets" / "staging" / dataset_name)
    if latest is None:
        raise FileNotFoundError(
            f"No staging files found for '{dataset_name}'. Run acquisition or use --offline-zip first."
        )
    return latest.name


def normalize_dataset(dataset_name: str, settings: Settings, version: str | None = None) -> dict:
    pd = _require_pandas()
    datasets = load_dataset_configs()
    cfg = datasets[dataset_name]
    lab = LabPaths.from_settings(settings)
    version = version or latest_staging_version(settings, dataset_name)
    staging_dir = lab.staging_dir(dataset_name, version)
    processed_dir = lab.processed_dir(dataset_name, version)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frames = [read_tabular(path) for path in find_tabular_files(staging_dir)]
    df = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    validation = validate_columns(df.columns, cfg.expected_targets)
    parquet_path = processed_dir / "data.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError("Writing Parquet requires pyarrow or fastparquet. Run `uv sync`.") from exc

    dictionary = data_dictionary(df)
    schema = {
        "dataset": dataset_name,
        "version": version,
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "original_columns_preserved": True,
    }
    profile = profile_dataframe(df, cfg.expected_targets)
    write_json(processed_dir / "data_dictionary.json", dictionary)
    write_json(processed_dir / "schema.json", schema)
    write_json(processed_dir / "profile.json", profile)
    write_json(processed_dir / "manifest.json", {"files": file_inventory(processed_dir), "validation": validation})
    write_checksums(
        [parquet_path, processed_dir / "data_dictionary.json", processed_dir / "schema.json", processed_dir / "profile.json"],
        processed_dir / "checksums.sha256",
    )
    return {"dataset": dataset_name, "version": version, "rows": len(df), "processed_path": str(processed_dir)}

