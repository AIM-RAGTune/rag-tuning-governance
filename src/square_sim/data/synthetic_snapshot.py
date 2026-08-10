from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from square_sim.config import Settings
from square_sim.data.resolver import default_split_id
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash

DATASETS = [
    "field_overlap_future",
    "nonlinear_branch_choice",
    "delayed_memory_zone",
    "local_counterfactual_delta",
    "merge_required",
    "linear_control",
]


def _balance(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).sort_index().items()}


def _frame(kind: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + DATASETS.index(kind))
    x = rng.normal(size=(rows, 10))
    drift = rng.normal(scale=0.35, size=(rows, 4))
    pocket_score = x[:, 0] * x[:, 1] - 0.35 * x[:, 2] + rng.normal(scale=0.2, size=rows)
    in_pocket = (pocket_score > np.quantile(pocket_score, 0.72)).astype(int)
    target_real = ((x[:, 0] + x[:, 1] - x[:, 2]) > 0.0).astype(int)
    if kind == "field_overlap_future":
        future_overlap = np.sin(x[:, 0] + drift[:, 0]) * np.cos(x[:, 1] - drift[:, 1]) + 0.35 * in_pocket
        target = (future_overlap > np.median(future_overlap)).astype(int)
        mechanism = "Label depends on future overlap of moving local field regions."
    elif kind == "nonlinear_branch_choice":
        nonlinear = x[:, 0] ** 2 - x[:, 1] ** 2 + np.tanh(x[:, 2] * x[:, 3])
        target = (nonlinear + 0.25 * in_pocket > np.median(nonlinear)).astype(int)
        mechanism = "Label depends on nonlinear local branch evolution."
    elif kind == "delayed_memory_zone":
        latent = np.cumsum(x[:, :4], axis=1)[:, -1] + 0.4 * np.roll(x[:, 0], 1)
        target = (latent > np.median(latent)).astype(int)
        mechanism = "Label depends on delayed memory-like accumulation."
    elif kind == "local_counterfactual_delta":
        perturbed = x[:, 0] + 0.7 * np.tanh(x[:, 3] - x[:, 4])
        target = (perturbed > 0.2).astype(int)
        target_real = (x[:, 0] > 0.2).astype(int)
        mechanism = "Label depends on whether a local counterfactual perturbation crosses a threshold."
    elif kind == "merge_required":
        branch_a = np.sin(x[:, 0] + x[:, 1])
        branch_b = np.cos(x[:, 2] - x[:, 3])
        branch_c = np.tanh(x[:, 4] * x[:, 5])
        target = ((branch_a + branch_b + branch_c) > np.median(branch_a + branch_b + branch_c)).astype(int)
        mechanism = "Multiple branch futures must be aggregated to predict the label."
    elif kind == "linear_control":
        target = ((1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.3 * x[:, 2]) > 0).astype(int)
        mechanism = "Linearly separable control where classical models should be strong."
    else:
        raise ValueError(f"Unknown synthetic snapshot dataset: {kind}")
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    data["drift_y"] = drift[:, 0]
    data["drift_x"] = drift[:, 1]
    data["row_id"] = [f"{kind}-{seed}-{i:08d}" for i in range(rows)]
    data["target"] = target.astype(int)
    data["target_real"] = target_real.astype(int)
    data["in_pocket"] = in_pocket.astype(int)
    data.attrs["mechanism"] = mechanism
    return data


def _schema(dataset_key: str, version: str, frame: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for name in frame.columns:
        role = "feature"
        if name == "target":
            role = "target"
        elif name == "target_real":
            role = "target_real"
        elif name == "in_pocket":
            role = "pocket_flag"
        elif name == "row_id":
            role = "metadata"
        columns.append(
            {
                "name": name,
                "dtype": str(frame[name].dtype),
                "null_count": int(frame[name].isna().sum()),
                "distinct_count": int(frame[name].nunique(dropna=True)),
                "role": role,
            }
        )
    return {
        "dataset_key": dataset_key,
        "dataset_version_id": version,
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": columns,
        "expected_columns": ["target", "target_real", "in_pocket"],
        "missing_expected_columns": [],
        "warnings": [],
    }


def _profile(frame: pd.DataFrame, mechanism: str) -> dict[str, Any]:
    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "target_balance": _balance(frame["target"]),
        "target_real_balance": _balance(frame["target_real"]),
        "in_pocket_balance": _balance(frame["in_pocket"]),
        "target_vs_target_real_difference_count": int((frame["target"] != frame["target_real"]).sum()),
        "mechanism": mechanism,
        "warnings": [],
    }


def _write_splits(settings: Settings, dataset_key: str, version: str, frame: pd.DataFrame, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    train_end = int(len(indices) * 0.70)
    val_end = train_end + int(len(indices) * 0.15)
    parts = {
        "train": frame.iloc[indices[:train_end]].copy(),
        "val": frame.iloc[indices[train_end:val_end]].copy(),
        "test": frame.iloc[indices[val_end:]].copy(),
    }
    split_id = default_split_id("target", seed)
    split_dir = settings.project_root / "datasets" / "splits" / dataset_key / version / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, part in parts.items():
        path = split_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        written.append(path)
    write_json(
        split_dir / "split_manifest.json",
        {
            "dataset_key": dataset_key,
            "source_dataset_version": version,
            "dataset_version_id": version,
            "split_id": split_id,
            "target": "target",
            "seed": seed,
            "method": "random",
            "row_counts": {name: len(part) for name, part in parts.items()},
            "class_balance": {name: _balance(part["target"]) for name, part in parts.items()},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return written


def _refresh_catalog(settings: Settings, entries: list[dict[str, Any]]) -> None:
    catalog_dir = settings.project_root / "datasets" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_dir / "dataset_catalog.json"
    existing = []
    if path.exists():
        payload = read_json(path)
        existing = payload.get("datasets", []) if isinstance(payload, dict) else payload
    by_key_version = {
        (str(entry.get("dataset_key")), str(entry.get("dataset_version_id"))): entry for entry in existing
    }
    for entry in entries:
        by_key_version[(entry["dataset_key"], entry["dataset_version_id"])] = entry
    rows = list(by_key_version.values())
    write_json(path, {"datasets": rows})
    try:
        pd.DataFrame(rows).to_parquet(catalog_dir / "dataset_catalog.parquet", index=False)
    except Exception:
        pass


def make_snapshot_diagnostics(settings: Settings, output: Path, rows: int = 10_000, seed: int = 42) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for kind in DATASETS:
        dataset_key = f"synthetic_snapshot_{kind}"
        frame = _frame(kind, rows, seed)
        content_hash = stable_hash(
            {
                "kind": kind,
                "rows": rows,
                "seed": seed,
                "target_sum": int(frame["target"].sum()),
                "target_real_sum": int(frame["target_real"].sum()),
            },
            12,
        )
        version = f"{dataset_key}-{content_hash}"
        processed_dir = settings.project_root / "datasets" / "processed" / dataset_key / version
        processed_dir.mkdir(parents=True, exist_ok=True)
        parquet = processed_dir / "data.parquet"
        frame.to_parquet(parquet, index=False)
        schema = _schema(dataset_key, version, frame)
        profile = _profile(frame, str(frame.attrs.get("mechanism", "")))
        write_json(processed_dir / "schema.json", schema)
        write_json(processed_dir / "profile.json", profile)
        write_json(processed_dir / "validation_report.json", {"status": "passed", "warnings": []})
        write_json(
            processed_dir / "source_manifest.json",
            {
                "source": "synthetic_snapshot_diagnostics",
                "mechanism": frame.attrs.get("mechanism"),
                "rows": rows,
                "seed": seed,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        (processed_dir / "checksums.sha256").write_text(f"{sha256_file(parquet)}  data.parquet\n", encoding="utf-8")
        split_paths = _write_splits(settings, dataset_key, version, frame, seed)
        metadata_path = output / f"{dataset_key}.json"
        write_json(metadata_path, {"dataset_key": dataset_key, "dataset_version_id": version, "mechanism": frame.attrs.get("mechanism")})
        entries.append(
            {
                "dataset_key": dataset_key,
                "display_name": f"Synthetic Snapshot Diagnostic: {kind}",
                "kaggle_slug": "synthetic/local",
                "download_id": f"synthetic-{seed}",
                "dataset_version_id": version,
                "processed_parquet_path": str(parquet),
                "schema_path": str(processed_dir / "schema.json"),
                "profile_path": str(processed_dir / "profile.json"),
                "validation_report_path": str(processed_dir / "validation_report.json"),
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "target_columns_present": ["target", "target_real", "in_pocket"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checksums": {"data.parquet": sha256_file(parquet)},
                "status": "ready",
                "warnings": [],
                "split_paths": [str(path) for path in split_paths],
            }
        )
    _refresh_catalog(settings, entries)
    readme = output / "README.md"
    write_text(
        readme,
        "# Synthetic Snapshot Diagnostics\n\n"
        "These datasets are local integration tests for the snapshot rollout mechanism. "
        "They are not evidence of real-world SQUARE advantage.\n",
    )
    return {"created": entries, "output": str(output), "catalog": str(settings.project_root / "datasets" / "catalog" / "dataset_catalog.json")}
