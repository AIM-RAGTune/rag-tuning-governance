from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def availability() -> dict[str, str]:
    try:
        import datasets  # noqa: F401
    except Exception:
        return {"status": "skipped", "reason": "datasets is not installed and no download will be attempted."}
    return {"status": "available"}


def load_hf_sample(item: dict[str, Any], *, max_rows: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency may be absent
        raise RuntimeError("datasets is not installed; use manual import or install optional dependency.") from exc
    dataset_id = item.get("dataset_id")
    if not dataset_id:
        raise ValueError("Hugging Face dataset config requires dataset_id.")
    split = item.get("split", "train")
    config_name = item.get("config_name") or item.get("subset")
    cache_dir = item.get("cache_dir")
    ds = load_dataset(dataset_id, config_name, split=split, cache_dir=cache_dir)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    info = getattr(ds, "info", None)
    metadata = {
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "license": getattr(info, "license", None) if info is not None else None,
        "homepage": getattr(info, "homepage", None) if info is not None else None,
        "citation": getattr(info, "citation", None) if info is not None else None,
    }
    return ds.to_pandas(), metadata


def write_hf_sample(path: Path, item: dict[str, Any], *, max_rows: int | None = None) -> dict[str, Any]:
    df, metadata = load_hf_sample(item, max_rows=max_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    metadata["path"] = str(path)
    metadata["row_count"] = len(df)
    return metadata
