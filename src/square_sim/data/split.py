from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from square_sim.config import Settings
from square_sim.data.catalog import latest_processed_version, load_dataset_configs
from square_sim.paths import LabPaths
from square_sim.utils.files import write_json
from square_sim.utils.hashing import write_checksums


def _class_balance(series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).sort_index().to_dict().items()}


def create_split(
    dataset_name: str,
    settings: Settings,
    split_id: str = "default",
    seed: int = 42,
    target: str | None = None,
    method: str = "random",
    version: str | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("pandas and scikit-learn are required for splitting. Run `uv sync`.") from exc

    datasets = load_dataset_configs()
    cfg = datasets[dataset_name]
    lab = LabPaths.from_settings(settings)
    version = version or latest_processed_version(settings.project_root, dataset_name)
    target = target or cfg.preferred_first_target
    processed = lab.processed_dir(dataset_name, version) / "data.parquet"
    df = pd.read_parquet(processed)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is absent. Available columns: {list(df.columns)}")

    stratify = df[target] if df[target].nunique(dropna=True) > 1 and df[target].nunique(dropna=True) <= 20 else None
    if method == "chronological":
        ts_cols = [c for c in df.columns if c.lower() in {"timestamp", "date", "datetime"}]
        if not ts_cols:
            raise ValueError("Chronological split requested but no timestamp/date/datetime column was found.")
        df = df.sort_values(ts_cols[0]).reset_index(drop=True)
        n = len(df)
        train, val, test = df.iloc[: int(n * 0.70)], df.iloc[int(n * 0.70) : int(n * 0.85)], df.iloc[int(n * 0.85) :]
    else:
        train, temp = train_test_split(
            df, test_size=0.30, random_state=seed, stratify=stratify
        )
        temp_stratify = temp[target] if stratify is not None and temp[target].nunique() > 1 else None
        val, test = train_test_split(
            temp, test_size=0.50, random_state=seed, stratify=temp_stratify
        )

    split_dir = lab.split_dir(dataset_name, version, split_id)
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": split_dir / "train.parquet",
        "val": split_dir / "val.parquet",
        "test": split_dir / "test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)
    manifest = {
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "target": target,
        "row_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_balance": {
            "train": _class_balance(train[target]),
            "val": _class_balance(val[target]),
            "test": _class_balance(test[target]),
        },
        "source_dataset_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(split_dir / "split_manifest.json", manifest)
    write_checksums(list(paths.values()) + [split_dir / "split_manifest.json"], split_dir / "checksums.sha256")
    return manifest | {"path": str(split_dir)}

