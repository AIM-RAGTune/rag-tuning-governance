from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_matched_cost.datasets import latest_dataset_manifest
from square_sim.square_tune_matched_cost.paths import scenario_root
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import stable_hash
from square_sim.utils.write_once import unique_id


def duplicate_query_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    owners: dict[str, set[str]] = {}
    for split, frame in splits.items():
        for query in frame["query"].astype(str):
            owners.setdefault(query, set()).add(split)
    leaked = {query: sorted(labels) for query, labels in owners.items() if len(labels) > 1}
    return {"leakage_count": len(leaked), "examples": dict(list(leaked.items())[:10])}


def compile_matched_cost_scenario(settings: Settings, *, split_seed: int = 101, max_rows: int | None = None) -> dict[str, Any]:
    manifest_path = latest_dataset_manifest(settings)
    if manifest_path is None:
        raise FileNotFoundError("No matched-cost RAG dataset manifest found. Run ingest first.")
    dataset_manifest = read_json(manifest_path)
    data = pd.read_parquet(dataset_manifest["data_path"])
    if max_rows:
        data = data.sample(min(max_rows, len(data)), random_state=split_seed).reset_index(drop=True)
    scenario_id = unique_id("real_rag_policy_matched_cost")
    out = scenario_root(settings) / "real_rag_policy_matched_cost" / scenario_id
    out.mkdir(parents=True)
    shuffled = data.sample(frac=1.0, random_state=split_seed).drop_duplicates("query").reset_index(drop=True)
    n = len(shuffled)
    train_end = max(1, int(n * 0.60))
    val_end = max(train_end + 1, int(n * 0.80)) if n > 2 else n
    train = shuffled.iloc[:train_end].copy()
    validation = shuffled.iloc[train_end:val_end].copy()
    test = shuffled.iloc[val_end:].copy()
    if test.empty and not validation.empty:
        test = validation.tail(1).copy()
        validation = validation.iloc[:-1].copy()
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frame = frame.copy()
        frame["split"] = name
        if name == "train":
            train = frame
        elif name == "validation":
            validation = frame
        else:
            test = frame
    scenario = pd.concat([train, validation, test], ignore_index=True)
    splits_dir = out / "splits"
    splits_dir.mkdir()
    scenario.to_parquet(out / "scenario.parquet", index=False)
    train.to_parquet(splits_dir / "train.parquet", index=False)
    validation.to_parquet(splits_dir / "validation.parquet", index=False)
    test.to_parquet(splits_dir / "test.parquet", index=False)
    leak = duplicate_query_leakage({"train": train, "validation": validation, "test": test})
    source_distribution = scenario["source_dataset"].value_counts().to_dict()
    payload = {
        "scenario_id": scenario_id,
        "scenario_name": "real_rag_policy_matched_cost",
        "dataset_manifest_path": str(manifest_path),
        "real_data_used": bool(dataset_manifest.get("real_data_used", False)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(scenario),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "split_seed": split_seed,
        "source_distribution": {str(k): int(v) for k, v in source_distribution.items()},
        "duplicate_query_leakage": leak,
        "policy_space": "RAG policy knobs with matched-cost expensive-compute gating controls.",
        "utility_function": "quality - cost - latency - regression penalty with sensitivity weights.",
        "kill_criteria": [
            "Adaptive compute must beat random matched-cost gating.",
            "Adaptive compute must beat uncertainty-threshold matched-cost gating.",
            "Adaptive compute must beat retrieval-confidence matched-cost gating when available.",
            "Adaptive compute must beat no_fork on cost-adjusted utility.",
        ],
    }
    write_json(out / "scenario_manifest.json", payload)
    write_json(out / "source_distribution.json", payload["source_distribution"])
    write_json(out / "split_manifest.json", {"splits": {"train": len(train), "validation": len(validation), "test": len(test)}, "duplicate_query_leakage": leak})
    write_text(out / "checksums.sha256", f"{stable_hash(payload, 32)}  scenario.parquet\n")
    write_text(
        out / "scenario_card.md",
        "# real_rag_policy_matched_cost\n\n"
        "This kill-test uses real/open RAG evaluation artifacts when available to test whether SQUARETune adaptive compute survives matched-cost gating controls.\n\n"
        "It does not test live RAG deployment, SQUARE hardware, commercial ROI, or broad RAG superiority.\n",
    )
    return payload


def latest_scenario_manifest(settings: Settings) -> Path | None:
    manifests = sorted((scenario_root(settings) / "real_rag_policy_matched_cost").glob("*/scenario_manifest.json"))
    return manifests[-1] if manifests else None
