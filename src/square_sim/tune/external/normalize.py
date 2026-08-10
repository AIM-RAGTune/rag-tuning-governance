from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.tune.external.licenses import scan_pii_phi_texts
from square_sim.tune.external.schemas import EXTERNAL_COLUMNS
from square_sim.utils.files import write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash, write_checksums


def _text(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return " ".join(str(item) for item in converted)
            return str(converted)
        try:
            if pd.isna(value):
                continue
        except ValueError:
            pass
        return str(value)
    return ""


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.5


def read_tabular(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        try:
            return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        except ValueError:
            return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def normalize_frame(
    df: pd.DataFrame,
    *,
    dataset_key: str,
    scenario_families: list[str],
    license_status: str,
    source_url: str = "",
    source_subset: str = "",
    max_rows: int | None = None,
) -> pd.DataFrame:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    rows: list[dict[str, Any]] = []
    scenario_family = scenario_families[0] if scenario_families else "external_transfer"
    for idx, row in df.reset_index(drop=True).iterrows():
        input_text = _text(row, ["input_text", "prompt", "question", "instruction", "query", "user_query"])
        context = _text(row, ["context_text", "context", "documents", "passages", "tool_schema"])
        reference = _text(row, ["reference_answer", "answer", "response", "output", "ground_truth"])
        candidate = _text(row, ["candidate_response", "generated_answer", "model_response", "response"])
        label_value = row.get("label", row.get("score", row.get("rating", row.get("target_utility", 0.5))))
        score_vector = {
            "quality": _score(row.get("quality", label_value)),
            "faithfulness": _score(row.get("faithfulness", label_value)),
            "safety": _score(row.get("safety", 0.75)),
            "cost": _score(row.get("cost", 0.25)),
        }
        metadata = {
            "original_columns": list(df.columns),
            "category": str(row.get("category", row.get("failure_cluster", ""))),
            "raw_index": int(idx),
        }
        base = {
            "source_dataset": dataset_key,
            "source_subset": source_subset,
            "source_split": str(row.get("split", row.get("source_split", "unknown"))),
            "source_record_id": str(row.get("id", row.get("row_id", idx))),
            "task_family": str(row.get("task_family", scenario_family)),
            "scenario_family": scenario_family,
            "input_text": input_text,
            "context_text": context,
            "reference_answer": reference,
            "candidate_response": candidate,
            "label": label_value,
            "label_text": str(label_value),
            "score_vector_json": json.dumps(score_vector, sort_keys=True),
            "metadata_json": json.dumps(metadata, sort_keys=True),
            "license_status": license_status,
            "source_url": source_url,
        }
        digest = hashlib.sha256(json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()
        base["row_id"] = f"{dataset_key}-{digest[:16]}"
        base["row_checksum"] = digest
        rows.append(base)
    out = pd.DataFrame(rows)
    for col in EXTERNAL_COLUMNS:
        if col not in out:
            out[col] = ""
    return out[EXTERNAL_COLUMNS]


def normalize_dataset_path(
    source_path: Path,
    output_dir: Path,
    *,
    dataset_key: str,
    scenario_families: list[str],
    license_status: str,
    source_url: str = "",
    source_subset: str = "",
    max_rows: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_tabular(source_path)
    normalized = normalize_frame(
        df,
        dataset_key=dataset_key,
        scenario_families=scenario_families,
        license_status=license_status,
        source_url=source_url,
        source_subset=source_subset,
        max_rows=max_rows,
    )
    version_id = f"{dataset_key}-{stable_hash(normalized.head(1000).to_dict(orient='list'), 12)}"
    version_dir = output_dir / dataset_key / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = version_dir / "data.parquet"
    normalized.to_parquet(parquet_path, index=False)
    pii = scan_pii_phi_texts(
        (normalized["input_text"].fillna("") + " " + normalized["context_text"].fillna("")).tolist()
    )
    manifest = {
        "dataset_key": dataset_key,
        "dataset_version_id": version_id,
        "normalized_path": str(parquet_path),
        "row_count": len(normalized),
        "column_count": len(normalized.columns),
        "columns": list(normalized.columns),
        "checksum": sha256_file(parquet_path),
        "license_status": license_status,
        "source_url": source_url,
        "warnings": pii["warnings"],
    }
    write_json(version_dir / "external_dataset_manifest.json", manifest)
    write_json(version_dir / "schema.json", {"columns": list(normalized.columns), "row_count": len(normalized)})
    write_json(version_dir / "profile.json", {"row_count": len(normalized), "pii_phi_scan": pii})
    write_text(
        version_dir / "normalization_report.md",
        f"# Normalization Report: {dataset_key}\n\nRows: `{len(normalized)}`\n\nWarnings: `{pii['warnings']}`\n",
    )
    write_checksums([parquet_path, version_dir / "external_dataset_manifest.json"], version_dir / "checksums.sha256")
    return manifest
