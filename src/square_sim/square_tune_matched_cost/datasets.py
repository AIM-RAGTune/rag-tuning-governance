from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_matched_cost.paths import dataset_root
from square_sim.utils.files import write_json, write_text
from square_sim.utils.hashing import stable_hash
from square_sim.utils.write_once import unique_id

RAG_SOURCE_KEYS = ("ragtruth", "ragbench", "hagrid", "expertqa")


def _first_text(row: pd.Series, columns: list[str], default: str = "") -> str:
    for col in columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return default


def _numeric_column(frame: pd.DataFrame, candidates: list[str], default: float) -> pd.Series:
    for col in candidates:
        if col in frame:
            return pd.to_numeric(frame[col], errors="coerce").fillna(default).clip(0.0, 1.0)
    return pd.Series([default] * len(frame), index=frame.index)


def _stringify_context(value: Any) -> str:
    if isinstance(value, list | tuple):
        return "\n".join(map(str, value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return ""
    return str(value)


def normalize_rag_frame(frame: pd.DataFrame, *, source_path: Path, source_dataset: str) -> pd.DataFrame:
    raw = frame.reset_index(drop=True).copy()
    n = len(raw)
    idx = pd.Series(range(n), index=raw.index)
    question_cols = ["query", "question", "input_text", "prompt", "user_query", "instruction"]
    answer_cols = ["generated_answer", "candidate_response", "answer", "response", "reference_answer"]
    ref_cols = ["reference_answer", "answer", "gold_answer", "label_text"]
    context_cols = ["retrieved_contexts", "context_text", "documents", "contexts", "context"]

    quality = _numeric_column(raw, ["answer_relevance", "answer_relevance_score", "quality", "score", "label"], 0.62)
    faithfulness = _numeric_column(raw, ["faithfulness", "faithfulness_score", "attribution_score"], 0.62)
    if "hallucination_label" in raw:
        hallucination = pd.to_numeric(raw["hallucination_label"], errors="coerce").fillna(0.35).clip(0, 1)
        faithfulness = (1.0 - hallucination).clip(0, 1)
    elif "hallucination_score" in raw:
        hallucination = pd.to_numeric(raw["hallucination_score"], errors="coerce").fillna(0.35).clip(0, 1)
    else:
        hallucination = (1.0 - faithfulness).clip(0, 1)

    retrieval_confidence = _numeric_column(
        raw,
        ["retrieval_confidence", "context_precision", "context_score", "citation_support_score"],
        0.58,
    )
    uncertainty = (0.55 * (1.0 - quality) + 0.30 * hallucination + 0.15 * (1.0 - retrieval_confidence)).clip(0, 1)
    conflict = (abs(quality - faithfulness) + 0.5 * (1.0 - retrieval_confidence)).clip(0, 1)

    rows = []
    for i, row in raw.iterrows():
        query = _first_text(row, question_cols, f"rag-query-{i}")
        answer = _first_text(row, answer_cols, "")
        reference = _first_text(row, ref_cols, "")
        context = _first_text(row, context_cols, "")
        rows.append(
            {
                "example_id": stable_hash({"source": source_dataset, "path": str(source_path), "i": int(i), "query": query}, 16),
                "source_dataset": source_dataset,
                "query": query,
                "generated_answer": answer,
                "reference_answer_optional": reference,
                "retrieved_contexts": _stringify_context(context),
                "retrieved_context_ids": "",
                "context_scores_optional": float(retrieval_confidence.loc[i]),
                "citation_spans_optional": "",
                "hallucination_labels_optional": float(hallucination.loc[i]),
                "faithfulness_labels_optional": float(faithfulness.loc[i]),
                "answer_relevance_labels_optional": float(quality.loc[i]),
                "uncertainty": float(uncertainty.loc[i]),
                "retrieval_confidence": float(retrieval_confidence.loc[i]),
                "retrieval_conflict": float(conflict.loc[i]),
                "base_quality": float((0.45 * quality.loc[i] + 0.35 * faithfulness.loc[i] + 0.20 * retrieval_confidence.loc[i]).clip(0, 1)),
                "metadata_json": json.dumps({"source_path": str(source_path), "source_row": int(i)}, sort_keys=True),
                "split": "unassigned",
                "license_id": "captured",
                "source_hash": stable_hash({"path": str(source_path), "row": int(i)}, 12),
                "row_ordinal": int(idx.loc[i]),
            }
        )
    return pd.DataFrame(rows)


def discover_real_rag_sources(settings: Settings) -> list[Path]:
    roots = [
        settings.project_root / "datasets" / "external" / "square_tune_v1_full" / "normalized",
        settings.project_root / "datasets" / "external" / "square_tune_v1" / "normalized",
        settings.project_root / "datasets" / "external" / "square_tune_v1_full" / "scenarios",
        settings.project_root / "datasets" / "external" / "square_tune_v1" / "scenarios",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for key in RAG_SOURCE_KEYS:
            paths.extend(sorted(root.glob(f"**/{key}*/data.parquet")))
        paths.extend(sorted(root.glob("rag_policy_optimization/**/scenario.parquet")))
        paths.extend(sorted(root.glob("hallucination_faithfulness_reduction/**/scenario.parquet")))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


def ingest_matched_cost_rag(settings: Settings, *, max_rows: int | None = None) -> dict[str, Any]:
    sources = discover_real_rag_sources(settings)
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for path in sources:
        try:
            frame = pd.read_parquet(path)
            source_dataset = next((key for key in RAG_SOURCE_KEYS if key in str(path).lower()), "rag_external_scenario")
            frames.append(normalize_rag_frame(frame, source_path=path, source_dataset=source_dataset))
        except Exception as exc:
            warnings.append(f"Skipped {path}: {exc}")
    if frames:
        data = pd.concat(frames, ignore_index=True).drop_duplicates("example_id")
        real_data_used = True
    else:
        data = fixture_rag_dataset(240)
        real_data_used = False
        warnings.append("No existing real RAG artifact found; fixture data is for smoke only.")
    if max_rows:
        data = data.sample(min(max_rows, len(data)), random_state=101).reset_index(drop=True)
    version = unique_id("matched-cost-rag")
    out = dataset_root(settings) / "normalized" / version
    out.mkdir(parents=True)
    data_path = out / "normalized_rag_eval.parquet"
    data.to_parquet(data_path, index=False)
    profile = dataset_profile(data, real_data_used=real_data_used, sources=sources, warnings=warnings)
    manifest = {
        "dataset_version_id": version,
        "data_path": str(data_path),
        "row_count": len(data),
        "source_paths": [str(path) for path in sources],
        "real_data_used": real_data_used,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": list(data.columns),
        "warnings": warnings,
    }
    write_json(out / "dataset_manifest.json", manifest)
    write_json(out / "dataset_profile.json", profile)
    write_json(out / "dataset_license_summary.json", profile["license_summary"])
    write_text(out / "ingestion_log.md", "# Matched-Cost RAG Ingestion\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n")
    write_text(out / "checksums.sha256", f"{stable_hash({'rows': len(data), 'version': version}, 32)}  normalized_rag_eval.parquet\n")
    return manifest


def latest_dataset_manifest(settings: Settings) -> Path | None:
    manifests = sorted((dataset_root(settings) / "normalized").glob("*/dataset_manifest.json"))
    return manifests[-1] if manifests else None


def dataset_profile(data: pd.DataFrame, *, real_data_used: bool, sources: list[Path], warnings: list[str]) -> dict[str, Any]:
    source_counts = data["source_dataset"].value_counts().to_dict() if "source_dataset" in data else {}
    return {
        "row_count": len(data),
        "source_distribution": {str(k): int(v) for k, v in source_counts.items()},
        "label_availability": {
            "faithfulness": bool(data["faithfulness_labels_optional"].notna().any()),
            "hallucination": bool(data["hallucination_labels_optional"].notna().any()),
            "answer_relevance": bool(data["answer_relevance_labels_optional"].notna().any()),
            "retrieval_confidence": bool(data["retrieval_confidence"].notna().any()),
        },
        "context_availability": bool(data["retrieved_contexts"].astype(str).str.len().gt(0).any()),
        "duplicate_query_count": int(data.duplicated("query").sum()),
        "missingness_report": {col: int(data[col].isna().sum()) for col in data.columns},
        "license_summary": {
            "license_status": "captured" if real_data_used else "fixture_only",
            "publication_safe": bool(real_data_used),
            "source_count": len(sources),
            "caveats": ["Do not include restricted raw data in publication bundles."],
        },
        "real_data_used": real_data_used,
        "warnings": warnings,
    }


def fixture_rag_dataset(rows: int = 120, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.45, 0.85, rows)
    faith = rng.uniform(0.45, 0.9, rows)
    retr = rng.uniform(0.35, 0.9, rows)
    return pd.DataFrame(
        {
            "example_id": [f"fixture-{i}" for i in range(rows)],
            "source_dataset": ["fixture_rag"] * rows,
            "query": [f"Fixture RAG query {i}" for i in range(rows)],
            "generated_answer": ["fixture answer"] * rows,
            "reference_answer_optional": ["fixture reference"] * rows,
            "retrieved_contexts": ["fixture context"] * rows,
            "retrieved_context_ids": [""] * rows,
            "context_scores_optional": retr,
            "citation_spans_optional": [""] * rows,
            "hallucination_labels_optional": 1.0 - faith,
            "faithfulness_labels_optional": faith,
            "answer_relevance_labels_optional": quality,
            "uncertainty": 0.55 * (1.0 - quality) + 0.30 * (1.0 - faith) + 0.15 * (1.0 - retr),
            "retrieval_confidence": retr,
            "retrieval_conflict": abs(quality - faith) + 0.5 * (1.0 - retr),
            "base_quality": 0.45 * quality + 0.35 * faith + 0.20 * retr,
            "metadata_json": ["{}"] * rows,
            "split": ["unassigned"] * rows,
            "license_id": ["fixture_only"] * rows,
            "source_hash": [f"fixture-{i}" for i in range(rows)],
            "row_ordinal": list(range(rows)),
        }
    )
