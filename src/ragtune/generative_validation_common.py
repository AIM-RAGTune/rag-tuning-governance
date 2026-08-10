from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

GEN_LLM_RESULT_CLASSES = {
    "GEN_LLM_VALIDATION_LOCAL_OPEN_MODEL_COMPLETED",
    "GEN_LLM_VALIDATION_HOSTED_MODEL_COMPLETED",
    "GEN_LLM_VALIDATION_CRAG_GENERATED_ANSWER_SIGNAL",
    "GEN_LLM_VALIDATION_HOTPOTQA_GENERATED_ANSWER_SIGNAL",
    "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY",
    "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY",
    "GEN_LLM_GOVERNANCE_IMPROVES_GENERATED_QUALITY_UNDER_FIXED_BUDGET",
    "GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS",
    "GEN_LLM_GOVERNANCE_MATCHES_QUALITY_ONLY",
    "GEN_LLM_GOVERNANCE_INCONCLUSIVE",
    "GEN_LLM_GOVERNANCE_NEGATIVE",
    "GEN_LLM_VALIDATION_BLOCKED_NO_GENERATOR",
    "GEN_LLM_VALIDATION_BLOCKED_NO_MODEL_CREDENTIALS",
    "GEN_LLM_VALIDATION_BLOCKED_CRAG_UNAVAILABLE",
    "GEN_LLM_VALIDATION_BLOCKED_HOTPOTQA_UNAVAILABLE",
    "GEN_LLM_VALIDATION_BLOCKED_EVALUATOR_MAPPING",
    "GEN_LLM_VALIDATION_BLOCKED_NO_USABLE_QUALITY_SIGNAL",
    "GEN_LLM_VALIDATION_BLOCKED_PUBLICATION_HYGIENE",
    "GEN_LLM_VALIDATION_BLOCKED_RATE_LIMIT",
    "GEN_LLM_VALIDATION_BLOCKED_COST_LIMIT",
}

GEN_LLM_SYNTHESIS_CLASSES = {
    "GEN_LLM_SYNTHESIS_GENERATIVE_VALIDATION_SUPPORTED",
    "GEN_LLM_SYNTHESIS_DIRECTIONAL",
    "GEN_LLM_SYNTHESIS_MIXED",
    "GEN_LLM_SYNTHESIS_INCONCLUSIVE",
    "GEN_LLM_SYNTHESIS_BLOCKED",
    "GEN_LLM_SYNTHESIS_NEGATIVE",
}

GENERATION_FIELDNAMES = [
    "example_id",
    "question_hash",
    "split",
    "dataset",
    "policy_id",
    "provider",
    "model",
    "prompt_hash",
    "generated_answer_hash",
    "generated_answer_char_count",
    "generated_answer_token_estimate",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "retrieval_cost_units",
    "generation_cost_units",
    "total_cost_units",
    "input_token_estimate",
    "output_token_estimate",
    "api_call_count",
    "generator_call_count",
    "answer_correctness_f1",
    "answer_exact_match",
    "answer_containment",
    "evidence_support_score",
    "citation_support_score",
    "abstention_correctness",
    "final_generated_quality_score",
    "raw_prompt_exported",
    "raw_generated_answer_exported",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def zero_ci() -> dict[str, float]:
    return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
