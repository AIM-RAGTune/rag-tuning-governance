from __future__ import annotations

import json
from pathlib import Path

from ragtune.generative_validation_common import write_csv, write_json, write_md


POSITIVE = {
    "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY",
    "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY",
    "GEN_LLM_GOVERNANCE_IMPROVES_GENERATED_QUALITY_UNDER_FIXED_BUDGET",
    "GEN_LLM_VALIDATION_LOCAL_OPEN_MODEL_COMPLETED",
    "GEN_LLM_VALIDATION_HOSTED_MODEL_COMPLETED",
    "GEN_LLM_VALIDATION_CRAG_GENERATED_ANSWER_SIGNAL",
    "GEN_LLM_VALIDATION_HOTPOTQA_GENERATED_ANSWER_SIGNAL",
}

NEGATIVE = {
    "GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS",
    "GEN_LLM_GOVERNANCE_NEGATIVE",
}


def load_result(path: Path, default: str) -> dict[str, object]:
    if not path.exists():
        return {"result_class": default}
    return json.loads(path.read_text(encoding="utf-8"))


def synthesize_generative_validation(root: Path, *, output_root: Path) -> dict[str, object]:
    crag = load_result(root / "artifacts/generative_llm_validation/crag/primary_outcome_statistics.json", "GEN_LLM_VALIDATION_BLOCKED_CRAG_UNAVAILABLE")
    hotpotqa = load_result(root / "artifacts/generative_llm_validation/hotpotqa/primary_outcome_statistics.json", "GEN_LLM_VALIDATION_BLOCKED_HOTPOTQA_UNAVAILABLE")
    crag_class = str(crag.get("result_class", ""))
    hotpotqa_class = str(hotpotqa.get("result_class", ""))
    if "NO_GENERATOR" in crag_class and "NO_GENERATOR" in hotpotqa_class:
        result_class = "GEN_LLM_SYNTHESIS_BLOCKED"
        interpretation = "No pinned local or hosted generator was available, so generative validation remains blocked."
    elif crag_class in POSITIVE and hotpotqa_class in POSITIVE:
        result_class = "GEN_LLM_SYNTHESIS_GENERATIVE_VALIDATION_SUPPORTED"
        interpretation = "Both datasets produced generative validation support."
    elif crag_class in POSITIVE or hotpotqa_class in POSITIVE:
        result_class = "GEN_LLM_SYNTHESIS_DIRECTIONAL"
        interpretation = "One dataset produced generative support while the other did not."
    elif crag_class in NEGATIVE and hotpotqa_class in NEGATIVE:
        result_class = "GEN_LLM_SYNTHESIS_NEGATIVE"
        interpretation = "Both datasets showed negative or quality-loss generative results."
    else:
        result_class = "GEN_LLM_SYNTHESIS_INCONCLUSIVE"
        interpretation = "Generative validation did not produce enough usable evidence for a governance claim."
    payload = {
        "suite": "ragtune_generative_llm_validation_synthesis_v1",
        "result_class": result_class,
        "crag_result_class": crag_class,
        "hotpotqa_result_class": hotpotqa_class,
        "prior_frozen_crag_behavioral_result": "GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY",
        "fresh_live_crag_proxy_result": "FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY",
        "hotpotqa_non_generative_result": "HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS",
        "interpretation": interpretation,
        "unsupported_claims": [
            "official platform benchmarking",
            "human validation",
            "production readiness",
            "RAG Compass superiority",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "synthesis_result.json", payload)
    write_json(output_root / "claim_update.json", payload)
    write_csv(
        output_root / "dataset_comparison_table.csv",
        ["dataset", "result_class", "generator_provider", "quality_metric_class"],
        [
            {"dataset": "CRAG", "result_class": crag_class, "generator_provider": crag.get("generator_provider", ""), "quality_metric_class": crag.get("quality_metric_class", "")},
            {"dataset": "HotpotQA", "result_class": hotpotqa_class, "generator_provider": hotpotqa.get("generator_provider", ""), "quality_metric_class": hotpotqa.get("quality_metric_class", "")},
        ],
    )
    report = f"""
# Generative LLM Validation Synthesis

Result class: `{result_class}`

CRAG generative result: `{crag_class}`

HotpotQA generative result: `{hotpotqa_class}`

Interpretation: {interpretation}

No raw prompts, raw generated answers, raw dataset questions, raw source documents, secrets, or private local paths are included.
"""
    for filename in ["synthesis_report.md", "paper_ready_summary.md", "executive_summary.md"]:
        write_md(output_root / filename, report)
    write_md(
        output_root / "limitations.md",
        """
# Generative LLM Validation Limitations

Generative validation remains blocked or inconclusive unless a pinned local or hosted generator actually runs and produces a usable generated-answer quality signal. Local generator validation is not official platform benchmarking. No human validation or production readiness is claimed.
""",
    )
    return payload
