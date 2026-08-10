from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_hotpotqa_quality_signal_audit_manifest_exists() -> None:
    manifest = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json")
    assert manifest["suite"] == "ragtune_hotpotqa_generative_quality_signal_audit_v1"
    assert "result_class" in manifest


def test_hotpotqa_quality_signal_audit_checks_answer_hash_diversity() -> None:
    manifest = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json")
    assert "unique_answer_hash_count" in manifest
    assert int(manifest["unique_answer_hash_count"]) >= 0


def test_hotpotqa_quality_signal_audit_checks_quality_variance() -> None:
    manifest = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json")
    assert "quality_variance" in manifest
    assert float(manifest["quality_variance"]) >= 0.0


def test_hotpotqa_zero_quality_delta_not_auto_promoted() -> None:
    stats = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/primary_outcome_statistics.json")
    if stats["generated_quality_delta"]["mean"] == 0.0 and stats["cost_delta"]["mean"] == 0.0:
        assert stats["result_class"] != "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY"


def test_hotpotqa_constant_zero_quality_blocks_success() -> None:
    stats = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/primary_outcome_statistics.json")
    if stats.get("quality_variance") == 0.0 or not stats.get("usable_quality_signal"):
        assert stats["result_class"] not in {
            "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY",
            "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY",
        }


def test_generator_access_diagnosis_artifacts_sanitized() -> None:
    diagnosis = load_json("deployment_review/generative_llm_validation_quality_signal_audit/generator_access_diagnosis.json")
    assert diagnosis["raw_test_response_committed"] is False
    assert "test_answer_hash" in diagnosis
    assert "test_answer_text" not in diagnosis


def test_crag_generator_uses_shared_generator_factory() -> None:
    source = (ROOT / "src/ragtune/crag_generative_validation.py").read_text(encoding="utf-8")
    assert "discover_generator" in source
    assert "from ragtune.generators.factory import discover_generator" in source


def test_qwen3_ollama_disables_thinking_by_default() -> None:
    source = (ROOT / "src/ragtune/generators/ollama.py").read_text(encoding="utf-8")
    assert "RAGTUNE_OLLAMA_THINK" in source
    assert 'model.lower().startswith("qwen3")' in source
    assert 'payload["think"] = think' in source


def test_crag_evaluator_mapping_result_class_machine_readable() -> None:
    stats = load_json("artifacts/generative_llm_validation/crag/primary_outcome_statistics.json")
    assert stats["crag_evaluator_mapping_result_class"] in {
        "CRAG_GENERATED_QUALITY_LOCAL_EVALUATOR_ACTIVE",
        "CRAG_GENERATED_QUALITY_LOCAL_EVALUATOR_PARTIAL",
        "CRAG_GENERATED_QUALITY_PROXY_PLUS_EVIDENCE_ONLY",
        "CRAG_GENERATED_QUALITY_BLOCKED_SCHEMA_MAPPING",
        "CRAG_GENERATED_QUALITY_BLOCKED_NO_LABELS",
        "CRAG_GENERATED_QUALITY_BLOCKED_NO_USABLE_SIGNAL",
    }


def test_crag_no_success_claim_when_quality_signal_constant_zero() -> None:
    stats = load_json("artifacts/generative_llm_validation/crag/primary_outcome_statistics.json")
    if not stats.get("usable_quality_signal"):
        assert str(stats["result_class"]).endswith("_CRAG")
        assert "BLOCKED" in str(stats["result_class"]) or "INCONCLUSIVE" in str(stats["result_class"])


def test_raw_prompts_not_committed() -> None:
    for path in (ROOT / "artifacts/generative_llm_validation").rglob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            assert "prompt_text" not in (csv.DictReader(handle).fieldnames or [])


def test_raw_generated_answers_not_committed() -> None:
    for path in (ROOT / "artifacts/generative_llm_validation").rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "generated_answer_text" not in payload
        assert "raw_answer" not in payload
        assert "answer_text" not in payload


def test_no_crag_raw_text_committed() -> None:
    stats = load_json("artifacts/generative_llm_validation/crag/primary_outcome_statistics.json")
    assert stats["raw_questions_committed"] is False
    assert stats["raw_evidence_committed"] is False


def test_no_hotpotqa_raw_text_committed() -> None:
    manifest = load_json("artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json")
    assert manifest["raw_questions_committed"] is False
    assert manifest["raw_contexts_committed"] is False


def test_synthesis_downgrades_invalid_quality_signal() -> None:
    synthesis = load_json("results/generative_llm_validation/synthesis_result.json")
    crag = load_json("artifacts/generative_llm_validation/crag/primary_outcome_statistics.json")
    if not crag.get("usable_quality_signal"):
        assert synthesis["result_class"] != "GEN_LLM_SYNTHESIS_GENERATIVE_VALIDATION_SUPPORTED"


def test_no_official_platform_benchmark_claim_from_local_generator() -> None:
    docs = (ROOT / "docs/generative_llm_validation.md").read_text(encoding="utf-8").lower()
    assert "official platform benchmarking completed" not in docs


def test_no_human_validation_claim_without_annotations() -> None:
    docs = (ROOT / "docs/generative_llm_validation.md").read_text(encoding="utf-8").lower()
    assert "human validated" not in docs


def test_no_rag_compass_superiority_claim() -> None:
    docs = (ROOT / "docs/generative_llm_validation.md").read_text(encoding="utf-8").lower()
    assert "rag compass superiority" in docs
    assert "rag compass is superior" not in docs
