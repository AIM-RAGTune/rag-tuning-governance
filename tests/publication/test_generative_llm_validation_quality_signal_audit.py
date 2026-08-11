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
    assert 'model.lower().startswith(("qwen3", "gpt-oss"))' in source
    assert 'payload["think"] = think' in source


def test_gpt_oss_ollama_uses_chat_endpoint_by_default() -> None:
    source = (ROOT / "src/ragtune/generators/ollama.py").read_text(encoding="utf-8")
    assert "RAGTUNE_OLLAMA_ENDPOINT" in source
    assert 'model.lower().startswith("gpt-oss")' in source
    assert "api/{'chat' if use_chat else 'generate'}" in source


def test_crag_prompt_forbids_blank_answers() -> None:
    source = (ROOT / "src/ragtune/generative_prompts.py").read_text(encoding="utf-8")
    assert "Never return a blank response" in source
    assert "build_answer_emission_repair_prompt" in source


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


def test_crag_repeat_comparison_result_class_machine_readable() -> None:
    comparison = load_json("results/generative_llm_validation/crag_repeat_comparison.json")
    assert comparison["result_class"] in {
        "CRAG_GEN_LLM_COST_RESULT_PERSISTED_IN_INDEPENDENT_REPEAT",
        "CRAG_GEN_LLM_COST_RESULT_NOT_REPLICATED",
        "CRAG_GEN_LLM_COST_RESULT_DIRECTIONAL_REPEAT_ONLY",
        "CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE",
    }
    assert comparison["raw_prompts_committed"] is False
    assert comparison["raw_generated_answers_committed"] is False


def test_crag_stability_comparison_result_class_machine_readable() -> None:
    comparison = load_json("results/generative_llm_validation/crag_stability_comparison.json")
    assert comparison["result_class"] in {
        "CRAG_GEN_LLM_STABILITY_BLOCKED_NO_RUNS",
        "CRAG_GEN_LLM_STABILITY_BLOCKED_NO_USABLE_QUALITY_SIGNAL",
        "CRAG_GEN_LLM_COST_RESULT_STABLE_ACROSS_REPEATS",
        "CRAG_GEN_LLM_COST_RESULT_MIXED_ACROSS_REPEATS",
        "CRAG_GEN_LLM_COST_RESULT_NOT_STABLE_ACROSS_REPEATS",
        "CRAG_GEN_LLM_COST_RESULT_DIRECTIONAL_BUT_UNSTABLE",
        "CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS",
        "CRAG_GEN_LLM_LATENCY_RESULT_STABLE_ACROSS_REPEATS",
        "CRAG_GEN_LLM_LATENCY_RESULT_MIXED_ACROSS_REPEATS",
        "CRAG_GEN_LLM_LATENCY_RESULT_NOT_STABLE_ACROSS_REPEATS",
        "CRAG_GEN_LLM_LATENCY_RESULT_DIRECTIONAL_BUT_UNSTABLE",
        "CRAG_GEN_LLM_LATENCY_RESULT_INCONCLUSIVE_ACROSS_REPEATS",
    }
    assert int(comparison["run_count"]) >= 3
    assert comparison["raw_prompts_committed"] is False
    assert comparison["raw_generated_answers_committed"] is False


def test_crag_second_model_comparison_result_class_machine_readable() -> None:
    comparison = load_json("results/generative_llm_validation/crag_second_model_comparison.json")
    assert comparison["result_class"] in {
        "CRAG_GEN_LLM_COST_RESULT_STABLE_ACROSS_MODELS",
        "CRAG_GEN_LLM_COST_RESULT_NOT_STABLE_ACROSS_MODELS",
        "CRAG_GEN_LLM_COST_RESULT_SECOND_MODEL_ONLY",
        "CRAG_GEN_LLM_COST_RESULT_PRIMARY_MODEL_ONLY",
        "CRAG_GEN_LLM_COST_RESULT_MIXED_OR_INCONCLUSIVE_ACROSS_MODELS",
    }
    assert comparison["raw_prompts_committed"] is False
    assert comparison["raw_generated_answers_committed"] is False


def test_crag_answer_emission_model_comparison_machine_readable() -> None:
    comparison = load_json("results/generative_llm_validation/crag_answer_emission_model_comparison.json")
    assert comparison["result_class"] in {
        "CRAG_GEN_LLM_ANSWER_EMISSION_REPAIRED_NO_COST_RESULT",
        "CRAG_GEN_LLM_ANSWER_EMISSION_REPAIRED_WITH_COST_SIGNAL",
        "CRAG_GEN_LLM_ANSWER_EMISSION_NOT_REPAIRED",
    }
    assert comparison["materially_reduced_parse_failures"] is True
    assert comparison["raw_prompts_committed"] is False
    assert comparison["raw_generated_answers_committed"] is False


def test_llama3_crag_offsets_have_no_parse_failures() -> None:
    for offset in ["0", "24", "36", "60"]:
        stats = load_json(f"artifacts/generative_llm_validation/crag_llama3_2_3b_offset_{offset}/primary_outcome_statistics.json")
        assert stats["generator_model"] == "llama3.2:3b"
        assert stats["non_empty_generated_answers"] == stats["generation_rows"]
        assert stats["usable_quality_signal"] is True


def test_llama3_latency_selector_offsets_separate_winners() -> None:
    positive_latency = 0
    quality_loss = 0
    for offset in ["0", "24", "36", "60"]:
        stats = load_json(f"artifacts/generative_llm_validation/crag_llama3_2_3b_latency_selector_offset_{offset}/primary_outcome_statistics.json")
        assert stats["generator_model"] == "llama3.2:3b"
        assert stats["primary_endpoint"] == "latency"
        assert stats["selector_design"] == "validation_split_quality_only_high_evidence_vs_governed_latency_feasible_confirmatory_eval"
        assert stats["governed_winner"] != stats["quality_only_winner"]
        assert stats["non_empty_generated_answers"] == stats["generation_rows"]
        assert stats["usable_quality_signal"] is True
        assert stats["raw_generated_answers_committed"] is False
        if stats["result_class"] == "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY_CRAG":
            positive_latency += 1
        if stats["result_class"] == "GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG":
            quality_loss += 1
    assert positive_latency == 1
    assert quality_loss == 3


def test_llama3_guarded_latency_offsets_are_quality_protective() -> None:
    for offset in ["0", "24", "36", "60"]:
        stats = load_json(f"artifacts/generative_llm_validation/crag_llama3_2_3b_guarded_latency_offset_{offset}/primary_outcome_statistics.json")
        assert stats["generator_model"] == "llama3.2:3b"
        assert stats["primary_endpoint"] == "latency"
        assert stats["quality_risk_guardrail_enabled"] is True
        assert stats["governed_winner"] == "quality_guarded_latency_adaptive_expansion"
        assert stats["quality_only_winner"] == "expanded_retrieval_multi_endpoint"
        assert stats["non_empty_generated_answers"] == stats["generation_rows"]
        assert stats["usable_quality_signal"] is True
        assert stats["result_class"] == "GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG"
        assert stats["raw_generated_answers_committed"] is False
    comparison = load_json("results/generative_llm_validation/crag_stability_comparison.json")
    assert comparison["result_class"] == "CRAG_GEN_LLM_LATENCY_RESULT_INCONCLUSIVE_ACROSS_REPEATS"
    assert comparison["positive_latency_result_count"] == 0


def test_crag_generative_selector_uses_deployable_validation_candidates() -> None:
    source = (ROOT / "src/ragtune/crag_generative_validation.py").read_text(encoding="utf-8")
    assert "DEPLOYABLE_CRAG_GENERATIVE_POLICIES" in source
    assert "QUALITY_ONLY_CRAG_GENERATIVE_POLICIES" in source
    assert "validation_split_quality_only_high_evidence_vs_governed_latency_feasible_confirmatory_eval" in source
    assert "RAGTUNE_CRAG_GEN_PRIMARY_ENDPOINT" in source


def test_crag_generative_latency_guardrail_is_predeclared() -> None:
    source = (ROOT / "src/ragtune/crag_generative_validation.py").read_text(encoding="utf-8")
    assert "QUALITY_GUARDED_LATENCY_POLICY" in source
    assert "quality_guarded_latency_adaptive_expansion" in source
    assert "RAGTUNE_CRAG_GEN_LATENCY_GUARDRAIL" in source
    assert "validation_split_quality_only_high_evidence_vs_quality_guarded_latency_confirmatory_eval" in source
    assert "expand to five" in source


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
