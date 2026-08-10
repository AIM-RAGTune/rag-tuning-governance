from __future__ import annotations

from pathlib import Path

from ragtune.fresh_live_behavioral_governance import inspect_crag_environment
from ragtune.generative_validation_common import GENERATION_FIELDNAMES, write_csv, write_json, write_md, zero_ci
from ragtune.generators.factory import discover_generator


def run_crag_generative_validation(root: Path, *, output_root: Path, dry_run: bool = False) -> dict[str, object]:
    env = inspect_crag_environment()
    discovery = discover_generator(dry_run=dry_run)
    if not env["approved_noncommercial_research_only"] or not env["crag_data_exists"] or not env["mock_api_available"]:
        result_class = "GEN_LLM_VALIDATION_BLOCKED_CRAG_UNAVAILABLE"
        blocker = "approved CRAG data/mock API environment is not fully available"
    elif not discovery.available:
        result_class = discovery.status
        blocker = discovery.instructions
    else:
        result_class = "GEN_LLM_VALIDATION_BLOCKED_EVALUATOR_MAPPING"
        blocker = "CRAG generative evaluator mapping is not implemented for public sanitized export"

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": "ragtune_crag_generative_llm_validation_v1",
        "evidence_class": "crag_generative_validation_attempt",
        "result_class": result_class,
        "blocker": blocker,
        "generator_provider": discovery.provider,
        "generator_model": discovery.model,
        "generator_available": discovery.available,
        "generator_local_or_hosted": discovery.local_or_hosted,
        "crag_approval_env_var_present": bool(env["approved_noncommercial_research_only"]),
        "crag_root_configured": bool(env["crag_root_configured"]),
        "crag_data_configured": bool(env["crag_data_configured"]),
        "mock_api_available": bool(env["mock_api_available"]),
        "local_evaluator_available": bool(env["local_evaluation_available"]),
        "raw_prompts_committed": False,
        "raw_generated_answers_committed": False,
        "raw_questions_committed": False,
        "raw_evidence_committed": False,
        "secrets_committed": False,
    }
    stats = {
        **manifest,
        "quality_metric_class": "GENERATED_QUALITY_BLOCKED_NO_SIGNAL",
        "governed_winner": "",
        "quality_only_winner": "",
        "constrained_optimizer_winner": "",
        "pareto_frontier": [],
        "rag_compass_rank": "",
        "generated_quality_delta": zero_ci(),
        "evidence_support_delta": zero_ci(),
        "cost_delta": zero_ci(),
        "latency_delta_ms": zero_ci(),
        "api_call_delta": zero_ci(),
        "generation_rows": 0,
    }
    write_json(output_root / "generative_crag_manifest.json", manifest)
    write_json(output_root / "primary_outcome_statistics.json", stats)
    write_csv(output_root / "per_query_generation_metrics.csv", GENERATION_FIELDNAMES, [])
    write_csv(output_root / "policy_summary_metrics.csv", ["policy_id", "final_generated_quality_score", "total_cost_units", "total_latency_ms"], [])
    write_csv(output_root / "selector_comparison.csv", ["selector", "winner", "reason"], [])
    write_csv(output_root / "pareto_frontier.csv", ["policy_id", "frontier_reason"], [])
    write_md(
        output_root / "generator_environment_report.md",
        f"""
# CRAG Generative LLM Environment

Provider: `{discovery.provider}`
Model: `{discovery.model or 'not configured'}`
Status: `{result_class}`

Raw prompts, generated answers, CRAG questions, CRAG evidence, and CRAG API responses are not committed.

Blocker: {blocker}
""",
    )
    write_md(
        output_root / "primary_outcome_report.md",
        f"""
# CRAG Generative LLM Validation

Result class: `{result_class}`

This run did not produce a generative governance claim. Public artifacts contain only sanitized status, hashes, counts, and metric fields. Raw prompts and raw generated answers are excluded.
""",
    )
    return stats
