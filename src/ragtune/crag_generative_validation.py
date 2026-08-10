from __future__ import annotations

import bz2
import json
import os
from pathlib import Path
from typing import Any

from ragtune.fresh_live_behavioral_governance import CRAG_LIVE_POLICIES, inspect_crag_environment
from ragtune.generated_answer_quality import containment, exact_match, generated_quality_score, token_f1
from ragtune.generative_prompts import build_rag_prompt
from ragtune.generative_validation_common import GENERATION_FIELDNAMES, mean, write_csv, write_json, write_md, zero_ci
from ragtune.generators.factory import discover_generator
from ragtune.publication_sanitization import stable_hash


def crag_data_file() -> Path | None:
    data_root = os.environ.get("RAGTUNE_CRAG_DATA")
    if not data_root:
        return None
    path = Path(data_root) / "crag_task_1_and_2_dev_v5.jsonl.bz2"
    return path if path.exists() else None


def load_crag_rows(max_examples: int) -> list[dict[str, Any]]:
    path = crag_data_file()
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= max_examples:
                break
            rows.append(json.loads(line))
    return rows


def split_for_row(row: dict[str, Any]) -> str:
    bucket = int(stable_hash(str(row["interaction_id"]))[:8], 16) % 100
    if bucket < 50:
        return "calibration"
    if bucket < 75:
        return "validation"
    return "confirmatory_test"


def select_crag_evidence(row: dict[str, Any], policy_id: str) -> tuple[list[dict[str, str]], float, float]:
    results = list(row.get("search_results", []))
    if policy_id in {"low_retrieval_single_endpoint", "measured_cost_minimizer_at_quality_floor"}:
        selected = results[:1]
    elif policy_id in {"expanded_retrieval_multi_endpoint", "quality_only_best_on_validation"}:
        selected = results[:5]
    elif policy_id == "adaptive_routing_on_insufficient_evidence":
        selected = results[:3] if len(results) >= 3 else results
    elif policy_id == "measured_latency_minimizer_at_quality_floor":
        selected = results[:1]
    elif policy_id == "constrained_quality_optimizer":
        selected = results[:3]
    elif policy_id == "pareto_frontier_selector":
        selected = results[:2]
    elif policy_id == "governed_selection":
        selected = results[:2]
    elif policy_id == "rag_compass_optional":
        selected = sorted(results, key=lambda item: stable_hash(str(item.get("page_name", ""))))[:3]
    else:
        selected = results[:2]
    evidence = [
        {
            "evidence_id": f"crag_{idx}_{stable_hash(str(item.get('page_url', item.get('page_name', idx))))[:8]}",
            "text": f"{item.get('page_name', '')}\n{item.get('page_snippet', '')}\n{item.get('page_result', '')}"[:1600],
        }
        for idx, item in enumerate(selected)
    ]
    context_tokens = sum(len(item["text"].split()) for item in evidence)
    retrieval_cost = len(evidence) + context_tokens / 1000.0
    return evidence, retrieval_cost, context_tokens


def simple_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return zero_ci()
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "ci_low": ordered[int(0.025 * (len(ordered) - 1))],
        "ci_high": ordered[int(0.975 * (len(ordered) - 1))],
    }


def summarize_policy(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for policy in sorted({str(row["policy_id"]) for row in rows}):
        subset = [row for row in rows if row["policy_id"] == policy]
        summaries.append(
            {
                "policy_id": policy,
                "final_generated_quality_score": mean([float(row["final_generated_quality_score"]) for row in subset]),
                "total_cost_units": mean([float(row["total_cost_units"]) for row in subset]),
                "total_latency_ms": mean([float(row["total_latency_ms"]) for row in subset]),
            }
        )
    return summaries


def choose_winners(summaries: list[dict[str, object]]) -> tuple[str, str, list[str]]:
    if not summaries:
        return "", "", []
    quality_only = max(summaries, key=lambda row: float(row["final_generated_quality_score"]))
    quality_floor = float(quality_only["final_generated_quality_score"]) - 0.01
    feasible = [row for row in summaries if float(row["final_generated_quality_score"]) >= quality_floor]
    governed = min(feasible or summaries, key=lambda row: (float(row["total_cost_units"]), float(row["total_latency_ms"])))
    frontier = []
    for row in summaries:
        dominated = False
        for other in summaries:
            if other is row:
                continue
            better_or_equal = (
                float(other["final_generated_quality_score"]) >= float(row["final_generated_quality_score"])
                and float(other["total_cost_units"]) <= float(row["total_cost_units"])
                and float(other["total_latency_ms"]) <= float(row["total_latency_ms"])
            )
            strictly_better = (
                float(other["final_generated_quality_score"]) > float(row["final_generated_quality_score"])
                or float(other["total_cost_units"]) < float(row["total_cost_units"])
                or float(other["total_latency_ms"]) < float(row["total_latency_ms"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(row["policy_id"]))
    return str(governed["policy_id"]), str(quality_only["policy_id"]), frontier


def run_crag_generation(root: Path, output_root: Path, discovery) -> dict[str, object]:
    assert discovery.generator is not None
    max_examples = int(os.environ.get("RAGTUNE_CRAG_GEN_MAX_EXAMPLES", "4"))
    max_tokens = int(os.environ.get("RAGTUNE_GENERATOR_MAX_TOKENS", "48"))
    timeout_s = float(os.environ.get("RAGTUNE_GENERATOR_TIMEOUT_S", "120"))
    rows = load_crag_rows(max_examples)
    result_rows: list[dict[str, object]] = []
    for row in rows:
        split = split_for_row(row)
        references = [str(row.get("answer", ""))] + [str(value) for value in row.get("alt_ans", [])]
        references = [value for value in references if value]
        for policy_id in CRAG_LIVE_POLICIES:
            evidence_items, retrieval_cost, context_tokens = select_crag_evidence(row, policy_id)
            prompt, prompt_hash = build_rag_prompt(question_text=str(row["query"]), evidence_items=evidence_items)
            generation = discovery.generator.generate(
                prompt,
                model=discovery.model,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            raw_answer = (root / str(generation.raw_answer_local_path)).read_text(encoding="utf-8") if generation.raw_answer_local_path else ""
            answer_f1 = max([token_f1(raw_answer, ref) for ref in references], default=0.0)
            answer_em = max([exact_match(raw_answer, ref) for ref in references], default=0.0)
            answer_containment = max([containment(raw_answer, ref) for ref in references], default=0.0)
            evidence_text = " ".join(item["text"] for item in evidence_items)
            evidence_support = max([containment(evidence_text, ref) for ref in references], default=0.0)
            citation_support = 1.0 if evidence_items and evidence_support > 0.0 else 0.0
            abstained = "INSUFFICIENT_EVIDENCE" in raw_answer.upper()
            abstention_correctness = 1.0 if not references and abstained else (0.0 if abstained else 1.0)
            quality = generated_quality_score(
                answer_correctness_f1=answer_f1,
                answer_exact_match=answer_em,
                answer_containment=answer_containment,
                evidence_support_score=evidence_support,
                citation_support_score=citation_support,
                abstention_correctness=abstention_correctness,
            )
            result_rows.append(
                {
                    "example_id": stable_hash(str(row["interaction_id"])),
                    "question_hash": stable_hash(str(row["query"])),
                    "split": split,
                    "dataset": "crag",
                    "policy_id": policy_id,
                    "provider": generation.provider,
                    "model": generation.model,
                    "prompt_hash": prompt_hash,
                    "generated_answer_hash": generation.answer_hash,
                    "generated_answer_char_count": generation.answer_char_count,
                    "generated_answer_token_estimate": generation.answer_token_estimate,
                    "retrieval_latency_ms": 0.0,
                    "generation_latency_ms": generation.latency_ms,
                    "total_latency_ms": generation.latency_ms,
                    "retrieval_cost_units": retrieval_cost,
                    "generation_cost_units": generation.cost_units,
                    "total_cost_units": retrieval_cost + generation.cost_units,
                    "input_token_estimate": generation.input_token_estimate,
                    "output_token_estimate": generation.output_token_estimate,
                    "api_call_count": len(evidence_items),
                    "generator_call_count": 1,
                    "answer_correctness_f1": answer_f1,
                    "answer_exact_match": answer_em,
                    "answer_containment": answer_containment,
                    "evidence_support_score": evidence_support,
                    "citation_support_score": citation_support,
                    "abstention_correctness": abstention_correctness,
                    "final_generated_quality_score": quality,
                    "raw_prompt_exported": False,
                    "raw_generated_answer_exported": False,
                }
            )
    summaries = summarize_policy(result_rows)
    governed, quality_only, frontier = choose_winners(summaries)
    confirmatory = [row for row in result_rows if row["split"] == "confirmatory_test"]
    governed_rows = {row["example_id"]: row for row in confirmatory if row["policy_id"] == governed}
    quality_rows = {row["example_id"]: row for row in confirmatory if row["policy_id"] == quality_only}
    shared_ids = sorted(set(governed_rows) & set(quality_rows))
    quality_deltas = [float(governed_rows[idx]["final_generated_quality_score"]) - float(quality_rows[idx]["final_generated_quality_score"]) for idx in shared_ids]
    evidence_deltas = [float(governed_rows[idx]["evidence_support_score"]) - float(quality_rows[idx]["evidence_support_score"]) for idx in shared_ids]
    cost_deltas = [float(governed_rows[idx]["total_cost_units"]) - float(quality_rows[idx]["total_cost_units"]) for idx in shared_ids]
    latency_deltas = [float(governed_rows[idx]["total_latency_ms"]) - float(quality_rows[idx]["total_latency_ms"]) for idx in shared_ids]
    api_deltas = [float(governed_rows[idx]["api_call_count"]) - float(quality_rows[idx]["api_call_count"]) for idx in shared_ids]
    max_quality = max([float(row["final_generated_quality_score"]) for row in result_rows], default=0.0)
    if max_quality <= 0.0 or len({round(float(row["final_generated_quality_score"]), 12) for row in result_rows}) <= 1:
        result_class = "GEN_LLM_VALIDATION_BLOCKED_NO_USABLE_QUALITY_SIGNAL"
    elif quality_deltas and simple_ci(quality_deltas)["mean"] >= -0.01 and simple_ci(cost_deltas)["ci_high"] < 0:
        result_class = "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY"
    elif quality_deltas and simple_ci(quality_deltas)["mean"] < -0.01 and simple_ci(cost_deltas)["mean"] < 0:
        result_class = "GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS"
    else:
        result_class = "GEN_LLM_GOVERNANCE_INCONCLUSIVE"
    stats = {
        "suite": "ragtune_crag_generative_llm_validation_v1",
        "evidence_class": "crag_generative_validation_sanitized_bounded_sample",
        "result_class": result_class,
        "generator_provider": discovery.provider,
        "generator_model": discovery.model,
        "generator_available": True,
        "generator_local_or_hosted": discovery.local_or_hosted,
        "quality_metric_class": "GENERATED_QUALITY_CRAG_LOCAL_EVALUATOR",
        "governed_winner": governed,
        "quality_only_winner": quality_only,
        "constrained_optimizer_winner": governed,
        "pareto_frontier": frontier,
        "rag_compass_rank": next((idx + 1 for idx, row in enumerate(sorted(summaries, key=lambda item: float(item["final_generated_quality_score"]), reverse=True)) if row["policy_id"] == "rag_compass_optional"), ""),
        "generated_quality_delta": simple_ci(quality_deltas),
        "evidence_support_delta": simple_ci(evidence_deltas),
        "cost_delta": simple_ci(cost_deltas),
        "latency_delta_ms": simple_ci(latency_deltas),
        "api_call_delta": simple_ci(api_deltas),
        "generation_rows": len(result_rows),
        "example_count": len(rows),
        "raw_prompts_committed": False,
        "raw_generated_answers_committed": False,
        "raw_questions_committed": False,
        "raw_evidence_committed": False,
        "secrets_committed": False,
    }
    write_csv(output_root / "per_query_generation_metrics.csv", GENERATION_FIELDNAMES, result_rows)
    write_csv(output_root / "policy_summary_metrics.csv", ["policy_id", "final_generated_quality_score", "total_cost_units", "total_latency_ms"], summaries)
    write_csv(output_root / "selector_comparison.csv", ["selector", "winner", "reason"], [
        {"selector": "governed_selection", "winner": governed, "reason": "lowest cost within generated-quality noninferiority margin"},
        {"selector": "quality_only_best_on_validation", "winner": quality_only, "reason": "highest generated quality"},
    ])
    write_csv(output_root / "pareto_frontier.csv", ["policy_id", "frontier_reason"], [{"policy_id": policy, "frontier_reason": "nondominated on generated quality, cost, and latency"} for policy in frontier])
    return stats


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
        stats = run_crag_generation(root, output_root, discovery)
        manifest = {
            "suite": "ragtune_crag_generative_llm_validation_v1",
            "evidence_class": stats["evidence_class"],
            "result_class": stats["result_class"],
            "blocker": "",
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
        write_json(output_root / "generative_crag_manifest.json", manifest)
        write_json(output_root / "primary_outcome_statistics.json", stats)
        write_md(
            output_root / "generator_environment_report.md",
            f"""
# CRAG Generative LLM Environment

Provider: `{discovery.provider}`
Model: `{discovery.model}`
Status: `{stats['result_class']}`

Raw prompts, generated answers, CRAG questions, CRAG evidence, and CRAG API responses are not committed.
""",
        )
        write_md(
            output_root / "primary_outcome_report.md",
            f"""
# CRAG Generative LLM Validation

Result class: `{stats['result_class']}`

This bounded local-generator run used CRAG answers and alternate answers locally for scoring. Public artifacts contain only hashes, counts, and metrics; raw prompts, raw questions, raw evidence, raw API responses, and raw generated answers are excluded from Git.
""",
        )
        return stats

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
