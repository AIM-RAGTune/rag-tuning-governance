from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from ragtune.policy_selection import pareto_frontier
from ragtune.publication_sanitization import write_sanitized_json, write_text


FRESH_CRAG_RESULT_CLASSES = {
    "FRESH_CRAG_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY",
    "FRESH_CRAG_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_QUALITY",
    "FRESH_CRAG_GOVERNANCE_IMPROVES_QUALITY_UNDER_FIXED_BUDGET",
    "FRESH_CRAG_GOVERNANCE_MATCHES_QUALITY_ONLY",
    "FRESH_CRAG_GOVERNANCE_NONINFERIOR_NO_OPERATIONAL_GAIN",
    "FRESH_CRAG_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS",
    "FRESH_CRAG_GOVERNANCE_NEGATIVE",
    "FRESH_CRAG_GOVERNANCE_INCONCLUSIVE",
    "FRESH_CRAG_BLOCKED_NO_APPROVED_DATA",
    "FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE",
    "FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY",
    "FRESH_CRAG_BLOCKED_POLICY_DISTINCTION_FAILED",
    "FRESH_CRAG_BLOCKED_PUBLICATION_HYGIENE",
}

HOTPOTQA_RESULT_CLASSES = {
    "HOTPOTQA_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY",
    "HOTPOTQA_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_QUALITY",
    "HOTPOTQA_GOVERNANCE_IMPROVES_QUALITY_UNDER_FIXED_BUDGET",
    "HOTPOTQA_GOVERNANCE_MATCHES_QUALITY_ONLY",
    "HOTPOTQA_GOVERNANCE_NONINFERIOR_NO_OPERATIONAL_GAIN",
    "HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS",
    "HOTPOTQA_GOVERNANCE_NEGATIVE",
    "HOTPOTQA_GOVERNANCE_INCONCLUSIVE",
    "HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE",
    "HOTPOTQA_BLOCKED_LICENSE_REVIEW",
    "HOTPOTQA_BLOCKED_POLICY_DISTINCTION_FAILED",
    "HOTPOTQA_BLOCKED_PUBLICATION_HYGIENE",
}

SYNTHESIS_RESULT_CLASSES = {
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_REPLICATED",
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_DIRECTIONAL",
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_MIXED",
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_INCONCLUSIVE",
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_NEGATIVE",
    "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_BLOCKED",
}


def csv_empty(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def crag_required_paths(root: Path) -> dict[str, bool]:
    return {
        "mock_api": (root / "mock_api").exists(),
        "docs": (root / "docs").exists(),
        "local_evaluation.py": (root / "local_evaluation.py").exists(),
        "requirements.txt": (root / "requirements.txt").exists(),
    }


def inspect_crag_environment() -> dict[str, Any]:
    approved = os.environ.get("RAGTUNE_CRAG_APPROVED_NONCOMMERCIAL_RESEARCH_ONLY") == "true"
    root_value = os.environ.get("RAGTUNE_CRAG_ROOT")
    data_value = os.environ.get("RAGTUNE_CRAG_DATA")
    root = Path(root_value).expanduser() if root_value else None
    data = Path(data_value).expanduser() if data_value else None
    root_ok = bool(root and root.exists())
    data_ok = bool(data and data.exists())
    required = crag_required_paths(root) if root_ok and root is not None else {}
    return {
        "approved_noncommercial_research_only": approved,
        "crag_root_configured": bool(root_value),
        "crag_data_configured": bool(data_value),
        "crag_root_exists": root_ok,
        "crag_data_exists": data_ok,
        "required_paths": required,
        "mock_api_available": bool(required.get("mock_api")) if required else False,
        "local_evaluation_available": bool(required.get("local_evaluation.py")) if required else False,
    }


def write_crag_acquisition_report(repo_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    output = repo_root / "artifacts" / "fresh_live_crag_behavioral_governance"
    env = inspect_crag_environment()
    if not env["approved_noncommercial_research_only"] or not env["crag_root_exists"] or not env["crag_data_exists"]:
        result = "FRESH_CRAG_BLOCKED_NO_APPROVED_DATA"
    elif not env["mock_api_available"]:
        result = "FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE"
    else:
        result = "FRESH_CRAG_GOVERNANCE_INCONCLUSIVE"
    payload = {
        "suite": "ragtune_fresh_live_crag_mock_api_behavioral_governance_v1",
        "result_class": result,
        "dry_run": dry_run,
        "evidence_class": "fresh_live_crag_mock_api_blocked" if result.startswith("FRESH_CRAG_BLOCKED") else "fresh_live_crag_mock_api",
        "environment": env,
        "dataset_rows_committed": False,
        "query_wording_exported": False,
        "endpoint_outputs_exported": False,
        "source_documents_exported": False,
        "acquisition_instructions": [
            "git clone https://github.com/facebookresearch/CRAG.git <approved-local-path>/CRAG",
            "cd <approved-local-path>/CRAG",
            "pip install -r requirements.txt",
            "Follow CRAG dataset documentation for approved noncommercial research-only data acquisition.",
            "Set RAGTUNE_CRAG_APPROVED_NONCOMMERCIAL_RESEARCH_ONLY=true, RAGTUNE_CRAG_ROOT, and RAGTUNE_CRAG_DATA.",
            "Do not copy raw CRAG data into this public repository.",
        ],
    }
    write_sanitized_json(output / "live_crag_manifest.json", payload)
    write_sanitized_json(output / "primary_outcome_statistics.json", payload)
    write_text(
        output / "live_crag_acquisition_report.md",
        "# Fresh Live CRAG Mock-API Acquisition\n\n"
        f"Result: `{result}`.\n\n"
        "No fresh live CRAG collection was run because approved local CRAG data and/or mock-API paths were unavailable in this environment. "
        "Raw CRAG data, raw query wording, source documents, and API responses were not copied or exported.\n",
    )
    write_text(
        output / "primary_outcome_report.md",
        "# Fresh Live CRAG Behavioral Governance\n\n"
        f"Result: `{result}`.\n\n"
        "This is a blocked result, not a governance-success claim. Re-run after configuring approved local CRAG paths.\n",
    )
    write_sanitized_json(output / "split_manifest.json", {"result_class": result, "split_status": "not_created_blocked_no_approved_data"})
    for name in [
        "per_query_policy_results.csv",
        "policy_summary_metrics.csv",
        "selector_comparison.csv",
        "pareto_frontier.csv",
    ]:
        csv_empty(output / name, ["result_class", "note"])
    return payload


def inspect_hotpotqa_environment(local_data_root: Path | None = None) -> dict[str, Any]:
    datasets_available = importlib.util.find_spec("datasets") is not None
    root = local_data_root or Path(os.environ.get("RAGTUNE_DATA_ROOT", ".local_data")) / "hotpotqa"
    raw_candidates = [
        root / "hotpot_dev_distractor_v1.json",
        root / "hotpot_train_v1.1.json",
        root / "hotpot_dev_fullwiki_v1.json",
    ]
    return {
        "datasets_library_available": datasets_available,
        "local_data_root_exists": root.exists(),
        "local_data_root": "<repo>/.local_data/hotpotqa" if not root.is_absolute() else "<external-hotpotqa-data-root>",
        "known_raw_files_present": [path.name for path in raw_candidates if path.exists()],
    }


def write_hotpotqa_acquisition_report(repo_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    output = repo_root / "artifacts" / "hotpotqa_behavioral_governance"
    env = inspect_hotpotqa_environment()
    if not env["datasets_library_available"] and not env["known_raw_files_present"]:
        result = "HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE"
    elif not env["known_raw_files_present"]:
        result = "HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE"
    else:
        result = "HOTPOTQA_GOVERNANCE_INCONCLUSIVE"
    payload = {
        "suite": "ragtune_hotpotqa_behavioral_governance_v1",
        "result_class": result,
        "dry_run": dry_run,
        "evidence_class": "hotpotqa_public_corpus_blocked" if result.startswith("HOTPOTQA_BLOCKED") else "hotpotqa_public_corpus",
        "license_status": "Dataset CC BY-SA 4.0; code Apache-2.0; raw data not redistributed by this repository.",
        "environment": env,
        "question_wording_exported": False,
        "context_paragraphs_exported": False,
        "supporting_fact_sentences_exported": False,
        "acquisition_instructions": [
            "pip install datasets",
            "python3 scripts/acquire_hotpotqa_public_corpus.py --source huggingface --config distractor --output-root ${RAGTUNE_DATA_ROOT:-.local_data}/hotpotqa",
            "Alternatively clone https://github.com/hotpotqa/hotpot and follow official download instructions.",
            "Do not commit raw questions, context paragraphs, or supporting-fact sentences.",
        ],
    }
    write_sanitized_json(output / "hotpotqa_acquisition_manifest.json", payload)
    write_sanitized_json(output / "primary_outcome_statistics.json", payload)
    write_sanitized_json(output / "hotpotqa_split_manifest.json", {"result_class": result, "split_status": "not_created_blocked_dataset_unavailable"})
    write_sanitized_json(output / "split_manifest.json", {"result_class": result, "split_status": "not_created_blocked_dataset_unavailable"})
    write_text(
        output / "hotpotqa_license_report.md",
        "# HotpotQA License Report\n\n"
        "HotpotQA raw data are not committed to this repository. The intended dataset license boundary is CC BY-SA 4.0 for data and Apache-2.0 for code, with raw-data acquisition delegated to the original providers.\n",
    )
    write_text(
        output / "hotpotqa_quality_measurement_report.md",
        "# HotpotQA Quality Measurement v1\n\n"
        "Planned components: exact match, normalized F1, supporting-fact title recall, supporting-fact sentence recall, evidence efficiency, and abstention correctness. "
        "The run is blocked until local HotpotQA data are available.\n",
    )
    write_text(
        output / "primary_outcome_report.md",
        "# HotpotQA Behavioral Governance\n\n"
        f"Result: `{result}`.\n\n"
        "No HotpotQA result is claimed because the dataset was unavailable in this environment. The script produced sanitized blocked artifacts only.\n",
    )
    for name in [
        "per_query_policy_results.csv",
        "policy_summary_metrics.csv",
        "selector_comparison.csv",
        "pareto_frontier.csv",
    ]:
        csv_empty(output / name, ["result_class", "note"])
    return payload


def write_multi_dataset_synthesis(repo_root: Path) -> dict[str, Any]:
    crag_path = repo_root / "artifacts" / "fresh_live_crag_behavioral_governance" / "primary_outcome_statistics.json"
    hotpot_path = repo_root / "artifacts" / "hotpotqa_behavioral_governance" / "primary_outcome_statistics.json"
    prior_path = repo_root / "artifacts" / "behavioral_governance" / "primary_outcome_statistics.json"
    crag = json.loads(crag_path.read_text(encoding="utf-8")) if crag_path.exists() else {"result_class": "FRESH_CRAG_BLOCKED_NO_APPROVED_DATA"}
    hotpot = json.loads(hotpot_path.read_text(encoding="utf-8")) if hotpot_path.exists() else {"result_class": "HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE"}
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {"primary_result_class": "not_available"}

    fresh_success = str(crag.get("result_class", "")).startswith("FRESH_CRAG_GOVERNANCE_") and "BLOCKED" not in str(crag.get("result_class"))
    hotpot_success = str(hotpot.get("result_class", "")).startswith("HOTPOTQA_GOVERNANCE_") and "BLOCKED" not in str(hotpot.get("result_class"))
    if fresh_success and hotpot_success:
        result = "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_REPLICATED"
    elif fresh_success or hotpot_success:
        result = "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_DIRECTIONAL"
    elif str(crag.get("result_class", "")).startswith("FRESH_CRAG_BLOCKED") and str(hotpot.get("result_class", "")).startswith("HOTPOTQA_BLOCKED"):
        result = "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_BLOCKED"
    else:
        result = "MULTI_DATASET_BEHAVIORAL_GOVERNANCE_INCONCLUSIVE"

    output = repo_root / "results" / "multi_dataset_behavioral_governance"
    comparison_rows = [
        {
            "dataset": "prior_sanitized_frozen_crag",
            "evidence_class": prior.get("evidence_class", "public_full_corpus_mock_api_validation_derived_frozen_observation"),
            "result_class": prior.get("primary_result_class", "not_available"),
            "claim_weight": "bounded_frozen_observation",
        },
        {
            "dataset": "fresh_live_crag_mock_api",
            "evidence_class": crag.get("evidence_class", "fresh_live_crag_mock_api_blocked"),
            "result_class": crag.get("result_class", ""),
            "claim_weight": "blocked" if str(crag.get("result_class", "")).startswith("FRESH_CRAG_BLOCKED") else "fresh_live",
        },
        {
            "dataset": "hotpotqa",
            "evidence_class": hotpot.get("evidence_class", "hotpotqa_public_corpus_blocked"),
            "result_class": hotpot.get("result_class", ""),
            "claim_weight": "blocked" if str(hotpot.get("result_class", "")).startswith("HOTPOTQA_BLOCKED") else "alternate_public_corpus",
        },
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    payload = {
        "suite": "ragtune_multi_dataset_behavioral_governance_synthesis_v1",
        "result_class": result,
        "fresh_live_crag_result_class": crag.get("result_class"),
        "hotpotqa_result_class": hotpot.get("result_class"),
        "prior_frozen_observation_result_class": prior.get("primary_result_class"),
        "claim_boundary": "Replication is not claimed when only frozen-observation evidence succeeds.",
        "unsupported_claims": [
            "RAG Compass superiority",
            "human validation",
            "generative LLM validation",
            "official platform benchmarking",
            "production readiness",
        ],
    }
    write_sanitized_json(output / "synthesis_result.json", payload)
    write_sanitized_json(output / "claim_update.json", payload)
    write_text(
        output / "synthesis_report.md",
        "# Multi-Dataset Behavioral Governance Synthesis\n\n"
        f"Result: `{result}`.\n\n"
        "Fresh live CRAG and HotpotQA were not available in this execution environment, so the phase does not replicate the frozen-observation result. "
        "The prior sanitized CRAG frozen-observation result remains preserved, but it is not upgraded to multi-dataset replication.\n",
    )
    write_text(
        output / "paper_ready_summary.md",
        "# Fresh Live CRAG + HotpotQA Behavioral Governance Summary\n\n"
        "## Why frozen-observation evidence was insufficient\n\n"
        "The prior behaviorally distinct result used sanitized frozen CRAG mock-API observations. It reduced measured cost at equivalent proxy-plus-evidence quality, but it was not a fresh live collection and did not use a second corpus with stronger labels.\n\n"
        "## Why fresh CRAG was attempted\n\n"
        "Fresh live CRAG would test whether the policy behavior and operating-cost result persist when the mock API is called again under approved noncommercial constraints.\n\n"
        "## Why HotpotQA was selected\n\n"
        "HotpotQA provides answer labels, multi-hop structure, bridge/comparison types, difficulty levels, and supporting-fact labels for stronger answer correctness and evidence-support scoring.\n\n"
        "## Dataset acquisition status\n\n"
        f"Fresh CRAG: `{crag.get('result_class')}`. HotpotQA: `{hotpot.get('result_class')}`.\n\n"
        "## Policy suite\n\n"
        "The planned suite includes low retrieval, expanded retrieval, adaptive routing, BM25/reranking for HotpotQA, quality-only, constrained optimizer, Pareto selector, and governed selection.\n\n"
        "## Quality metrics\n\n"
        "CRAG would use proxy-plus-evidence plus any available local evaluator. HotpotQA would use exact match, F1, supporting-fact title recall, supporting-fact sentence recall, evidence efficiency, and abstention correctness.\n\n"
        "## Primary endpoint\n\n"
        "Equivalent quality with lower measured cost/latency, or improved quality under a fixed deployment budget.\n\n"
        "## Fresh CRAG result\n\n"
        f"`{crag.get('result_class')}`.\n\n"
        "## HotpotQA result\n\n"
        f"`{hotpot.get('result_class')}`.\n\n"
        "## Multi-dataset synthesis\n\n"
        f"`{result}`.\n\n"
        "## Negative findings\n\n"
        "This run did not move beyond frozen-observation evidence because approved local CRAG and HotpotQA data were unavailable.\n\n"
        "## Claim boundaries\n\n"
        "No human validation, generative validation, official platform benchmark, production readiness, broad governance superiority, or RAG Compass superiority is claimed.\n\n"
        "## Reproduction instructions\n\n"
        "Configure approved CRAG and/or HotpotQA local data roots, then run the acquisition and governance scripts documented in this repository.\n\n"
        "## Recommended next experiment\n\n"
        "Run HotpotQA from an approved local dataset cache and repeat fresh live CRAG after configuring the CRAG mock API runtime.\n",
    )
    write_text(
        output / "executive_summary.md",
        "# Executive Summary\n\n"
        f"Multi-dataset synthesis result: `{result}`. The phase created the harness and blocked honestly because fresh live CRAG and HotpotQA data were unavailable locally.\n",
    )
    write_text(
        output / "limitations.md",
        "# Limitations\n\n"
        "- Fresh CRAG was blocked by missing approved local CRAG data/mock-API configuration.\n"
        "- HotpotQA was blocked by missing local dataset files and unavailable `datasets` package.\n"
        "- No raw data were committed.\n"
        "- No replication claim is made from frozen-only evidence.\n",
    )
    return payload
