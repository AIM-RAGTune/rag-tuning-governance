#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = Path(os.environ["EXPORT_ROOT"]).resolve() if os.environ.get("EXPORT_ROOT") else ROOT.parent

REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "docs/claim_boundaries.md",
    "data/DATA_AVAILABILITY.md",
    "results/run_index.csv",
    "results/evidence_summary.json",
    "results/claim_status/claim_status_table.csv",
    ".gitattributes",
    "artifacts/fresh_live_crag_behavioral_governance/live_crag_manifest.json",
    "artifacts/hotpotqa_behavioral_governance/hotpotqa_acquisition_manifest.json",
    "results/multi_dataset_behavioral_governance/synthesis_result.json",
    "docs/fresh_live_crag_hotpotqa_behavioral_governance_plan.md",
    "docs/dataset_acquisition.md",
    "artifacts/generative_llm_validation/crag/generative_crag_manifest.json",
    "artifacts/generative_llm_validation/hotpotqa/generative_hotpotqa_manifest.json",
    "results/generative_llm_validation/synthesis_result.json",
    "docs/generative_llm_validation.md",
    "docs/generator_configuration.md",
    "docs/platform_benchmarking_boundary.md",
    "configs/experiments/ragtune_hotpotqa_generative_quality_signal_audit_v1.yaml",
    "configs/experiments/ragtune_crag_generative_quality_risk_guardrail_v2.yaml",
    "artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json",
    "artifacts/generative_llm_validation/crag_quality_risk_guardrail_v2/audit_manifest.json",
    "results/generative_llm_validation/crag_quality_risk_guardrail_v2_comparison.json",
    "deployment_review/generative_llm_validation_quality_signal_audit/generator_access_diagnosis.json",
]

EXPORT_REQUIRED = [
    "approval_package/approval_request_summary.md",
    "approval_package/publication_safety_audit.md",
    "approval_package/publication_safety_audit.json",
    "approval_package/data_license_audit.md",
    "approval_package/upload_blocker_record.md",
    "approval_package/github_upload_commands_NOT_RUN.md",
    "validation_reports/secret_scan_report.md",
    "validation_reports/secret_scan_report.json",
]

SECRET_PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"\b(?:ghp|github_pat|gho)_[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{30,}", re.I),
}

FORBIDDEN_TRACKED_PARTS = {
    ".env",
    "data/raw",
    "results/raw/crag",
    "artifacts/raw",
    "human_eval_answer_key_private.json",
}

MAX_FILE_BYTES = 50 * 1024 * 1024

CRAG_ARTIFACT_ROOT = Path("artifacts/selected_run_summaries")
RAW_TEXT_JSON_KEYS = {
    "query_text",
    "question_text",
    "raw_query",
    "raw_question",
    "source_snippet",
    "raw_response",
    "api_response",
    "document_text",
    "context_text",
    "snippet",
    "generated_answer",
    "generated_answer_text",
    "answer_text",
    "raw_answer",
    "prompt_text",
}
GENERATIVE_ARTIFACT_ROOT = Path("artifacts/generative_llm_validation")


def tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    except Exception:
        excluded_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return [p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file() and not (excluded_parts & set(p.parts))]
    return [Path(line) for line in out.splitlines() if line.strip()]


def fail(message: str) -> None:
    print(f"publication validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def approved_deployment_remote() -> str | None:
    report_path = ROOT / "deployment_review" / "reports" / "github_deployment_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if report.get("deployment_status") != "DEPLOYED":
        return None
    if report.get("environment_allows_github") != "PASS_AFTER_EXPLICIT_USER_APPROVAL":
        return None
    return report.get("remote_url")


def publication_remote_mode() -> str:
    explicit = os.environ.get("RAGTUNE_PUBLICATION_REMOTE_MODE")
    if explicit:
        return explicit
    if os.environ.get("GITHUB_ACTIONS") == "true":
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        if repository == "AIM-RAGTune/rag-tuning-governance-public":
            return "github_actions_public_repo"
        return "github_actions_unapproved_repo"
    return "deployed_public_repo"


def normalize_remote_line(line: str) -> str:
    for suffix in (" (fetch)", " (push)"):
        line = line.replace(suffix, "")
    if "\t" in line:
        line = line.split("\t", 1)[1]
    return line.strip()


def public_repository_remote_allowed(remotes: str) -> bool:
    """Allow the approved public clean-history repository in deployed mode.

    Local unpublished export packages still reject external remotes. The
    public repository is intentionally different: it is already deployed, has
    a fresh one-commit history, and carries a public repository note. GitHub
    Actions mode is allowed only for the expected public repository slug.
    """

    mode = publication_remote_mode()
    if mode in {"local_unpublished", "github_actions_unapproved_repo"}:
        return False
    if mode not in {"deployed_public_repo", "github_actions_public_repo"}:
        return False
    allowed_remotes = {
        "https://github.com/AIM-RAGTune/rag-tuning-governance-public",
        "https://github.com/AIM-RAGTune/rag-tuning-governance-public.git",
        "git@github.com:AIM-RAGTune/rag-tuning-governance-public",
        "git@github.com:AIM-RAGTune/rag-tuning-governance-public.git",
    }
    remote_lines = [normalize_remote_line(line) for line in remotes.splitlines() if line.strip()]
    if not remote_lines:
        return False
    public_note = ROOT / "docs" / "public_repository_note.md"
    if not public_note.exists():
        return False
    return all(line in allowed_remotes for line in remote_lines)


def validate_no_crag_raw_text_fields() -> None:
    root = ROOT / CRAG_ARTIFACT_ROOT
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        lower_name = path.name.lower()
        if path.suffix.lower() == ".csv" and "crag" in rel.lower():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
            raw_fields = [field for field in fieldnames if field.lower() in RAW_TEXT_JSON_KEYS]
            if raw_fields:
                fail(f"raw CRAG text field(s) in {rel}: {raw_fields}")
        elif path.suffix.lower() == ".json" and "crag" in rel.lower():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            hits: list[str] = []

            def walk(value: object, prefix: str = "") -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        child_path = f"{prefix}.{key}" if prefix else key
                        if key.lower() in RAW_TEXT_JSON_KEYS:
                            hits.append(child_path)
                        walk(child, child_path)
                elif isinstance(value, list):
                    for idx, child in enumerate(value):
                        walk(child, f"{prefix}[{idx}]")

            walk(payload)
            if hits:
                fail(f"raw CRAG text key(s) in {rel}: {hits[:5]}")
        elif path.suffix.lower() == ".md" and "crag" in rel.lower() and ("case" in lower_name or "pack" in lower_name):
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?im)^\\s*-?\\s*Query:\\s*\\S", text):
                fail(f"raw CRAG query line in {rel}")


def validate_generative_artifacts() -> None:
    synthesis_path = ROOT / "results/generative_llm_validation/synthesis_result.json"
    if not synthesis_path.exists():
        fail("missing generative validation synthesis result")
    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    result_class = str(synthesis.get("result_class", ""))
    allowed = {
        "GEN_LLM_SYNTHESIS_GENERATIVE_VALIDATION_SUPPORTED",
        "GEN_LLM_SYNTHESIS_DIRECTIONAL",
        "GEN_LLM_SYNTHESIS_MIXED",
        "GEN_LLM_SYNTHESIS_INCONCLUSIVE",
        "GEN_LLM_SYNTHESIS_BLOCKED",
        "GEN_LLM_SYNTHESIS_NEGATIVE",
    }
    if result_class not in allowed:
        fail(f"unknown generative synthesis result class: {result_class}")
    root = ROOT / GENERATIVE_ARTIFACT_ROOT
    if not root.exists():
        fail("missing generative validation artifact root")
    positive_result_classes = {
        "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY",
        "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY",
        "GEN_LLM_GOVERNANCE_IMPROVES_GENERATED_QUALITY_UNDER_FIXED_BUDGET",
        "GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG",
        "GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY_CRAG",
        "GEN_LLM_GOVERNANCE_IMPROVES_GENERATED_QUALITY_UNDER_FIXED_BUDGET_CRAG",
    }
    forbidden_keys = {"prompt_text", "generated_answer", "generated_answer_text", "answer_text", "raw_answer"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            hits: list[str] = []

            def walk(value: object, prefix: str = "") -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        child_path = f"{prefix}.{key}" if prefix else key
                        if key.lower() in forbidden_keys:
                            hits.append(child_path)
                        walk(child, child_path)
                elif isinstance(value, list):
                    for idx, child in enumerate(value):
                        walk(child, f"{prefix}[{idx}]")

            walk(payload)
            if hits:
                fail(f"raw generative text key(s) in {rel}: {hits[:5]}")
            if path.name == "primary_outcome_statistics.json":
                result = str(payload.get("result_class", ""))
                if result in positive_result_classes:
                    if not payload.get("usable_quality_signal", False):
                        fail(f"positive generated-governance result lacks usable quality signal in {rel}")
                    if int(payload.get("non_empty_generated_answers", 0)) <= 0:
                        fail(f"positive generated-governance result has no nonempty generated answers in {rel}")
                    if int(payload.get("unique_answer_hash_count", 0)) <= 1:
                        fail(f"positive generated-governance result lacks answer-hash diversity in {rel}")
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                fieldnames = csv.DictReader(handle).fieldnames or []
            raw_fields = [field for field in fieldnames if field.lower() in forbidden_keys]
            if raw_fields:
                fail(f"raw generated-answer field(s) in {rel}: {raw_fields}")
    audit_manifest = ROOT / "artifacts/generative_llm_validation/hotpotqa_quality_signal_audit/audit_manifest.json"
    if audit_manifest.exists():
        audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
        if "zero" in str(audit.get("prior_zero_delta_explanation", "")).lower() and not audit.get("result_class"):
            fail("HotpotQA quality-signal audit lacks a machine-readable result class")
    guardrail_v2_path = ROOT / "results/generative_llm_validation/crag_quality_risk_guardrail_v2_comparison.json"
    if guardrail_v2_path.exists():
        guardrail_v2 = json.loads(guardrail_v2_path.read_text(encoding="utf-8"))
        if not guardrail_v2.get("pooled_cross_offset_validation"):
            fail("CRAG quality-risk guardrail v2 lacks pooled cross-offset validation")
        if not guardrail_v2.get("heldout_offset_testing"):
            fail("CRAG quality-risk guardrail v2 lacks held-out-offset testing")
        if not guardrail_v2.get("deployable_features_only"):
            fail("CRAG quality-risk guardrail v2 does not declare deployable-only features")
        if guardrail_v2.get("raw_text_features_used"):
            fail("CRAG quality-risk guardrail v2 used raw text features")
        if guardrail_v2.get("raw_prompts_committed") or guardrail_v2.get("raw_generated_answers_committed"):
            fail("CRAG quality-risk guardrail v2 committed raw prompt or generated-answer text")
        if int(guardrail_v2.get("quality_loss_blocked_count", 0)) > 0:
            blocked_class = "CRAG_GEN_LLM_QUALITY_RISK_GUARDRAIL_V2_BLOCKED_HELDOUT_QUALITY_LOSS"
            if guardrail_v2.get("result_class") != blocked_class:
                fail("CRAG quality-risk guardrail v2 did not block promotion after held-out quality loss")
    docs_text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in [
            ROOT / "docs/generative_llm_validation.md",
            ROOT / "docs/platform_benchmarking_boundary.md",
        ]
        if path.exists()
    )
    if "official platform benchmarking completed" in docs_text:
        fail("local or hosted generative validation is described as official platform benchmarking")


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).exists()]
    if missing:
        fail(f"missing required files: {missing}")
    export_missing = [path for path in EXPORT_REQUIRED if not (EXPORT_ROOT / path).exists()]
    if export_missing and os.environ.get("EXPORT_ROOT"):
        fail(f"missing export approval files: {export_missing}")

    files = tracked_files()
    for rel in files:
        rel_text = rel.as_posix()
        if any(part in rel_text for part in FORBIDDEN_TRACKED_PARTS):
            fail(f"forbidden tracked path: {rel_text}")
        path = ROOT / rel
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            fail(f"file exceeds 50 MB threshold: {rel_text}")
        if path.is_file() and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    fail(f"secret-like pattern {label} in {rel_text}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    forbidden_claims = [
        "rag compass is proven superior",
        "production validated",
        "human evaluation completed",
        "generative llm validation completed",
        "official platform benchmark completed",
    ]
    for claim in forbidden_claims:
        if claim in readme:
            fail(f"unsupported claim found in README: {claim}")

    summary = json.loads((ROOT / "results/evidence_summary.json").read_text(encoding="utf-8"))
    if "unsupported_claims" not in summary:
        fail("evidence summary lacks unsupported_claims")

    validate_no_crag_raw_text_fields()
    validate_generative_artifacts()

    try:
        remotes = subprocess.check_output(["git", "remote", "-v"], cwd=ROOT, text=True)
    except Exception:
        remotes = ""
    if remotes.strip():
        approved_remote = approved_deployment_remote()
        approved = "approved-internal-git-url" in remotes
        if approved_remote:
            remote_lines = [line for line in remotes.splitlines() if line.strip()]
            approved = approved or all(approved_remote in line for line in remote_lines)
        approved = approved or public_repository_remote_allowed(remotes)
        if not approved:
            fail("external git remote configured in publication bundle")

    print("publication validation passed")


if __name__ == "__main__":
    main()
