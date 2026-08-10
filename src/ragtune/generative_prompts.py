from __future__ import annotations

from ragtune.generators.util import hash_text


def build_rag_prompt(*, question_text: str, evidence_items: list[dict[str, str]]) -> tuple[str, str]:
    evidence_lines = []
    for item in evidence_items:
        evidence_id = item.get("evidence_id", "evidence")
        text = item.get("text", "")
        evidence_lines.append(f"[{evidence_id}] {text}")
    prompt = (
        "You are answering a retrieval-augmented question.\n\n"
        "Use only the provided evidence. If the evidence is insufficient, answer exactly: "
        "INSUFFICIENT_EVIDENCE.\n\n"
        f"Question:\n{question_text}\n\n"
        "Evidence:\n"
        + "\n".join(evidence_lines)
        + "\n\nReturn a concise answer and cited evidence ids."
    )
    return prompt, hash_text(prompt)


def sanitized_prompt_record(*, prompt_hash: str, evidence_count: int, question_hash: str) -> dict[str, object]:
    return {
        "prompt_hash": prompt_hash,
        "question_hash": question_hash,
        "evidence_count": evidence_count,
        "prompt_text_exported": False,
        "raw_question_exported": False,
        "raw_evidence_exported": False,
    }
