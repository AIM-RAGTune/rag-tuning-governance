# Generative LLM Validation

RAGTune Generative LLM Validation v1 adds a pinned-generator path for policy-specific answer generation and sanitized generated-answer scoring.

## Current Status

- CRAG generative validation: `GEN_LLM_VALIDATION_BLOCKED_NO_GENERATOR` or blocked by CRAG evaluator mapping in environments where the approved CRAG runtime cannot be invoked safely.
- HotpotQA generative validation: `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY` on a bounded local Ollama run with `qwen3:8b`.
- Synthesis: `GEN_LLM_SYNTHESIS_DIRECTIONAL`.

The HotpotQA run used a local generator and sanitized generated-answer metrics. It does not commit raw prompts, raw generated answers, raw questions, raw contexts, or supporting-fact sentence text.

## Claim Boundary

This supports a bounded local generative validation signal on HotpotQA. It does not establish production readiness, official platform benchmarking, human validation, broad governance superiority, or RAG Compass superiority.

The current generative result is directional because CRAG generative validation did not produce a completed generated-answer quality run in the public artifact set.

## Artifacts

- `artifacts/generative_llm_validation/crag/`
- `artifacts/generative_llm_validation/hotpotqa/`
- `results/generative_llm_validation/`

Public artifacts contain hashes, counts, model identifiers, policy identifiers, and metrics only.
