# Generative LLM Validation

RAGTune Generative LLM Validation v1 adds a pinned-generator path for policy-specific answer generation and sanitized generated-answer scoring.

## Current Status

- HotpotQA v1.1 quality-signal audit: `HOTPOTQA_GEN_LLM_QUALITY_SIGNAL_CONFIRMED` on a bounded 12-example local Ollama run with `qwen3:8b`; the primary generated-governance result is `GEN_LLM_GOVERNANCE_INCONCLUSIVE` because governed and quality-only selection chose the same policy.
- CRAG generator/evaluator repair and larger bounded rerun: qwen3 answer emission is repaired by disabling Ollama thinking output; the 12-example CRAG run is `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG` with usable generated-answer quality, lower measured cost, and lower measured latency for the governed selector relative to quality-only selection.
- Synthesis: `GEN_LLM_SYNTHESIS_DIRECTIONAL` because CRAG produced bounded local generative support while HotpotQA remained inconclusive.

The runs used a local generator and sanitized generated-answer metrics. They do not commit raw prompts, raw generated answers, raw questions, raw contexts, raw CRAG evidence, raw CRAG API responses, or supporting-fact sentence text.

## Claim Boundary

This supports a bounded local CRAG generated-governance cost-reduction result and a HotpotQA generated-answer quality-signal audit. It does not establish production readiness, official platform benchmarking, human validation, broad governance superiority, or RAG Compass superiority.

The current generative result is small-sample local evidence. Larger independent repeats and human/platform validation remain future work.

## Artifacts

- `artifacts/generative_llm_validation/crag/`
- `artifacts/generative_llm_validation/hotpotqa/`
- `results/generative_llm_validation/`

Public artifacts contain hashes, counts, model identifiers, policy identifiers, and metrics only.
