# Generative LLM Validation

RAGTune Generative LLM Validation v1 adds a pinned-generator path for policy-specific answer generation and sanitized generated-answer scoring.

## Current Status

- HotpotQA v1.1 quality-signal audit: `HOTPOTQA_GEN_LLM_QUALITY_SIGNAL_CONFIRMED` on a bounded 12-example local Ollama run with `qwen3:8b`; the primary generated-governance result is `GEN_LLM_GOVERNANCE_INCONCLUSIVE` because governed and quality-only selection chose the same policy.
- CRAG generator/evaluator repair and larger bounded primary rerun: qwen3 answer emission is repaired by disabling Ollama thinking output; the 12-example CRAG primary slice is `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG` with usable generated-answer quality, lower measured cost, and lower measured latency for the governed selector relative to quality-only selection.
- Independent CRAG repeats: deterministic non-overlapping 12-example slices at offsets 24, 36, and 60 all produced usable generated-answer quality but returned `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`; none reproduced the primary cost result.
- Stability comparison: `CRAG_GEN_LLM_COST_RESULT_NOT_STABLE_ACROSS_REPEATS` across four usable CRAG slices. The primary offset-0 slice was positive; offsets 24, 36, and 60 were not.
- Second-model comparison: Ollama `gpt-oss:20b` was run on the same offsets 0, 24, 36, and 60. All four slices had usable generated-quality signals, but zero slices produced a positive cost-at-equivalent-generated-quality result. The comparison result is `CRAG_GEN_LLM_COST_RESULT_MIXED_OR_INCONCLUSIVE_ACROSS_MODELS`.
- Answer-emission repair model: Ollama `llama3.2:3b` was added as a faster non-thinking instruct model after `gpt-oss:20b` showed high blank-answer rates. On four slightly larger 16-example fixed-offset CRAG slices, it produced 704 / 704 non-empty generated answers, 0 parse failures, and usable nonconstant quality signals. The stability result was `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`, with zero positive cost-result slices.
- Synthesis: `GEN_LLM_SYNTHESIS_MIXED` because the primary qwen CRAG slice produced bounded local generative support, independent CRAG repeats did not reproduce it, the second pinned local generator did not recover the cost result, the faster instruct model repaired answer emission but did not recover the cost result, and HotpotQA remained inconclusive.

The runs used a local generator and sanitized generated-answer metrics. They do not commit raw prompts, raw generated answers, raw questions, raw contexts, raw CRAG evidence, raw CRAG API responses, or supporting-fact sentence text.

## Claim Boundary

This supports a bounded local CRAG generated-governance cost-reduction result on one deterministic qwen slice, plus a three-repeat non-replication pattern, a second-model inconclusive pattern, a faster-model answer-emission repair without a cost-result recovery, and a HotpotQA generated-answer quality-signal audit. It does not establish production readiness, official platform benchmarking, human validation, broad governance superiority, or RAG Compass superiority.

The current generative result is small-sample local evidence and is not stable across deterministic CRAG repeats or recovered by the second pinned local model. A faster instruct model fixed answer emission, but that narrowed the limitation to governance-effect instability rather than supporting a stronger claim. Additional offsets, larger samples, improved selector design, and human/platform validation remain future work.

## Artifacts

- `artifacts/generative_llm_validation/crag/`
- `artifacts/generative_llm_validation/crag_repeat/`
- `artifacts/generative_llm_validation/crag_repeat_offset_36/`
- `artifacts/generative_llm_validation/crag_repeat_offset_60/`
- `artifacts/generative_llm_validation/crag_second_model_gpt_oss_20b_offset_0/`
- `artifacts/generative_llm_validation/crag_second_model_gpt_oss_20b_offset_24/`
- `artifacts/generative_llm_validation/crag_second_model_gpt_oss_20b_offset_36/`
- `artifacts/generative_llm_validation/crag_second_model_gpt_oss_20b_offset_60/`
- `artifacts/generative_llm_validation/crag_llama3_2_3b_offset_0/`
- `artifacts/generative_llm_validation/crag_llama3_2_3b_offset_24/`
- `artifacts/generative_llm_validation/crag_llama3_2_3b_offset_36/`
- `artifacts/generative_llm_validation/crag_llama3_2_3b_offset_60/`
- `artifacts/generative_llm_validation/hotpotqa/`
- `results/generative_llm_validation/`

Public artifacts contain hashes, counts, model identifiers, policy identifiers, and metrics only.
