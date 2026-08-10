# CRAG Generative Second-Model Validation Report

## Experiment

- Branch: `crag-generative-second-model-validation`
- Starting commit: `244280786b7ba886ac7f5fda089709e2a6cd17a0`
- Primary model previously compared: Ollama `qwen3:8b`
- Second pinned local model: Ollama `gpt-oss:20b`
- Repeat type: same deterministic CRAG offset slices
- Offsets compared: `0`, `24`, `36`, `60`
- Second-model runs compared: `4`
- Second-model usable generated-quality runs: `4`
- Second-model positive cost-result runs: `0`

## Result

- Second-model stability result: `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`
- Cross-model comparison result: `CRAG_GEN_LLM_COST_RESULT_MIXED_OR_INCONCLUSIVE_ACROSS_MODELS`
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`
- Interpretation: the qwen primary slice produced one cost-reduction signal, but qwen repeats did not reproduce it and the second pinned local model did not recover a stable cost-at-equivalent-generated-quality result.

## Publication Hygiene

- Raw CRAG questions committed: no
- Raw CRAG evidence/source text committed: no
- Raw CRAG API responses committed: no
- Raw prompts committed: no
- Raw generated answers committed: no
- Secrets committed: no
- Private paths committed: no

## Validation

- publication validator: passed
- pytest: `72 passed`
- make validate-publication: passed
- make test: `72 passed`
- compile: passed
- large-file scan: passed; no public tracked files over threshold
- raw prompt/generated-answer scan: passed with expected sanitizer, test, field-name, and sanitized artifact references only
- raw dataset text scan: passed with expected sanitizer, test, field-name, and sanitized artifact references only
- secret scan: passed with expected scanner/config variable-name references only
- private path scan: passed; no committed private paths found
- overclaim scan: passed; hits were explicit unsupported-claim boundary language only

## Claim Boundary

This second-model experiment does not support broad generative governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
