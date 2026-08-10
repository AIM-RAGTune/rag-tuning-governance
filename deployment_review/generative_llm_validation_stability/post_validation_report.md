# CRAG Generative Stability Validation Report

## Experiment

- Branch: `crag-generative-stability-validation`
- Starting commit: `9830b6163f4d5dc5e61205e26d98df5ea9190e71`
- Generator provider: `ollama`
- Generator model: `qwen3:8b`
- Repeat type: independent deterministic CRAG offset slices
- Offsets compared: `0`, `24`, `36`, `60`
- Runs compared: `4`
- Usable generated-quality runs: `4`
- Positive cost-at-equivalent-generated-quality runs: `1`

## Result

- Stability result: `CRAG_GEN_LLM_COST_RESULT_NOT_STABLE_ACROSS_REPEATS`
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`
- Interpretation: the primary offset-0 CRAG slice produced a cost-reduction signal, but independent deterministic repeats at offsets 24, 36, and 60 did not reproduce it.

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
- pytest: 71 passed
- make validate-publication: passed
- make test: 71 passed
- compile: passed
- large-file scan: no public tracked files over threshold
- raw prompt/generated-answer scan: expected sanitizer, test, field-name, and sanitized artifact references only
- raw dataset text scan: expected sanitizer, test, field-name, and sanitized artifact references only
- secret scan: expected scanner/config variable-name references only
- private path scan: no committed private paths
- overclaim scan: explicit unsupported-claim boundary language only

## Claim Boundary

This stability experiment does not support broad generative governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
