## Summary

Adds a CRAG generative stability experiment using additional independent deterministic offset slices with the pinned local Ollama `qwen3:8b` generator. The stability comparison finds that the primary cost-at-equivalent-generated-quality result is not stable across repeats.

## CRAG Stability Experiment

- Repeat type: independent deterministic CRAG offset slices
- Offsets compared: `0`, `24`, `36`, `60`
- Runs compared: `4`
- Usable generated-quality runs: `4`
- Positive cost-at-equivalent-generated-quality runs: `1`
- Stability result class: `CRAG_GEN_LLM_COST_RESULT_NOT_STABLE_ACROSS_REPEATS`
- Synthesis result class: `GEN_LLM_SYNTHESIS_MIXED`

## Slice Outcomes

- Offset `0`: `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG`
- Offset `24`: `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`
- Offset `36`: `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`
- Offset `60`: `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`

## Interpretation

The primary offset-0 CRAG slice produced a cost-reduction signal, but no independent usable repeat reproduced it. This preserves the result as mixed bounded local evidence and avoids upgrading the claim.

## Validation

- publication validator: passed
- pytest: 71 passed
- make validate-publication: passed
- make test: 71 passed
- compile: passed
- large-file scan: no public tracked files over threshold
- raw prompt/generated-answer scan: expected sanitizer/test/field-name references only
- raw dataset text scan: expected sanitizer/test/field-name references only
- secret scan: expected scanner/config variable-name references only
- private path scan: no committed private paths

## Claim Boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, broad generative governance validation, or RAG Compass superiority.
