## Summary

Runs a second pinned local instruct model, Ollama `gpt-oss:20b`, on the same CRAG deterministic offsets used by the qwen stability experiment. The second model produced usable generated-quality signals on all four slices but did not recover a stable cost-at-equivalent-generated-quality result.

## Second-Model Experiment

- Primary comparison model: `qwen3:8b`
- Second pinned local model: `gpt-oss:20b`
- Offsets compared: `0`, `24`, `36`, `60`
- Second-model runs compared: `4`
- Second-model usable generated-quality runs: `4`
- Second-model positive cost-result runs: `0`
- Second-model stability result class: `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`
- Cross-model comparison result class: `CRAG_GEN_LLM_COST_RESULT_MIXED_OR_INCONCLUSIVE_ACROSS_MODELS`
- Synthesis result class: `GEN_LLM_SYNTHESIS_MIXED`

## Interpretation

The original qwen primary slice remains a bounded positive result, but it was not stable across qwen repeats and was not recovered by the second pinned local generator. This preserves the generative validation result as mixed/inconclusive evidence rather than a stronger governance claim.

## Validation

- publication validator: passed
- pytest: `72 passed`
- make validate-publication: passed
- make test: `72 passed`
- compile: passed
- large-file scan: passed; no public tracked files over threshold
- raw prompt/generated-answer scan: passed with expected sanitizer/test/field-name/sanitized artifact references only
- raw dataset text scan: passed with expected sanitizer/test/field-name/sanitized artifact references only
- secret scan: passed with expected scanner/config variable-name references only
- private path scan: passed; no committed private paths found

## Claim Boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, broad generative governance validation, or RAG Compass superiority.
