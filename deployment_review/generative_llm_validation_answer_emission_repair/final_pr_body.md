## Summary

Repairs the CRAG generative answer-emission path by adding stricter final-answer prompting, charging empty-answer retries to cost/latency, using Ollama chat by default for `gpt-oss*`, and adding a faster non-thinking instruct model (`llama3.2:3b`) for a slightly larger fixed-offset CRAG comparison.

## Answer-emission repair

- Initial repair target: `gpt-oss:20b`
- gpt-oss repair smoke: 20 / 44 parse failures
- Faster instruct model: `llama3.2:3b`
- llama3.2 smoke: 0 / 44 parse failures
- llama3.2 fixed-offset comparison: 0 / 704 parse failures
- Answer-emission comparison result: `CRAG_GEN_LLM_ANSWER_EMISSION_REPAIRED_NO_COST_RESULT`

## Larger fixed-offset CRAG comparison

- Offsets: `0`, `24`, `36`, `60`
- Sample size: 16 CRAG examples per offset
- Generation rows: 704
- Result: `CRAG_GEN_LLM_COST_RESULT_INCONCLUSIVE_ACROSS_REPEATS`
- Positive cost-result slices: 0 / 4
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`

## Validation

- publication validator: passed
- pytest: `76 passed`
- make validate-publication: passed
- make test: `76 passed`
- compile: passed
- large-file scan: passed
- raw prompt/generated-answer scan: passed with expected sanitizer/test/field-name/sanitized artifact references only
- raw dataset text scan: passed with expected sanitizer/test/field-name/sanitized artifact references only
- secret scan: passed with expected scanner/config variable-name references only
- private path scan: passed

## Claim Boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, broad generative governance validation, or RAG Compass superiority. The result repairs answer emission but does not recover stable cost-at-equivalent-generated-quality evidence.
