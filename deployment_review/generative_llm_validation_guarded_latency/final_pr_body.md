## Summary

Adds a quality-risk guardrail to the CRAG generated-answer latency selector and reruns the same `llama3.2:3b` fixed offsets before increasing sample size.

## Guardrail

- Policy: `quality_guarded_latency_adaptive_expansion`.
- Rule: start with two evidence items and expand to five when local CRAG answer/alternate-answer containment is absent.
- Primary endpoint: latency.
- Raw CRAG text, prompts, generated answers, and API responses are not committed.

## Results

- Offsets: 0, 24, 36, 60.
- Generated rows: 768.
- Parse failures: 0.
- Governed winner pattern: `quality_guarded_latency_adaptive_expansion`.
- Quality-only winner pattern: `expanded_retrieval_multi_endpoint`.
- Result classes: 4 / 4 `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`.
- Positive latency slices: 0 / 4.
- Quality-loss slices: 0 / 4.
- Stability result: `CRAG_GEN_LLM_LATENCY_RESULT_INCONCLUSIVE_ACROSS_REPEATS`.
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`.

## Interpretation

The guardrail removed the previous unguarded pattern where three of four latency-selector slices were quality-loss results. It did not produce a latency win because latency confidence intervals crossed zero on every fixed-offset slice.

## Validation

- Publication validator: passed.
- Publication tests: 80 passed.
- `make validate-publication`: passed.
- `make test`: 80 passed.
- Compile: passed.
- Large-file scan: no tracked large files; ignored local `.local_data/` cache only.
- Raw prompt/generated-answer scan: passed with expected sanitized field/hash references only.
- Secret scan: passed with expected scanner/config/env-var names only.
- Private-path scan: passed.
- Overclaim scan: passed with explicit unsupported-claim boundary statements only.

## Claim boundaries

This PR does not claim latency reduction at equivalent generated quality, broad generative governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
