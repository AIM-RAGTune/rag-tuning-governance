## Summary

Adds a validation-trained deployable quality-risk predictor for the CRAG generated-answer latency selector and reruns the same `llama3.2:3b` fixed offsets before increasing sample size.

## Predictor

- Policy: `learned_quality_risk_latency_adaptive_expansion`.
- Training: validation generated-quality loss within each fixed-offset slice.
- Runtime features: deployable retrieval metadata/count features only.
- Raw text features used: no.
- Gate: reduced expansion rate versus the label-aware guardrail while preserving validation generated-quality noninferiority.
- Gate result: passed on all four offsets.

## Results

- Offsets: 0, 24, 36, 60.
- Positive latency slices: 1 / 4.
- Quality-loss slices: 2 / 4.
- Inconclusive slices: 1 / 4.
- Learned-predictor comparison: `CRAG_GEN_LLM_LEARNED_RISK_PREDICTOR_LATENCY_MIXED_CONFIRMATORY_QUALITY_LOSS`.
- Stability comparison: `CRAG_GEN_LLM_LATENCY_RESULT_MIXED_ACROSS_REPEATS`.
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`.

## Interpretation

The learned predictor reduced expansions on validation-gated fixed offsets, but confirmatory quality protection did not persist. This does not support increasing sample size yet and does not support a stable latency-reduction claim.

## Validation

- Publication validator: passed.
- Publication tests: 82 passed.
- `make validate-publication`: passed.
- `make test`: 82 passed.
- Compile: passed.
- Large-file scan: no tracked large files; ignored local `.local_data/` cache only.
- Raw prompt/generated-answer scan: passed with expected sanitized field/hash references only.
- Secret scan: passed with expected scanner/config/env-var names only.
- Private-path scan: passed.
- Overclaim scan: passed with explicit unsupported-claim boundary statements only.

## Claim Boundaries

This PR does not claim stable latency reduction at equivalent generated quality, broad generative governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
