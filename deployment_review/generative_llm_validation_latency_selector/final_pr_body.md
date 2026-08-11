## Summary

Redesigns the CRAG generated-answer selector comparison so governed and quality-only winners are no longer accidentally identical, then reruns the local `llama3.2:3b` fixed-offset CRAG suite with a predeclared latency endpoint.

## Selector redesign

- Primary endpoint: latency.
- Governed selector: lowest validation latency among deployable policies within the generated-quality noninferiority margin.
- Quality-only selector: highest validation generated quality among predeclared high-evidence candidates.
- Selector design: `validation_split_quality_only_high_evidence_vs_governed_latency_feasible_confirmatory_eval`.

## CRAG latency result

- Generator: local Ollama `llama3.2:3b`.
- Fixed offsets: 0, 24, 36, 60.
- Generated rows: 704.
- Parse failures: 0.
- Governed winner pattern: `static_default_policy`.
- Quality-only winner pattern: `expanded_retrieval_multi_endpoint`.
- Positive latency-at-equivalent-generated-quality slices: 1 / 4.
- Quality-loss slices: 3 / 4.
- Stability result: `CRAG_GEN_LLM_LATENCY_RESULT_MIXED_ACROSS_REPEATS`.
- Synthesis result: `GEN_LLM_SYNTHESIS_MIXED`.

## Interpretation

The redesigned selector comparison succeeded at separating the governed and quality-only choices and consistently reduced latency, cost, and API calls. The result is not stable enough for a stronger governance claim because generated-quality noninferiority held on only one fixed-offset slice.

## Publication hygiene

- Raw prompts committed: no.
- Raw generated answers committed: no.
- Raw CRAG questions committed: no.
- Raw CRAG source/evidence text committed: no.
- Raw API responses committed: no.
- Secrets committed: no.
- Private paths committed: no.

## Validation

- Publication validator: passed.
- Publication tests: 78 passed.
- `make validate-publication`: passed.
- `make test`: 78 passed.
- Compile: passed.
- Large-file scan: only ignored local `.local_data` cache found; no `.local_data` files are tracked.
- Raw prompt/generated-answer scan: expected sanitized field names/hashes only.
- Secret scan: expected scanner/config environment-variable names only.
- Private-path scan: passed.
- Overclaim scan: explicit unsupported-claim boundary statements only.

## Claim boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, broad generative validation, or RAG Compass superiority.
