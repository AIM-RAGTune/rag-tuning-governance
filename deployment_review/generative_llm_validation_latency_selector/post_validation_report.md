# Generative LLM Latency-Selector Validation Report

## Scope

This phase redesigned the CRAG generated-answer selector comparison to reduce accidental identity between governed and quality-only winners, then reran the `llama3.2:3b` fixed-offset CRAG suite with a predeclared latency endpoint.

## Selector Design

- Primary endpoint: latency.
- Generator: local Ollama `llama3.2:3b`.
- Selector design: `validation_split_quality_only_high_evidence_vs_governed_latency_feasible_confirmatory_eval`.
- Governed selector: lowest validation latency among deployable policies within the generated-quality noninferiority margin.
- Quality-only selector: highest validation generated quality among predeclared high-evidence candidates; ignores cost and latency.
- Confirmatory evaluation: deterministic fixed offsets evaluated on held-out confirmatory rows when available.

## Fixed-Offset Results

| Offset | Result class | Governed | Quality-only | Quality delta mean | Latency delta mean ms | Cost delta mean | API-call delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | `GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG` | `static_default_policy` | `expanded_retrieval_multi_endpoint` | -0.0238125000 | -2506.0456044351 | -3.7772166667 | -3.0 |
| 24 | `GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY_CRAG` | `static_default_policy` | `expanded_retrieval_multi_endpoint` | -0.0064814815 | -2276.6198126289 | -3.7854333333 | -3.0 |
| 36 | `GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG` | `static_default_policy` | `expanded_retrieval_multi_endpoint` | -0.0186666667 | -2398.2915582135 | -3.7126000000 | -3.0 |
| 60 | `GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG` | `static_default_policy` | `expanded_retrieval_multi_endpoint` | -0.0333333333 | -2549.2699169554 | -3.7888666667 | -3.0 |

## Stability Result

`CRAG_GEN_LLM_LATENCY_RESULT_MIXED_ACROSS_REPEATS`

The redesigned selector comparison successfully separated governed and quality-only winners on all four fixed-offset slices and consistently reduced latency, cost, and API calls. However, only one of four slices met generated-quality noninferiority; the other three were quality-loss results. This is mixed bounded local evidence, not a stronger generative governance claim.

## Publication Hygiene

- Raw prompts committed: no.
- Raw generated answers committed: no.
- Raw CRAG questions committed: no.
- Raw CRAG evidence/source documents committed: no.
- Raw API responses committed: no.
- Secrets committed: no.
- Private local paths committed: no.
- Publication validator: passed.
- Publication tests: 78 passed.
- Compile: passed.
- `make validate-publication`: passed.
- `make test`: 78 passed.
- Large-file scan: found ignored local HotpotQA cache under `.local_data`; no `.local_data` files are tracked.
- Raw prompt/generated-answer scan: passed with expected sanitizer field names, hashes, and committed false flags.
- Secret scan: passed with expected scanner/config environment-variable names only.
- Private-path scan: passed.
- Overclaim scan: passed with explicit unsupported-claim boundary statements only.

## Claim Boundaries

This phase does not claim broad generative LLM governance validation, official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
