# Generative LLM Learned Risk-Predictor Validation Report

## Scope

This phase learns a deployable, non-oracle CRAG generated-quality risk predictor on validation rows, then reruns the `llama3.2:3b` latency selector on the same fixed offsets only after each predictor passes its validation gate.

## Predictor Gate

- Policy: `learned_quality_risk_latency_adaptive_expansion`.
- Training split: validation rows within each fixed-offset slice.
- Training labels: generated-quality loss versus high-evidence quality-only output, computed locally.
- Runtime features: deployable retrieval metadata/count features only.
- Raw text features used: no.
- Gate: predicted expansion rate must be lower than the label-aware guardrail and validation generated-quality delta must remain within the 0.01 noninferiority margin with no unprotected validation quality-risk examples.
- Gate result: passed on all four fixed offsets.

## Fixed-Offset Results

| Offset | Result class | Validation expansion rate | Old guardrail validation expansion rate | Confirmatory expansion rate | Quality delta mean | Latency delta mean ms | Latency CI high ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG` | 0.1667 | 0.6667 | 0.1875 | -0.0071458333 | -771.3268264197 | 471.9977928326 |
| 24 | `GEN_LLM_GOVERNANCE_REDUCES_LATENCY_AT_EQUIVALENT_GENERATED_QUALITY_CRAG` | 0.0000 | 0.6667 | 0.0000 | -0.0064814815 | -1201.8911386840 | -949.9215418473 |
| 36 | `GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG` | 0.2500 | 0.7500 | 0.0625 | -0.0186666667 | -1220.1537748799 | -1100.6712503731 |
| 60 | `GEN_LLM_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS_CRAG` | 0.3333 | 0.6667 | 0.1875 | -0.0500000000 | -913.1593328590 | -1096.5401660651 |

## Result

- Learned-predictor comparison: `CRAG_GEN_LLM_LEARNED_RISK_PREDICTOR_LATENCY_MIXED_CONFIRMATORY_QUALITY_LOSS`.
- Stability comparison: `CRAG_GEN_LLM_LATENCY_RESULT_MIXED_ACROSS_REPEATS`.
- Synthesis: `GEN_LLM_SYNTHESIS_MIXED`.

The learned predictor reduced expansions and produced one latency-positive fixed-offset slice, but two confirmatory slices exceeded the generated-quality loss threshold. This does not justify increasing sample size yet.

## Publication Hygiene

- Raw CRAG questions committed: no.
- Raw evidence/source text committed: no.
- Raw API responses committed: no.
- Raw prompts committed: no.
- Raw generated answers committed: no.
- Secrets committed: no.
- Private paths committed: no.

## Validation

- Publication validator: passed.
- Publication tests: 82 passed.
- `make validate-publication`: passed.
- `make test`: 82 passed.
- Compile: passed for `src` and `scripts`.
- Large-file scan: no tracked large files; ignored local HotpotQA cache exists under `.local_data/`.
- Raw prompt/generated-answer scan: passed with expected sanitized field names, hash fields, false export flags, scanner definitions, source code, and tests only.
- Secret scan: passed with expected scanner/config/environment-variable names only.
- Private-path scan: passed.
- Overclaim scan: passed with explicit unsupported-claim boundary statements only.

## Claim Boundary

This phase does not claim stable latency reduction at equivalent generated quality. It supports a narrower diagnostic statement: a validation-trained deployable risk predictor can reduce expansion rates, but this specific predictor did not reliably prevent confirmatory generated-quality loss.
