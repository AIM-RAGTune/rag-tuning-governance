# Generative LLM Guarded-Latency Validation Report

## Scope

This phase adds a quality-risk guardrail to the CRAG generated-answer latency selector and reruns the same `llama3.2:3b` fixed offsets before increasing sample size.

## Guardrail

- Policy: `quality_guarded_latency_adaptive_expansion`.
- Rule: start with two evidence items and expand to five when local CRAG answer/alternate-answer containment is absent.
- Export policy: public artifacts contain hashes, identifiers, counts, and metrics only.
- Raw CRAG questions, raw evidence/source text, raw API responses, prompts, and generated answers are not committed.

## Fixed-Offset Results

| Offset | Result class | Governed | Quality-only | Quality delta mean | Latency delta mean ms | Cost delta mean | API-call delta | Expansion rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG` | `quality_guarded_latency_adaptive_expansion` | `expanded_retrieval_multi_endpoint` | 0.0000000000 | 212.5706390167 | -0.6468000000 | -0.5 | 0.7500 |
| 24 | `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG` | `quality_guarded_latency_adaptive_expansion` | `expanded_retrieval_multi_endpoint` | 0.0101851852 | -314.9517220445 | -1.2846000000 | -1.0 | 0.5625 |
| 36 | `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG` | `quality_guarded_latency_adaptive_expansion` | `expanded_retrieval_multi_endpoint` | 0.0000000000 | -76.4160826802 | -0.6779200000 | -0.6 | 0.5625 |
| 60 | `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG` | `quality_guarded_latency_adaptive_expansion` | `expanded_retrieval_multi_endpoint` | 0.0000000000 | -116.1995415265 | -1.2585000000 | -1.0 | 0.6250 |

## Stability Result

`CRAG_GEN_LLM_LATENCY_RESULT_INCONCLUSIVE_ACROSS_REPEATS`

The guardrail removed the prior unguarded pattern of three quality-loss slices, but the expanded evidence path also weakened latency evidence. No fixed-offset slice met the predeclared latency-reduction criterion because each latency CI crossed zero.

## Publication Hygiene

- Generated rows: 768.
- Non-empty generated answers: 768.
- Parse failures: 0.
- Raw prompts committed: no.
- Raw generated answers committed: no.
- Raw CRAG questions committed: no.
- Raw CRAG evidence/source text committed: no.
- Raw API responses committed: no.
- Secrets committed: no.
- Private paths committed: no.

## Validation

- Publication validator: passed.
- Publication tests: 80 passed.
- `make validate-publication`: passed.
- `make test`: 80 passed.
- Compile: passed for `src` and `scripts`.
- Large-file scan: no tracked large files; an ignored local HotpotQA cache file exists under `.local_data/` and is not tracked.
- Raw prompt/generated-answer scan: passed with expected sanitized field names, hash fields, false export flags, scanner definitions, source code, and tests only.
- Secret scan: passed with expected scanner/config/environment-variable names only.
- Private-path scan: passed.
- Overclaim scan: passed with explicit unsupported-claim boundary statements only.

## Claim Boundary

This phase does not claim latency reduction at equivalent generated quality. It supports a narrower diagnostic statement: the quality-risk guardrail improved quality protection relative to the unguarded latency selector, but did not produce a stable latency win.
