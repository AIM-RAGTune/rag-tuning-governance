## Summary

Repairs qwen3:8b CRAG answer emission by disabling Ollama thinking mode for qwen3 generators, reruns the bounded CRAG generative validation, and preserves the resulting inconclusive governance outcome with sanitized generated-answer metrics.

## HotpotQA quality-signal audit

- Generator: `ollama`
- Model: `qwen3:8b`
- Sample size: 12 examples / 96 policy-generation rows
- Quality-signal result class: `HOTPOTQA_GEN_LLM_QUALITY_SIGNAL_CONFIRMED`
- Unique answer hashes: 2
- Non-empty answers: 2
- Abstentions: 0
- Parse failures: 94
- Answer-quality variance: 0.024150548562885805
- Supporting-fact/evidence variance: nonconstant evidence and quality scores; see `quality_signal_diagnostics.csv`
- Prior zero-delta explanation: generated answers and configured quality scores vary, but governed and quality-only selection chose the same policy in the confirmatory comparison
- Governed winner: `expanded_retrieval_multi_context`
- Quality-only winner: `expanded_retrieval_multi_context`
- RAG Compass rank: 2
- Generated-quality delta: 0.0 [0.0, 0.0]
- Answer-correctness delta: 0.0 [0.0, 0.0]
- Supporting-fact evidence delta: 0.0 [0.0, 0.0]
- Cost delta: 0.0 [0.0, 0.0]
- Latency delta: 0.0 [0.0, 0.0]

## CRAG generator/evaluator repair

- CRAG root/data configured: yes, via approved local paths at runtime only
- Mock API available: yes
- Generator reachable from CRAG script: yes
- Local evaluator available: yes
- qwen3 answer-emission repair: passed with Ollama `think: false`
- Evaluator mapping result class: `CRAG_GENERATED_QUALITY_LOCAL_EVALUATOR_ACTIVE`
- CRAG generative result class: `GEN_LLM_GOVERNANCE_INCONCLUSIVE_CRAG`
- Non-empty generated answers: 88
- Unique answer hashes: 32
- Governed winner: `expanded_retrieval_multi_endpoint`
- Quality-only winner: `expanded_retrieval_multi_endpoint`
- RAG Compass rank: 8
- Generated-quality delta: 0.0 [0.0, 0.0]
- Evidence-support delta: 0.0 [0.0, 0.0]
- Cost delta: 0.0 [0.0, 0.0]
- Latency delta: 0.0 [0.0, 0.0]

## Synthesis

- Result class: `GEN_LLM_SYNTHESIS_INCONCLUSIVE`
- Interpretation: HotpotQA and CRAG both have usable nonconstant generated-answer quality signals, but neither supported a governance improvement under the bounded samples.

## Validation

- publication validator: passed
- pytest: 69 passed
- make validate-publication: passed
- make test: 69 passed
- compile: passed
- raw prompt/generated-answer scan: expected sanitizer/test/field-name references only
- raw dataset text scan: expected sanitizer/test/field-name references only
- secret scan: expected scanner/config variable-name references only
- private path scan: no committed private paths
- large-file scan: no tracked public files over threshold

## Claim boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
