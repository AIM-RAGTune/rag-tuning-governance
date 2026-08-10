## Summary

Increases the CRAG generative validation sample after the qwen3:8b answer-emission repair, reruns the local Ollama CRAG generated-answer evaluation, and updates synthesis to reflect bounded directional CRAG evidence with HotpotQA still inconclusive.

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
- CRAG generative result class: `GEN_LLM_GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_GENERATED_QUALITY_CRAG`
- Sample size: 12 examples / 132 policy-generation rows
- Non-empty generated answers: 132
- Unique answer hashes: 48
- Governed winner: `pareto_frontier_selector`
- Quality-only winner: `expanded_retrieval_multi_endpoint`
- RAG Compass rank: 8
- Generated-quality delta: +0.0166533759 [-0.0363636364, +0.0362838915]
- Evidence-support delta: 0.0 [0.0, 0.0]
- Cost delta: -3.7790000000 [-3.8808000000, -3.7257000000]
- Latency delta: -5971.8510413853 ms [-8768.2858742774, -5159.1000836343]

## Synthesis

- Result class: `GEN_LLM_SYNTHESIS_DIRECTIONAL`
- Interpretation: CRAG produced bounded local generative support for reduced measured cost at equivalent generated-answer quality, while HotpotQA remained inconclusive.

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
