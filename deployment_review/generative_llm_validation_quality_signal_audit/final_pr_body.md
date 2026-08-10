## Summary

Reruns HotpotQA generative validation at a larger bounded sample size, audits why the initial generated-quality deltas were all zero, repairs CRAG generator access, and attempts CRAG generated-answer evaluator mapping.

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
- Evaluator mapping result class: `CRAG_GENERATED_QUALITY_BLOCKED_NO_USABLE_SIGNAL`
- CRAG generative result class: `GEN_LLM_VALIDATION_BLOCKED_NO_USABLE_QUALITY_SIGNAL_CRAG`
- Governed winner: `governed_selection`
- Quality-only winner: `adaptive_routing_on_insufficient_evidence`
- RAG Compass rank: 8
- Generated-quality delta: 0.0 [0.0, 0.0]
- Evidence-support delta: 0.0 [0.0, 0.0]
- Cost delta: -1.242 [-1.3106, -1.2904]
- Latency delta: -37.26543020457029 [-126.12729147076607, 3.849375993013382]

## Synthesis

- Result class: `GEN_LLM_SYNTHESIS_INCONCLUSIVE`
- Interpretation: HotpotQA confirmed a usable nonconstant generated-answer quality signal but did not show governance improvement; CRAG generator access was repaired but generated answers remained empty, so CRAG generative validation remains blocked.

## Validation

- publication validator: see post-validation report
- pytest: see post-validation report
- make validate-publication: see post-validation report
- make test: see post-validation report
- compile: see post-validation report
- raw prompt/generated-answer scan: expected sanitizer/test/field-name references only
- raw dataset text scan: expected sanitizer/test/field-name references only
- secret scan: expected scanner/config variable-name references only
- private path scan: no committed private paths
- large-file scan: no tracked public files over threshold

## Claim boundaries

This PR does not claim official platform benchmarking, human validation, production readiness, hallucination elimination, broad universal governance superiority, or RAG Compass superiority.
