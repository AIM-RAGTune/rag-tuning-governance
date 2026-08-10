## Summary

Restores approved local CRAG data and a readable mock-API KG/runtime, then runs a sanitized fresh live CRAG sample. The result remains blocked because the sample did not produce a usable answer/evidence quality signal.

## CRAG acquisition status

- Approval env var: present during local run
- CRAG root configured: true
- CRAG data configured: true
- Mock API path available: true
- Mock API runtime available: true
- Local evaluator available: true
- Examples loaded locally: 50
- Per-query policy rows: 550
- Raw CRAG data committed: no
- Raw query text exported: no
- Raw API responses exported: no
- Raw source documents exported: no

## Fresh CRAG result

- Evidence class: `fresh_live_crag_mock_api_sanitized_live_sample`
- Result class: `FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY`
- Governed winner: `measured_cost_minimizer_at_quality_floor`
- Quality-only winner: `adaptive_routing_on_insufficient_evidence`
- Constrained optimizer winner: `measured_cost_minimizer_at_quality_floor`
- Pareto frontier: `static_default_policy`
- RAG Compass rank: 10
- Quality metric class: `QUALITY_MEASURE_PROXY_PLUS_LOCAL_ANSWER_EVIDENCE`
- Final quality delta: 0.0 [0.0, 0.0]
- Evidence-support delta: 0.0 [0.0, 0.0]
- Cost delta: -1.0 [-1.0, -1.0]
- Latency delta: -0.035 ms approx
- API-call delta: -1.0 [-1.0, -1.0]

## Multi-dataset synthesis

- Prior frozen CRAG: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`
- Fresh CRAG: `FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY`
- HotpotQA: `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`
- Synthesis result: `MULTI_DATASET_BEHAVIORAL_GOVERNANCE_INCONCLUSIVE`

## Validation

- publication validator: passed
- pytest: 34 passed
- make validate-publication: passed
- make test: 34 passed
- compile: passed
- raw data scan: passed with expected sanitizer/test/audit references only
- secret scan: passed with expected scanner-pattern references only
- private path scan: passed
- large-file scan: passed

## Claim boundaries

This PR does not claim RAG Compass superiority, human validation, generative LLM validation, official platform benchmarking, production readiness, hallucination elimination, or broad universal governance superiority.
