## Summary

Runs the fresh live CRAG mock-API behaviorally distinct governance validation using approved local CRAG data/mock-API paths where available, and preserves the blocked status because the mock-API runtime could not read required KG data.

## CRAG acquisition status

- Approval env var: present during local run
- CRAG root configured: true
- CRAG data configured: true
- Mock API path available: true
- Mock API runtime available: false
- Local evaluator available: true
- Raw CRAG data committed: no
- Raw query text exported: no
- Raw API responses exported: no
- Raw source documents exported: no

## Fresh CRAG result

- Evidence class: `fresh_live_crag_mock_api_blocked`
- Result class: `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`
- Governed winner: not available
- Quality-only winner: not available
- Constrained optimizer winner: not available
- Pareto frontier: not available
- RAG Compass rank: not available
- Quality metric class: not available
- Final quality delta: not available
- Evidence-support delta: not available
- Cost delta: not available
- Latency delta: not available
- API-call delta: not available

## Multi-dataset synthesis

- Prior frozen CRAG: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`
- Fresh CRAG: `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`
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
