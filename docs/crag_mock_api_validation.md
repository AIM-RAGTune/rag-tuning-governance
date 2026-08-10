# CRAG Mock-API Validation

Parent run: `ragtune_crag_mock_api_validation_v1_20260809-165415-92d8c0edd4`

Result: `MOCK_API_VALIDATION_GOVERNANCE_SUPERIOR`

Key facts:

- Validation rows: 431 / 431
- Confirmatory rows: 571 / 571
- API calls: 14,172
- Failure rate: 0.0
- Governed winner: `top_k_low`
- Quality-only winner: `greedy_regression_aware_search`
- RAG Compass rank: 5th
- Governance delta: +0.0010025405
- Bootstrap CI: [0.0010022708, 0.0010028250]
- Win/tie/loss: 571 / 0 / 0
- Sensitivity: governance superior in 14 / 15 settings

Interpretation: this supports RAGTune governance value more directly than RAG Compass optimizer superiority.

CRAG raw data, raw question wording, raw source passages, and raw mock-API responses are not included here. Case explanations use sanitized summaries and query hashes rather than CRAG wording. Reproduction requires obtaining CRAG from the original approved source, verifying expected hashes, and respecting the noncommercial-research-only restriction used by the local validation. The CRAG mock-API result supports source/retrieval governance evidence under the configured utility, not generative LLM answer-quality validation. Commercial use requires separate license and legal review.

## Behaviorally Distinct Follow-Up

The follow-up suite `ragtune_behavioral_governance_primary_outcome_v1` reuses the sanitized frozen CRAG mock-API observations and compares genuinely different operating behaviors: one-endpoint low retrieval, two-endpoint expanded retrieval, variable-call adaptive routing, measured-cost selection, measured-latency selection, quality-only selection, constrained optimization, and Pareto frontier selection.

Result: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`.

- Governed winner: `low_retrieval_single_endpoint`
- Quality-only winner: `optuna_tpe`
- Quality metric: `QUALITY_MEASURE_PROXY_PLUS_EVIDENCE`
- Quality noninferiority margin: 0.01
- Evidence class: `public_full_corpus_mock_api_validation_derived_frozen_observation`

This follow-up avoids relying only on a small weighted-utility delta by using a predeclared quality floor and measured cost/latency outcomes. It remains bounded: no raw CRAG text is included, no new live API collection is claimed, and no human/generative validation is claimed.

## Fresh Live CRAG Attempt

The fresh live CRAG phase adds `ragtune_fresh_live_crag_mock_api_behavioral_governance_v1`. In this execution environment, approved local CRAG data and the mock-API KG/runtime were restored, and a 50-example sanitized live sample ran. The result is `FRESH_CRAG_BLOCKED_QUALITY_MEASURE_PROXY_ONLY`: endpoint behavior, API calls, latency, and cost were measured, but the sample did not produce a usable answer/evidence quality signal. This is a blocked result, not a replication claim.

To run it in an approved environment, set `RAGTUNE_CRAG_APPROVED_NONCOMMERCIAL_RESEARCH_ONLY=true`, `RAGTUNE_CRAG_ROOT`, and `RAGTUNE_CRAG_DATA`; verify the mock-API KG files are complete and readable; start the CRAG mock API; then run `scripts/run_fresh_live_crag_behavioral_governance.py`.

## CRAG Generative LLM Status

`ragtune_crag_generative_llm_validation_v1` is implemented as a sanitized harness, but the public artifact set does not include a completed CRAG generated-answer quality run. CRAG generative validation remains blocked or incomplete until a publication-safe evaluator mapping can score generated answers without exporting CRAG question text, source documents, raw API responses, or raw generated answers.
