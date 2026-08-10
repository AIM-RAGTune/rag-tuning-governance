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
