# Fresh Live CRAG + HotpotQA Post-Validation

## Results

- Fresh live CRAG: `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`
- HotpotQA: `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`
- Multi-dataset synthesis: `MULTI_DATASET_BEHAVIORAL_GOVERNANCE_INCONCLUSIVE`

## HotpotQA Details

- Evidence class: `hotpotqa_public_corpus_behavioral_governance_sanitized_sample`
- Examples loaded locally: 1,000
- Validation rows: 262
- Confirmatory rows: 249
- Governed winner: `expanded_retrieval_multi_context`
- Quality-only winner: `bm25_high_k`
- Constrained optimizer winner: `rag_compass_optional`
- Quality metric class: `QUALITY_MEASURE_ANSWER_LABELS_PLUS_SUPPORTING_FACT_EVIDENCE`
- Result interpretation: governance reduced measured cost but the confirmatory quality delta crossed the predeclared `0.01` noninferiority margin, so this is a quality-loss result rather than a positive replication.

## Validation

- `python3 scripts/validate_publication_bundle.py`: passed
- `pytest`: 32 passed
- `make validate-publication`: passed
- `make test`: 32 passed
- `python3 -m compileall src scripts`: passed
- Tracked large-file scan: passed; no tracked files over 50 MB
- Local data cache: HotpotQA Arrow cache exists under gitignored `.local_data/` and is not tracked

## Publication Hygiene

The raw-text scan produced expected references in documentation, tests, validators, source code that redacts raw fields, and prior sanitization reports. No new raw CRAG query text, HotpotQA questions, source documents, supporting-fact text, or raw API responses were added.

## Claim Boundary

This phase replaces the blocked HotpotQA artifact with real alternate-corpus evidence, but it does not replicate the prior frozen-observation result. Fresh CRAG remains blocked because the configured local mock-API runtime could not read required KG data, and HotpotQA showed operational gain with quality loss.
