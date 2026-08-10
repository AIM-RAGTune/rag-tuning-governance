# Fresh Live CRAG Unblock Post-Validation

## Validation

- `python3 scripts/validate_publication_bundle.py`: passed
- `pytest -q tests/publication`: 34 passed
- `make validate-publication`: passed
- `make test`: 34 passed
- `python3 -m compileall src scripts`: passed

## Publication Hygiene

- Large-file scan: passed; no non-git, non-local-cache files over 50 MB.
- Raw data/query/API-response scan: passed with expected references only. Hits were sanitizer code, validator patterns, sanitized audit inventories, and synthetic test fixtures; no fresh CRAG raw query text, raw source documents, or raw API responses were committed.
- Secret scan: passed with expected scanner-pattern references only; no candidate secret values were found.
- Private-path scan: passed; no private local paths were found in committed files after excluding gitignored local caches.
- Overclaim scan: passed with expected unsupported-claim statements only.

## Fresh CRAG Result

- Evidence class: `fresh_live_crag_mock_api_blocked`
- Result class: `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`
- Approval env var: present during run
- CRAG root configured: true
- CRAG data configured: true
- Local evaluator available: true
- Mock API path available: true
- Mock API runtime available: false
- Mock API blocker: `open_kg_unreadable:OSError`

No raw CRAG data, query wording, source documents, or mock-API responses were copied or exported.

## Multi-Dataset Synthesis

- Prior frozen CRAG: `GOVERNANCE_REDUCES_COST_AT_EQUIVALENT_QUALITY`
- Fresh CRAG: `FRESH_CRAG_BLOCKED_MOCK_API_NOT_AVAILABLE`
- HotpotQA: `HOTPOTQA_GOVERNANCE_OPERATIONAL_GAIN_QUALITY_LOSS`
- Synthesis result: `MULTI_DATASET_BEHAVIORAL_GOVERNANCE_INCONCLUSIVE`

## GitHub Actions Readiness

Local publication gates passed. The branch is ready for push and PR creation; GitHub Actions should run the same publication validator and publication tests.
