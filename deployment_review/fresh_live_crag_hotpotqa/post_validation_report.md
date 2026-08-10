# Fresh Live CRAG + HotpotQA Post-Validation

## Results

- Fresh live CRAG: `FRESH_CRAG_BLOCKED_NO_APPROVED_DATA`
- HotpotQA: `HOTPOTQA_BLOCKED_DATASET_UNAVAILABLE`
- Multi-dataset synthesis: `MULTI_DATASET_BEHAVIORAL_GOVERNANCE_BLOCKED`

## Validation

- `python3 scripts/validate_publication_bundle.py`: passed
- `pytest`: 32 passed
- `make validate-publication`: passed
- `make test`: 32 passed
- `python3 -m compileall src scripts`: passed
- Large-file scan: passed; no files over 50 MB outside `.git`

## Publication Hygiene

The raw-text scan produced expected references in documentation, tests, validators, source code that redacts raw fields, and prior sanitization reports. No new raw CRAG query text, HotpotQA questions, source documents, supporting-fact text, or raw API responses were added.

## Claim Boundary

This phase adds the fresh/live and HotpotQA harnesses, but it does not replicate the prior frozen-observation result because approved local CRAG and HotpotQA data were unavailable.
