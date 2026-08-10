# Reproduce CRAG Mock-API Validation

Prerequisites:

- Approved environment.
- CRAG data obtained from the original provider under the documented noncommercial restriction.
- Raw data mounted at `CRAG_RAW_DIR`, for example `/data/crag/raw`.

Command:

```bash
CRAG_RAW_DIR=/data/crag/raw make reproduce-crag
```

Expected parent result:

- `MOCK_API_VALIDATION_GOVERNANCE_SUPERIOR`
- Governed winner: `top_k_low`
- Quality-only winner: `greedy_regression_aware_search`
- Confirmatory rows: 571
- Failure rate: 0.0
