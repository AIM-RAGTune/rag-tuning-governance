# Promotion Decision Schema

`promotion_decision.json` is the machine-readable output for deployment jobs.

Required fields include:

- `schema_version`
- `timestamp_utc`
- `run_id`
- `suite`
- `result_class`
- `decision`
- `selected_policy`
- `baseline_policy`
- `decision_reason`
- `quality_gates`
- `cost_latency_gates`
- `risk_flags`
- `claim_boundaries`
- `artifact_uris`
- `validator_status`

Valid decisions are:

```text
PROMOTE
BLOCK
REJECT
INCONCLUSIVE
ERROR
```

The JSON schema lives at `schemas/promotion_decision.schema.json`. Decisions are intentionally conservative: blocked and inconclusive outcomes are valid scientific outcomes when quality signals, input evidence, or publication hygiene are insufficient.
