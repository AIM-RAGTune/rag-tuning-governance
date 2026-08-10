from __future__ import annotations

CLAIM_FAITHFULNESS_POLICIES = {
    "claim_level_check_enabled": [False, True],
    "citation_required": [True],
    "abstention_threshold": [0.45, 0.65, 0.80],
    "retrieval_retry_policy": ["none", "unsupported_claim_only", "uncertain_only"],
}
