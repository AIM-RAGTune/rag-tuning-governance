from __future__ import annotations

RAG_POLICY_SPACE = {
    "chunk_size": [256, 512, 768],
    "chunk_overlap": [32, 64, 128],
    "top_k": [3, 5, 8],
    "reranker_enabled": [False, True],
    "citation_required": [False, True],
    "source_trust_filter": ["none", "strict"],
    "freshness_bias": [0.0, 0.4],
    "abstention_threshold": [0.35, 0.55, 0.75],
    "claim_level_check_enabled": [False, True],
    "retrieval_retry_policy": ["none", "uncertain_only"],
    "answer_length_policy": ["compact", "complete"],
}
