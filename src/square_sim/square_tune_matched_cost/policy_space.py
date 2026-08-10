from __future__ import annotations

RAG_POLICY_SPACE = {
    "chunk_size": [256, 512, 1024],
    "chunk_overlap": [0, 64, 128],
    "top_k": [3, 5, 8, 12],
    "reranker_enabled": [False, True],
    "reranker_top_n": [3, 5, 8],
    "citation_required": [False, True],
    "source_trust_filter": ["none", "moderate", "strict"],
    "freshness_bias": [0.0, 0.25, 0.5],
    "abstention_threshold": [0.25, 0.45, 0.65],
    "retrieval_retry_policy": ["none", "low_confidence", "conflict"],
    "claim_level_check_enabled": [False, True],
    "context_compression_enabled": [False, True],
    "answer_length_policy": ["short", "balanced", "complete"],
    "model_strength_policy": ["standard", "strong_on_hard_cases"],
    "verification_policy": ["none", "citation_check", "claim_check"],
}

