from __future__ import annotations

ML_TO_LLM_POLICY_SPACE = {
    "use_classical_prediction": [True],
    "use_llm_explanation": [False, True],
    "use_rag_evidence": [False, True],
    "use_action_recommendation": [False, True],
    "explanation_required_threshold": [0.45, 0.65],
    "exception_threshold": [0.35, 0.55, 0.75],
    "abstention_threshold": [0.45, 0.65],
    "human_review_threshold": [0.55, 0.75],
    "policy_retrieval_top_k": [2, 4, 6],
    "cost_budget": [0.3, 0.6, 1.0],
}
