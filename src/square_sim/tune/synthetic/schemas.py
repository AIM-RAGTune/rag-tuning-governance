from __future__ import annotations

REQUIRED_COLUMNS = [
    "row_id",
    "dataset_key",
    "mechanism_name",
    "split",
    "seed",
    "task_type",
    "input_text",
    "reference_answer",
    "candidate_response_optional",
    "expected_behavior",
    "failure_cluster",
    "difficulty",
    "domain",
    "prompt_variant_id",
    "retrieval_policy_id",
    "adapter_policy_id",
    "tool_policy_id",
    "safety_label",
    "regression_group",
    "cost_weight",
    "ground_truth_score_vector_json",
    "target_improvement",
    "target_regression",
    "target_utility",
    "target_branch_success",
    "target_merge_success",
]

FEATURE_COLUMNS = [
    "feature_domain_complexity",
    "feature_failure_severity",
    "feature_retrieval_ambiguity",
    "feature_instruction_conflict",
    "feature_safety_sensitivity",
    "feature_style_specificity",
    "feature_tool_need",
    "feature_data_quality",
    "feature_example_novelty",
    "feature_duplication_risk",
    "feature_regression_risk",
    "feature_adapter_sensitivity",
    "feature_prompt_sensitivity",
    "feature_rag_sensitivity",
    "feature_curriculum_sensitivity",
]

LATENT_COLUMNS = [
    "latent_cluster_optimum",
    "latent_poison_flag",
    "latent_order_effect",
    "latent_merge_group",
]

