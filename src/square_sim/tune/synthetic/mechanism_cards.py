from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MechanismSpec:
    key: str
    mechanism_name: str
    purpose: str
    ground_truth: str
    expected_winner: str
    expected_losing_ablations: list[str]
    control_type: str


MECHANISMS: dict[str, MechanismSpec] = {
    "synthetic_llm_linear_control": MechanismSpec(
        "synthetic_llm_linear_control",
        "linear_control",
        "Negative control for simple linear adaptation decisions.",
        "Eval utility is mostly a linear function of example quality and difficulty.",
        "greedy_eval_improvement",
        ["oracle_upper_bound"],
        "classical_control",
    ),
    "synthetic_llm_random_label": MechanismSpec(
        "synthetic_llm_random_label",
        "random_label",
        "Refusal control with no learnable adaptation signal.",
        "Targets are seeded random labels independent of observable features.",
        "none",
        ["all_non_oracle_methods"],
        "refusal_control",
    ),
    "synthetic_llm_failure_cluster_routing": MechanismSpec(
        "synthetic_llm_failure_cluster_routing",
        "failure_cluster_routing",
        "Tests localized failure-region selection.",
        "Interventions are useful only when matched to the local failure cluster.",
        "square_tune_full",
        ["square_tune_no_snapshot", "square_tune_global_only", "square_tune_random_branch"],
        "positive_control",
    ),
    "synthetic_llm_nonmonotonic_data_mix": MechanismSpec(
        "synthetic_llm_nonmonotonic_data_mix",
        "nonmonotonic_data_mix",
        "Tests nonlinear data-curation effects and diminishing returns.",
        "Over-concentrating one failure mode creates regressions after an optimum.",
        "square_tune_full",
        ["square_tune_linear_rollout", "square_tune_no_regression_sensor"],
        "positive_control",
    ),
    "synthetic_llm_adapter_tradeoff": MechanismSpec(
        "synthetic_llm_adapter_tradeoff",
        "adapter_tradeoff",
        "Tests adapter selection under multi-objective tradeoffs.",
        "Rank and learning-rate choices improve some metrics while hurting protected metrics.",
        "square_tune_full",
        ["square_tune_no_cost_sensor", "square_tune_no_regression_sensor"],
        "positive_control",
    ),
    "synthetic_llm_rag_policy_conflict": MechanismSpec(
        "synthetic_llm_rag_policy_conflict",
        "rag_policy_conflict",
        "Tests retrieval policy conflicts.",
        "Higher recall can reduce faithfulness and increase latency unless balanced.",
        "square_tune_full",
        ["square_tune_linear_rollout", "square_tune_no_cost_sensor"],
        "positive_control",
    ),
    "synthetic_llm_prompt_regression": MechanismSpec(
        "synthetic_llm_prompt_regression",
        "prompt_regression",
        "Tests prompt changes that fix one behavior while regressing another.",
        "Instruction pressure improves format following but may harm safety and nuance.",
        "square_tune_full",
        ["square_tune_no_memory", "square_tune_no_regression_sensor"],
        "positive_control",
    ),
    "synthetic_llm_tool_routing": MechanismSpec(
        "synthetic_llm_tool_routing",
        "tool_routing",
        "Tests local tool-selection policies.",
        "Correct routing depends on task cluster and confidence threshold.",
        "square_tune_full",
        ["square_tune_global_only", "square_tune_random_branch"],
        "positive_control",
    ),
    "synthetic_llm_data_poison_regression": MechanismSpec(
        "synthetic_llm_data_poison_regression",
        "data_poison_regression",
        "Tests detection of locally tempting but globally harmful examples.",
        "Some examples improve local metrics but carry latent regression risk.",
        "square_tune_full",
        ["square_tune_no_regression_sensor", "square_tune_no_memory"],
        "positive_control",
    ),
    "synthetic_llm_merge_required": MechanismSpec(
        "synthetic_llm_merge_required",
        "merge_required",
        "Tests merge and reintegration.",
        "No single branch solves all eval dimensions; weighted merge is required.",
        "square_tune_full",
        ["square_tune_no_merge", "square_tune_no_fork"],
        "positive_control",
    ),
    "synthetic_llm_curriculum_order": MechanismSpec(
        "synthetic_llm_curriculum_order",
        "curriculum_order",
        "Tests order-dependent adaptation.",
        "The same examples produce different outcomes depending on curriculum order.",
        "square_tune_full",
        ["square_tune_no_memory", "greedy_eval_improvement"],
        "positive_control",
    ),
    "synthetic_llm_hard_external_transfer_proxy": MechanismSpec(
        "synthetic_llm_hard_external_transfer_proxy",
        "hard_external_transfer_proxy",
        "Tests conservative adaptation under domain shift.",
        "A source-domain tuning path can overfit and regress held-out domain behavior.",
        "square_tune_full",
        ["square_tune_no_regression_sensor", "square_tune_no_feedback"],
        "external_transfer_proxy",
    ),
    "synthetic_llm_repeated_regression_memory": MechanismSpec(
        "synthetic_llm_repeated_regression_memory",
        "repeated_regression_memory",
        "Tests whether memory prevents repeated damaging interventions.",
        "Known-bad actions reappear through cluster aliases; memory should reduce repeated regressions.",
        "square_tune_full",
        ["square_tune_no_memory", "square_tune_no_regression_sensor"],
        "positive_control",
    ),
    "synthetic_llm_regression_veto": MechanismSpec(
        "synthetic_llm_regression_veto",
        "regression_veto",
        "Tests hard protected-metric veto behavior.",
        "Some high-raw-utility actions violate safety or faithfulness constraints and should be rejected.",
        "square_tune_full",
        ["square_tune_no_regression_sensor"],
        "positive_control",
    ),
    "synthetic_llm_cost_tradeoff": MechanismSpec(
        "synthetic_llm_cost_tradeoff",
        "cost_tradeoff",
        "Tests cost-aware adaptation choices.",
        "Expensive branches can maximize raw utility while lower-cost branches win cost-adjusted utility.",
        "square_tune_full",
        ["square_tune_no_cost_sensor"],
        "positive_control",
    ),
}


def mechanism_card(spec: MechanismSpec) -> str:
    losing = ", ".join(spec.expected_losing_ablations)
    return f"""# {spec.key}

## What This Dataset Tests

{spec.purpose}

## What It Does Not Test

This synthetic mechanism diagnostic does not test physical SQUARE hardware and does not establish external LLM benchmark performance.

## Ground-Truth Mechanism

{spec.ground_truth}

## Expected Winner

{spec.expected_winner}

## Expected Losing Ablations

{losing}

## Refusal Conditions

Do not award a mechanism signal if the expected controls fail, if repeated seeds are unstable, or if the full optimizer does not beat the relevant ablations.

## Caveats

Synthetic diagnostics are mechanism checks. They can verify that a simulated mechanism can be load-bearing under known ground truth, but they do not upgrade external benchmark claims.

## Control Type

{spec.control_type}
"""
