from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from square_sim.tune.config import DEFAULT_OBJECTIVE_WEIGHTS, EVAL_METRICS


@dataclass
class CostState:
    simulated_gpu_hours: float = 0.0
    wall_clock_estimate_seconds: float = 0.0
    data_volume: int = 0
    token_cost: float = 0.0
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TuneState:
    eval_vector: dict[str, float]
    failure_clusters: dict[str, float]
    data_pool_summary: dict[str, Any]
    adapter_state: dict[str, Any]
    prompt_policy_state: dict[str, Any]
    rag_policy_state: dict[str, Any]
    tool_policy_state: dict[str, Any]
    memory_state: dict[str, Any] = field(default_factory=lambda: {"known_good": [], "known_bad": []})
    cost_state: CostState = field(default_factory=CostState)
    snapshot_metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> TuneState:
        return TuneState(
            eval_vector=dict(self.eval_vector),
            failure_clusters=dict(self.failure_clusters),
            data_pool_summary=dict(self.data_pool_summary),
            adapter_state=dict(self.adapter_state),
            prompt_policy_state=dict(self.prompt_policy_state),
            rag_policy_state=dict(self.rag_policy_state),
            tool_policy_state=dict(self.tool_policy_state),
            memory_state={
                "known_good": list(self.memory_state.get("known_good", [])),
                "known_bad": list(self.memory_state.get("known_bad", [])),
            },
            cost_state=CostState(**self.cost_state.to_dict()),
            snapshot_metadata=dict(self.snapshot_metadata),
        )

    def utility(self, weights: dict[str, float] | None = None) -> float:
        weights = weights or DEFAULT_OBJECTIVE_WEIGHTS
        value = 0.0
        for metric, weight in weights.items():
            metric_value = self.eval_vector.get(metric, 0.0)
            if metric in {"cost", "latency"} and weight > 0:
                metric_value = 1.0 - metric_value
            value += weight * metric_value
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_vector": dict(self.eval_vector),
            "failure_clusters": dict(self.failure_clusters),
            "data_pool_summary": dict(self.data_pool_summary),
            "adapter_state": dict(self.adapter_state),
            "prompt_policy_state": dict(self.prompt_policy_state),
            "rag_policy_state": dict(self.rag_policy_state),
            "tool_policy_state": dict(self.tool_policy_state),
            "memory_state": dict(self.memory_state),
            "cost_state": self.cost_state.to_dict(),
            "snapshot_metadata": dict(self.snapshot_metadata),
        }


def initial_state_from_frame(df: pd.DataFrame, seed: int) -> TuneState:
    cluster_risk = df.groupby("failure_cluster")["feature_failure_severity"].mean().to_dict()
    eval_vector = {metric: 0.50 for metric in EVAL_METRICS}
    eval_vector.update(
        {
            "domain_accuracy": float(0.45 + 0.20 * df["feature_data_quality"].mean()),
            "retrieval_faithfulness": float(0.50 + 0.15 * (1 - df["feature_retrieval_ambiguity"].mean())),
            "instruction_following": float(0.48 + 0.15 * df["feature_style_specificity"].mean()),
            "style_match": float(0.48 + 0.10 * df["feature_style_specificity"].mean()),
            "safety": float(0.78 - 0.20 * df["feature_safety_sensitivity"].mean()),
            "latency": float(min(1.0, 0.25 + 0.20 * df["cost_weight"].mean())),
            "cost": float(min(1.0, 0.20 + 0.15 * df["cost_weight"].mean())),
            "calibration": float(0.50 + 0.10 * (1 - df["feature_instruction_conflict"].mean())),
            "regression_score": float(0.86 - 0.30 * df["feature_regression_risk"].mean()),
        }
    )
    return TuneState(
        eval_vector={k: float(max(0.0, min(1.0, v))) for k, v in eval_vector.items()},
        failure_clusters={k: float(v) for k, v in cluster_risk.items()},
        data_pool_summary={
            "rows": len(df),
            "mean_quality": float(df["feature_data_quality"].mean()),
            "mean_regression_risk": float(df["feature_regression_risk"].mean()),
            "mean_cost_weight": float(df["cost_weight"].mean()),
        },
        adapter_state={"rank": 8, "learning_rate": 1e-4, "epochs": 1, "regularization": 0.1},
        prompt_policy_state={"constraint_strength": 0.5, "citation_requirement": 0.5, "refusal_style": "balanced"},
        rag_policy_state={"chunk_size": 512, "overlap": 64, "top_k": 5, "reranker_enabled": False},
        tool_policy_state={"threshold": 0.55, "fallback_strategy": "abstain", "confidence_gate": 0.60},
        snapshot_metadata={"seed": seed, "step_index": 0},
    )
