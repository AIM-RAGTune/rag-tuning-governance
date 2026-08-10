from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateAction:
    action_type: str
    cluster: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type, "cluster": self.cluster, "params": dict(self.params)}


ACTION_TYPES = [
    "add_training_examples",
    "exclude_training_examples",
    "change_adapter_config",
    "change_prompt_policy",
    "change_rag_policy",
    "change_tool_policy",
    "change_curriculum",
    "merge_adapters_or_policies",
]


def make_action_space(clusters: list[str], rng: np.random.Generator, count: int = 48) -> list[CandidateAction]:
    actions: list[CandidateAction] = []
    for idx in range(count):
        action_type = ACTION_TYPES[idx % len(ACTION_TYPES)]
        cluster = str(clusters[idx % len(clusters)]) if clusters else "global"
        params = {
            "count": int(rng.integers(32, 512)),
            "quality_threshold": float(rng.uniform(0.25, 0.85)),
            "diversity_weight": float(rng.uniform(0.0, 1.0)),
            "risk_tolerance": float(rng.uniform(0.05, 0.8)),
            "rank": int(rng.choice([4, 8, 16, 32])),
            "learning_rate": float(rng.choice([5e-5, 1e-4, 2e-4, 5e-4])),
            "epochs": int(rng.choice([1, 2, 3])),
            "regularization": float(rng.uniform(0.0, 0.4)),
            "constraint_strength": float(rng.uniform(0.2, 0.9)),
            "top_k": int(rng.choice([3, 5, 8, 12])),
            "reranker": bool(rng.integers(0, 2)),
            "threshold": float(rng.uniform(0.35, 0.85)),
            "order_strategy": str(rng.choice(["easy_to_hard", "interleaved", "hard_cases_first"])),
        }
        actions.append(CandidateAction(action_type, cluster, params))
    return actions

