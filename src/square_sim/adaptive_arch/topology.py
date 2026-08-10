from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchitectureState:
    active_regions: list[str]
    local_policies: dict[str, str]
    topology_graph: dict[str, list[str]]
    memory_state: dict[str, list[str]] = field(default_factory=lambda: {"known_good": [], "known_bad": []})
    compute_budget: dict[str, float] = field(default_factory=dict)
    current_regime: str = "initial"
    uncertainty_map: dict[str, float] = field(default_factory=dict)
    conflict_map: dict[str, float] = field(default_factory=dict)
    risk_map: dict[str, float] = field(default_factory=dict)
    historical_adaptations: list[dict[str, object]] = field(default_factory=list)
    pending_rollouts: list[dict[str, object]] = field(default_factory=list)
    merge_candidates: list[dict[str, object]] = field(default_factory=list)

