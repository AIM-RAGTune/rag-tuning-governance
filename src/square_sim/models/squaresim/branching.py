from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from square_sim.models.squaresim.snapshot import SnapshotState


@dataclass(frozen=True)
class BranchingConfig:
    enabled: bool = True
    num_branches: int = 4
    include_identity_branch: bool = True
    perturbation_types: tuple[str, ...] = ("gaussian_field", "emitter_delta", "learned_policy")
    perturbation_scale: float = 0.05
    branch_dropout: float = 0.0
    learned_policy_hidden_dim: int = 64
    clamp_branch_values: bool = True
    max_abs_field_value: float = 10.0
    random_branch: bool = False


class BranchGenerator(nn.Module):
    def __init__(self, config: BranchingConfig, *, encoded_dim: int, emitter_count: int):
        super().__init__()
        self.config = config
        self.emitter_policy = nn.Sequential(
            nn.Linear(encoded_dim, config.learned_policy_hidden_dim),
            nn.GELU(),
            nn.Linear(config.learned_policy_hidden_dim, emitter_count),
        )

    def _field_noise(self, base: torch.Tensor, branch: int) -> torch.Tensor:
        if self.config.random_branch:
            return torch.randn_like(base) * self.config.perturbation_scale
        scale = self.config.perturbation_scale * float(branch + 1) / max(self.config.num_branches, 1)
        return torch.sin(base * (branch + 1.0)) * scale

    def forward(self, snapshot: SnapshotState) -> SnapshotState:
        source = snapshot.clone_for_branching()
        patch = source.field_patch
        branches = 1 if not self.config.enabled else max(1, self.config.num_branches)
        branch_patches = []
        encoded = source.encoded_features
        learned_scalar = None
        if encoded is not None and "learned_policy" in self.config.perturbation_types:
            learned_delta = torch.tanh(self.emitter_policy(encoded)).mean(dim=-1).view(-1, 1, 1, 1, 1)
            learned_scalar = learned_delta * self.config.perturbation_scale
        for branch in range(branches):
            if branch == 0 and self.config.include_identity_branch:
                candidate = patch
            elif "gaussian_field" in self.config.perturbation_types or self.config.random_branch:
                candidate = patch + self._field_noise(patch, branch)
            else:
                candidate = patch
            if learned_scalar is not None and branch > 0:
                candidate = candidate + learned_scalar
            if "emitter_delta" in self.config.perturbation_types and branch > 0:
                emitter_shift = torch.tanh(source.emitter_activations).mean(dim=-1).view(-1, 1, 1, 1, 1)
                candidate = candidate + emitter_shift * (self.config.perturbation_scale / float(branch + 1))
            if self.config.clamp_branch_values:
                candidate = candidate.clamp(-self.config.max_abs_field_value, self.config.max_abs_field_value)
            branch_patches.append(candidate)
        source.field_patch = torch.stack(branch_patches, dim=2)
        source.diagnostics.update(
            {
                "num_branches": branches,
                "include_identity_branch": self.config.include_identity_branch,
                "branch_memory_bytes": source.estimate_memory_bytes(),
            }
        )
        source.validate_shapes()
        return source


def branching_config_from_dict(payload: dict[str, Any] | None) -> BranchingConfig:
    payload = payload or {}
    return BranchingConfig(
        enabled=bool(payload.get("enabled", True)),
        num_branches=int(payload.get("num_branches", 4)),
        include_identity_branch=bool(payload.get("include_identity_branch", True)),
        perturbation_types=tuple(str(v) for v in payload.get("perturbation_types", ["gaussian_field", "emitter_delta", "learned_policy"])),
        perturbation_scale=float(payload.get("perturbation_scale", 0.05)),
        branch_dropout=float(payload.get("branch_dropout", 0.0)),
        learned_policy_hidden_dim=int(payload.get("learned_policy_hidden_dim", 64)),
        clamp_branch_values=bool(payload.get("clamp_branch_values", True)),
        max_abs_field_value=float(payload.get("max_abs_field_value", 10.0)),
        random_branch=bool(payload.get("random_branch", False)),
    )
