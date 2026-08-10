from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from square_sim.models.squaresim.snapshot import BranchScores, SnapshotState


@dataclass(frozen=True)
class BranchScoringConfig:
    mode: str = "learned_score"
    hidden_dim: int = 64
    include_energy_features: bool = True
    include_gradient_features: bool = True
    temperature: float = 1.0
    normalize_scores: bool = True


class BranchScorer(nn.Module):
    def __init__(self, config: BranchScoringConfig, *, encoded_dim: int, channels: int):
        super().__init__()
        self.config = config
        self.feature_dim = channels * 4 + encoded_dim
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _features(self, state: SnapshotState) -> torch.Tensor:
        patch = state.field_patch
        if patch.ndim != 6:
            raise ValueError("Branch scoring expects [batch, regions, branches, channels, h, w].")
        mean = patch.mean(dim=(-1, -2))
        maxv = patch.amax(dim=(-1, -2))
        energy = patch.square().mean(dim=(-1, -2))
        gy = patch[..., 1:, :] - patch[..., :-1, :]
        gx = patch[..., :, 1:] - patch[..., :, :-1]
        grad = (gy.square().mean(dim=(-1, -2)) + gx.square().mean(dim=(-1, -2)))
        features = [mean, maxv, energy, grad]
        out = torch.cat(features, dim=-1)
        if state.encoded_features is not None:
            encoded = state.encoded_features[:, None, None, :].expand(-1, patch.shape[1], patch.shape[2], -1)
            out = torch.cat([out, encoded], dim=-1)
        else:
            zeros = torch.zeros(*out.shape[:-1], self.feature_dim - out.shape[-1], device=out.device, dtype=out.dtype)
            out = torch.cat([out, zeros], dim=-1)
        return out

    def forward(self, state: SnapshotState) -> BranchScores:
        features = self._features(state)
        if self.config.mode == "random_score":
            raw = torch.randn(*features.shape[:-1], device=features.device, dtype=features.dtype)
        elif self.config.mode == "energy_stability_score":
            energy = state.field_patch.square().mean(dim=(-1, -2, -3))
            raw = -torch.abs(energy - energy.mean(dim=-1, keepdim=True))
        else:
            raw = self.net(features).squeeze(-1)
        if self.config.normalize_scores:
            weights = torch.softmax(raw / max(self.config.temperature, 1e-6), dim=-1)
        else:
            weights = raw
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return BranchScores(
            raw_scores=raw,
            weights=weights,
            score_features=features,
            diagnostics={
                "branch_entropy_mean": float(entropy.detach().cpu()),
                "branch_weight_max_mean": float(weights.max(dim=-1).values.detach().mean().cpu()),
                "branch_score_mean": float(raw.detach().mean().cpu()),
            },
        )


def branch_scoring_config_from_dict(payload: dict[str, Any] | None) -> BranchScoringConfig:
    payload = payload or {}
    return BranchScoringConfig(
        mode=str(payload.get("mode", "learned_score")),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        include_energy_features=bool(payload.get("include_energy_features", True)),
        include_gradient_features=bool(payload.get("include_gradient_features", True)),
        temperature=float(payload.get("temperature", 1.0)),
        normalize_scores=bool(payload.get("normalize_scores", True)),
    )
