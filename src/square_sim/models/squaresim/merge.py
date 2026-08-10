from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from square_sim.models.squaresim.snapshot import BranchScores, MergeResult, SnapshotState


@dataclass(frozen=True)
class MergeConfig:
    enabled: bool = True
    strategy: str = "weighted_readout"
    update_main_field: bool = False
    update_readout: bool = True
    residual_scale: float = 0.1
    preserve_identity_path: bool = True
    merge_temperature: float = 1.0


class BranchMerger(nn.Module):
    def __init__(self, config: MergeConfig):
        super().__init__()
        self.config = config

    def _branch_readouts(self, patch: torch.Tensor) -> torch.Tensor:
        mean = patch.mean(dim=(-1, -2))
        maxv = patch.amax(dim=(-1, -2))
        energy = patch.square().mean(dim=(-1, -2))
        gy = patch[..., 1:, :] - patch[..., :-1, :]
        gx = patch[..., :, 1:] - patch[..., :, :-1]
        grad = gy.square().mean(dim=(-1, -2)) + gx.square().mean(dim=(-1, -2))
        return torch.cat([mean, maxv, energy, grad], dim=-1)

    def forward(
        self,
        *,
        main_field: torch.Tensor,
        rolled_branches: SnapshotState,
        branch_scores: BranchScores,
        zone_readout: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
    ) -> MergeResult:
        patch = rolled_branches.field_patch
        if patch.ndim != 6:
            raise ValueError("Branch merge expects [batch, regions, branches, channels, h, w].")
        readouts = self._branch_readouts(patch)
        weights = branch_scores.weights
        if not self.config.enabled or self.config.strategy == "no_merge":
            zero = torch.zeros(
                patch.shape[0],
                patch.shape[1] * readouts.shape[-1],
                device=patch.device,
                dtype=patch.dtype,
            )
            return MergeResult(None, zero, {"merge_strategy": "no_merge", "merge_contribution": 0.0})
        if self.config.strategy == "best_branch":
            index = weights.argmax(dim=-1)
            gathered = readouts.gather(
                2,
                index[:, :, None, None].expand(-1, -1, 1, readouts.shape[-1]),
            ).squeeze(2)
            weighted_readout = gathered
        elif self.config.strategy == "random_merge":
            random_weights = torch.softmax(torch.randn_like(weights), dim=-1)
            weighted_readout = (readouts * random_weights.unsqueeze(-1)).sum(dim=2)
        else:
            weighted_readout = (readouts * weights.unsqueeze(-1)).sum(dim=2)
        field_update = None
        if self.config.update_main_field:
            weighted_patch = (patch * weights[:, :, :, None, None, None]).sum(dim=2)
            if rolled_branches.zone_masks is not None and weighted_patch.shape[-2:] == main_field.shape[-2:]:
                update = (weighted_patch * rolled_branches.zone_masks.unsqueeze(2)).sum(dim=1)
                field_update = main_field + self.config.residual_scale * update
        contribution = float(weighted_readout.detach().abs().mean().cpu())
        return MergeResult(
            field_update=field_update,
            readout_features=weighted_readout.flatten(1),
            diagnostics={"merge_strategy": self.config.strategy, "merge_contribution": contribution},
        )


def merge_config_from_dict(payload: dict[str, Any] | None) -> MergeConfig:
    payload = payload or {}
    return MergeConfig(
        enabled=bool(payload.get("enabled", True)),
        strategy=str(payload.get("strategy", "weighted_readout")),
        update_main_field=bool(payload.get("update_main_field", False)),
        update_readout=bool(payload.get("update_readout", True)),
        residual_scale=float(payload.get("residual_scale", 0.1)),
        preserve_identity_path=bool(payload.get("preserve_identity_path", True)),
        merge_temperature=float(payload.get("merge_temperature", 1.0)),
    )
