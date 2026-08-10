from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SnapshotState:
    """Differentiable computational snapshot of a local field substrate."""

    field_patch: torch.Tensor
    emitter_activations: torch.Tensor
    step_index: int
    region_descriptor: dict[str, Any]
    snapshot_id: str
    global_field_context: torch.Tensor | None = None
    emitter_patch_indices: torch.Tensor | list[int] | None = None
    zone_masks: torch.Tensor | None = None
    zone_readout: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    feedback_state: torch.Tensor | None = None
    encoded_features: torch.Tensor | None = None
    local_gradients: torch.Tensor | None = None
    local_energy: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def clone_for_branching(self) -> SnapshotState:
        return SnapshotState(
            field_patch=self.field_patch.clone(),
            emitter_activations=self.emitter_activations.clone(),
            step_index=self.step_index,
            region_descriptor=dict(self.region_descriptor),
            snapshot_id=self.snapshot_id,
            global_field_context=None if self.global_field_context is None else self.global_field_context.clone(),
            emitter_patch_indices=self.emitter_patch_indices,
            zone_masks=None if self.zone_masks is None else self.zone_masks.clone(),
            zone_readout=None if self.zone_readout is None else self.zone_readout.clone(),
            memory_state=None if self.memory_state is None else self.memory_state.clone(),
            feedback_state=None if self.feedback_state is None else self.feedback_state.clone(),
            encoded_features=None if self.encoded_features is None else self.encoded_features.clone(),
            local_gradients=None if self.local_gradients is None else self.local_gradients.clone(),
            local_energy=None if self.local_energy is None else self.local_energy.clone(),
            diagnostics=dict(self.diagnostics),
        )

    def detach_copy(self) -> SnapshotState:
        copied = self.clone_for_branching()
        for key, value in list(copied.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(copied, key, value.detach().clone())
        return copied

    def to_device(self, device: torch.device | str) -> SnapshotState:
        for key, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self

    def summary_stats(self) -> dict[str, float | int | str]:
        patch = self.field_patch.detach()
        return {
            "snapshot_id": self.snapshot_id,
            "step_index": self.step_index,
            "field_patch_dims": len(patch.shape),
            "field_patch_mean": float(patch.mean().cpu()),
            "field_patch_std": float(patch.std(unbiased=False).cpu()),
            "field_patch_abs_max": float(patch.abs().max().cpu()),
            "estimated_memory_bytes": self.estimate_memory_bytes(),
        }

    def validate_shapes(self) -> None:
        if self.field_patch.ndim not in {5, 6}:
            raise ValueError(
                "Snapshot field_patch must be [batch, regions, channels, h, w] "
                "or [batch, regions, branches, channels, h, w]."
            )
        if self.emitter_activations.ndim != 2:
            raise ValueError("emitter_activations must be [batch, emitters].")
        if self.field_patch.shape[0] != self.emitter_activations.shape[0]:
            raise ValueError("field_patch and emitter_activations batch dimensions differ.")

    def estimate_memory_bytes(self) -> int:
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return int(total)


@dataclass(frozen=True)
class BranchScores:
    raw_scores: torch.Tensor
    weights: torch.Tensor
    score_features: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MergeResult:
    field_update: torch.Tensor | None
    readout_features: torch.Tensor
    diagnostics: dict[str, Any]
