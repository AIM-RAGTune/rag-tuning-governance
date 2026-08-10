from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from square_sim.models.squaresim.region_selectors import make_region_selector
from square_sim.models.squaresim.snapshot import SnapshotState


@dataclass(frozen=True)
class SnapshotCaptureConfig:
    enabled: bool = True
    capture_policy: str = "fixed_steps"
    capture_steps: tuple[int, ...] = (2, 3)
    every_n: int = 1
    region_selector: str = "overlap"
    num_regions: int = 2
    patch_size: tuple[int, int] = (12, 12)
    include_global_context: bool = True
    include_memory_state: bool = True
    include_feedback_state: bool = True
    include_local_gradients: bool = True
    include_energy_proxy: bool = True
    detach_snapshots: bool = False

    def should_capture(self, step_index: int, final_step: int) -> bool:
        human_step = step_index + 1
        if not self.enabled:
            return False
        if self.capture_policy == "last":
            return step_index == final_step
        if self.capture_policy == "fixed_steps":
            return human_step in self.capture_steps
        if self.capture_policy == "every_n":
            return human_step % max(self.every_n, 1) == 0
        return step_index == final_step


class SnapshotCaptureEngine(nn.Module):
    def __init__(self, config: SnapshotCaptureConfig, *, grid_size: int, channels: int, seed: int = 42):
        super().__init__()
        self.config = config
        self.selector = make_region_selector(
            config.region_selector,
            grid_size=grid_size,
            channels=channels,
            num_regions=config.num_regions,
            patch_size=config.patch_size,
            seed=seed,
        )

    def _extract_patches(self, field: torch.Tensor, masks: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
        if coords is None:
            return field.unsqueeze(1) * masks.unsqueeze(2)
        patches = []
        for b in range(field.shape[0]):
            row = []
            for r in range(coords.shape[1]):
                y0, x0, y1, x1 = [int(v) for v in coords[b, r].tolist()]
                row.append(field[b, :, y0:y1, x0:x1])
            patches.append(torch.stack(row, dim=0))
        return torch.stack(patches, dim=0)

    def _local_gradients(self, patch: torch.Tensor) -> torch.Tensor:
        gy = patch[..., 1:, :] - patch[..., :-1, :]
        gx = patch[..., :, 1:] - patch[..., :, :-1]
        gy_energy = gy.square().mean(dim=(-1, -2))
        gx_energy = gx.square().mean(dim=(-1, -2))
        return gy_energy + gx_energy

    def capture(
        self,
        *,
        field: torch.Tensor,
        emitter_activations: torch.Tensor,
        step_index: int,
        zone_masks: torch.Tensor | None = None,
        zone_readout: torch.Tensor | None = None,
        memory_state: torch.Tensor | None = None,
        feedback_state: torch.Tensor | None = None,
        encoded_features: torch.Tensor | None = None,
    ) -> SnapshotState:
        selection = self.selector.select(field, zone_masks=zone_masks)
        patch = self._extract_patches(field, selection.masks, selection.crop_coords)
        global_context = None
        if self.config.include_global_context:
            global_context = torch.stack(
                [
                    field.mean(dim=(-1, -2)),
                    field.square().mean(dim=(-1, -2)),
                    field.amax(dim=(-1, -2)),
                    field.amin(dim=(-1, -2)),
                ],
                dim=-1,
            ).flatten(1)
        gradients = self._local_gradients(patch) if self.config.include_local_gradients else None
        energy = patch.square().mean(dim=(-1, -2, -3)) if self.config.include_energy_proxy else None
        state = SnapshotState(
            field_patch=patch,
            emitter_activations=emitter_activations,
            step_index=step_index,
            region_descriptor={
                "mode": self.config.region_selector,
                "region_names": selection.region_names,
                "crop_coords": None if selection.crop_coords is None else selection.crop_coords.detach().cpu().tolist(),
            },
            snapshot_id=f"step-{step_index}-{self.config.region_selector}",
            global_field_context=global_context,
            zone_masks=selection.masks,
            zone_readout=zone_readout,
            memory_state=memory_state if self.config.include_memory_state else None,
            feedback_state=feedback_state if self.config.include_feedback_state else None,
            encoded_features=encoded_features,
            local_gradients=gradients,
            local_energy=energy,
            diagnostics={
                "region_scores_mean": float(selection.region_scores.detach().mean().cpu()),
                "selector": self.config.region_selector,
            },
        )
        state.validate_shapes()
        return state.detach_copy() if self.config.detach_snapshots else state


def capture_config_from_dict(payload: dict[str, Any] | None) -> SnapshotCaptureConfig:
    payload = payload or {}
    steps = payload.get("capture_steps", [2, 3])
    patch = payload.get("patch_size", [12, 12])
    return SnapshotCaptureConfig(
        enabled=bool(payload.get("enabled", True)),
        capture_policy=str(payload.get("capture_policy", "fixed_steps")),
        capture_steps=tuple(int(v) for v in steps),
        every_n=int(payload.get("every_n", 1)),
        region_selector=str(payload.get("region_selector", "overlap")),
        num_regions=int(payload.get("num_regions", 2)),
        patch_size=(int(patch[0]), int(patch[1])) if isinstance(patch, list | tuple) else (int(patch), int(patch)),
        include_global_context=bool(payload.get("include_global_context", True)),
        include_memory_state=bool(payload.get("include_memory_state", True)),
        include_feedback_state=bool(payload.get("include_feedback_state", True)),
        include_local_gradients=bool(payload.get("include_local_gradients", True)),
        include_energy_proxy=bool(payload.get("include_energy_proxy", True)),
        detach_snapshots=bool(payload.get("detach_snapshots", False)),
    )
