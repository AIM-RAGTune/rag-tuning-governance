from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from square_sim.models.squaresim.snapshot import SnapshotState


@dataclass(frozen=True)
class RolloutConfig:
    enabled: bool = True
    steps: int = 4
    dt: float = 0.1
    dynamics: str = "nonlinear_field"
    nonlinear_term: str = "phi4_like"
    diffusion_coeff: float = 0.05
    damping: float = 0.01
    nonlinear_coeff: float = 0.1
    boundary_condition: str = "reflect"
    use_feedback: bool = True
    stabilize: bool = True
    clamp_values: bool = True
    max_abs_field_value: float = 10.0
    use_checkpointing: bool = False


class SnapshotRolloutEngine(nn.Module):
    def __init__(self, config: RolloutConfig, *, channels: int):
        super().__init__()
        self.config = config
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("laplace_kernel", kernel.view(1, 1, 3, 3))
        self.feedback_drive = nn.Conv2d(channels, channels, kernel_size=1)

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        channels = field.shape[1]
        kernel = self.laplace_kernel.expand(channels, 1, 3, 3)
        if self.config.boundary_condition == "reflect":
            padded = F.pad(field, (1, 1, 1, 1), mode="reflect")
            return F.conv2d(padded, kernel, groups=channels)
        if self.config.boundary_condition == "circular":
            padded = F.pad(field, (1, 1, 1, 1), mode="circular")
            return F.conv2d(padded, kernel, groups=channels)
        return F.conv2d(field, kernel, padding=1, groups=channels)

    def _nonlinear(self, field: torch.Tensor) -> torch.Tensor:
        if self.config.dynamics == "linear_field":
            return torch.zeros_like(field)
        if self.config.nonlinear_term == "cubic":
            return -field.pow(3)
        if self.config.nonlinear_term == "tanh_saturation":
            return torch.tanh(field)
        return field - field.pow(3)

    def forward(self, branches: SnapshotState) -> SnapshotState:
        rolled = branches.clone_for_branching()
        patch = rolled.field_patch
        if patch.ndim != 6:
            raise ValueError("Snapshot rollout expects branched patch shape [batch, regions, branches, channels, h, w].")
        if not self.config.enabled or self.config.dynamics == "frozen_field":
            rolled.diagnostics.update({"rollout_steps": 0, "rollout_stability_warning_count": 0})
            return rolled
        b, r, k, c, h, w = patch.shape
        flat = patch.reshape(b * r * k, c, h, w)
        warnings = 0
        for _ in range(max(0, self.config.steps)):
            lap = self._laplacian(flat)
            update = (
                self.config.diffusion_coeff * lap
                - self.config.damping * flat
                + self.config.nonlinear_coeff * self._nonlinear(flat)
            )
            if self.config.use_feedback:
                update = update + 0.01 * torch.tanh(self.feedback_drive(flat))
            flat = flat + self.config.dt * update
            if self.config.stabilize:
                flat = torch.nan_to_num(flat, nan=0.0, posinf=self.config.max_abs_field_value, neginf=-self.config.max_abs_field_value)
            if self.config.clamp_values:
                flat = flat.clamp(-self.config.max_abs_field_value, self.config.max_abs_field_value)
            if not torch.isfinite(flat).all():
                warnings += 1
                flat = torch.nan_to_num(flat)
        rolled.field_patch = flat.reshape(b, r, k, c, h, w)
        rolled.local_energy = rolled.field_patch.square().mean(dim=(-1, -2, -3))
        rolled.diagnostics.update(
            {
                "rollout_steps": self.config.steps,
                "rollout_dynamics": self.config.dynamics,
                "rollout_stability_warning_count": warnings,
            }
        )
        return rolled


def rollout_config_from_dict(payload: dict[str, Any] | None) -> RolloutConfig:
    payload = payload or {}
    return RolloutConfig(
        enabled=bool(payload.get("enabled", True)),
        steps=int(payload.get("steps", 4)),
        dt=float(payload.get("dt", 0.1)),
        dynamics=str(payload.get("dynamics", "nonlinear_field")),
        nonlinear_term=str(payload.get("nonlinear_term", "phi4_like")),
        diffusion_coeff=float(payload.get("diffusion_coeff", 0.05)),
        damping=float(payload.get("damping", 0.01)),
        nonlinear_coeff=float(payload.get("nonlinear_coeff", 0.1)),
        boundary_condition=str(payload.get("boundary_condition", "reflect")),
        use_feedback=bool(payload.get("use_feedback", True)),
        stabilize=bool(payload.get("stabilize", True)),
        clamp_values=bool(payload.get("clamp_values", True)),
        max_abs_field_value=float(payload.get("max_abs_field_value", 10.0)),
        use_checkpointing=bool(payload.get("use_checkpointing", False)),
    )
