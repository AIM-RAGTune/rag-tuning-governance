from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NonlinearFieldDynamics(nn.Module):
    def __init__(
        self,
        damping: float = 0.08,
        diffusion: float = 0.08,
        nonlinear: float = 0.05,
        residual: bool = True,
        nonlinear_enabled: bool = True,
        linear_only: bool = False,
    ):
        super().__init__()
        self.damping = damping
        self.diffusion = diffusion
        self.nonlinear = nonlinear
        self.residual = residual
        self.nonlinear_enabled = nonlinear_enabled
        self.linear_only = linear_only
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("laplace_kernel", kernel.view(1, 1, 3, 3))

    def forward(self, field: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        channels = field.shape[1]
        kernel = self.laplace_kernel.expand(channels, 1, 3, 3)
        lap = F.conv2d(field, kernel, padding=1, groups=channels)
        update = (1.0 - self.damping) * field + self.diffusion * lap + 0.20 * drive
        if self.nonlinear_enabled and not self.linear_only:
            update = update + self.nonlinear * torch.tanh(field) * field
        if self.residual:
            update = 0.5 * field + 0.5 * update
        return update.clamp(-10.0, 10.0)

