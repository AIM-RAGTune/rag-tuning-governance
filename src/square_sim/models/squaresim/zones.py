from __future__ import annotations

import torch
from torch import nn


def fixed_zone_masks(grid_size: int, include_overlap: bool = True) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, grid_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    masks = [
        ((xx + 0.45) ** 2 + (yy + 0.45) ** 2 < 0.35**2).float(),
        ((xx - 0.45) ** 2 + (yy + 0.45) ** 2 < 0.35**2).float(),
        ((xx + 0.45) ** 2 + (yy - 0.45) ** 2 < 0.35**2).float(),
        ((xx - 0.45) ** 2 + (yy - 0.45) ** 2 < 0.35**2).float(),
        ((xx**2 + yy**2) < 0.45**2).float(),
    ]
    if include_overlap:
        masks.append(((xx.abs() < 0.25) & (yy.abs() < 0.65)).float())
    stacked = torch.stack(masks)
    return stacked / stacked.flatten(1).sum(dim=1).view(-1, 1, 1).clamp_min(1.0)


class ZoneReadout(nn.Module):
    def __init__(self, grid_size: int = 32, channels: int = 1, include_overlap: bool = True):
        super().__init__()
        masks = fixed_zone_masks(grid_size, include_overlap)
        self.register_buffer("masks", masks)
        self.output_dim = channels * masks.shape[0] * 4

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        masked = field.unsqueeze(2) * self.masks.view(1, 1, *self.masks.shape)
        mean = masked.sum(dim=(-1, -2))
        maxv = masked.amax(dim=(-1, -2))
        energy = (masked.square()).sum(dim=(-1, -2))
        gy = field[..., 1:, :] - field[..., :-1, :]
        gx = field[..., :, 1:] - field[..., :, :-1]
        grad_energy = (gy.square().mean(dim=(-1, -2)) + gx.square().mean(dim=(-1, -2))).unsqueeze(-1)
        grad_energy = grad_energy.expand(-1, -1, self.masks.shape[0])
        return torch.cat([mean, maxv, energy, grad_energy], dim=-1).flatten(1)


class ZoneMemory(nn.Module):
    def __init__(self, zone_dim: int, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.output_dim = zone_dim

    def forward(self, previous: torch.Tensor | None, zone_readout: torch.Tensor) -> torch.Tensor:
        if previous is None:
            return zone_readout
        return self.alpha * previous + self.beta * zone_readout

