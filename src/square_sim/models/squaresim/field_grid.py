from __future__ import annotations

import math

import torch
from torch import nn


class ScalarFieldGrid(nn.Module):
    def __init__(self, emitter_count: int = 32, grid_size: int = 32, channels: int = 1):
        super().__init__()
        self.emitter_count = emitter_count
        self.grid_size = grid_size
        self.channels = channels
        coords = torch.linspace(-1.0, 1.0, grid_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        centers = []
        for i in range(emitter_count):
            angle = 2 * math.pi * i / emitter_count
            radius = 0.25 + 0.65 * ((i % 4) / 3)
            centers.append((radius * math.cos(angle), radius * math.sin(angle)))
        center_tensor = torch.tensor(centers, dtype=torch.float32)
        cx = center_tensor[:, 0].view(emitter_count, 1, 1)
        cy = center_tensor[:, 1].view(emitter_count, 1, 1)
        sigma = 0.18
        footprints = torch.exp(-((xx.unsqueeze(0) - cx) ** 2 + (yy.unsqueeze(0) - cy) ** 2) / (2 * sigma**2))
        footprints = footprints / footprints.flatten(1).sum(dim=1).view(-1, 1, 1).clamp_min(1e-6)
        self.register_buffer("footprints", footprints)
        self.channel_scale = nn.Parameter(torch.ones(channels, emitter_count))

    def forward(self, emitter_activations: torch.Tensor) -> torch.Tensor:
        # b,e and c,e with e,h,w -> b,c,h,w
        weighted = emitter_activations.unsqueeze(1) * self.channel_scale.unsqueeze(0)
        return torch.einsum("bce,ehw->bchw", weighted, self.footprints)

