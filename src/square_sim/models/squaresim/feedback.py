from __future__ import annotations

import torch
from torch import nn


class FeedbackController(nn.Module):
    def __init__(self, encoded_dim: int, zone_dim: int, emitter_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoded_dim + zone_dim, encoded_dim),
            nn.GELU(),
            nn.Linear(encoded_dim, emitter_count),
        )

    def forward(self, encoded: torch.Tensor, zone_readout: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.tanh(self.net(torch.cat([encoded, zone_readout], dim=-1)))

