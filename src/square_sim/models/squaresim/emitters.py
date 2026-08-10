from __future__ import annotations

import torch
from torch import nn


class EmitterLayer(nn.Module):
    def __init__(self, encoded_dim: int, emitter_count: int = 32, static_emitters: bool = False):
        super().__init__()
        self.static_emitters = static_emitters
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Linear(encoded_dim, emitter_count),
        )
        self.static = nn.Parameter(torch.zeros(emitter_count))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.static_emitters:
            return self.static.unsqueeze(0).expand(encoded.shape[0], -1)
        return torch.tanh(self.net(encoded))

