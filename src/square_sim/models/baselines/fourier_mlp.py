from __future__ import annotations

import torch
from torch import nn


class FourierMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        fourier_dim = input_dim * 4
        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([torch.sin(x), torch.cos(x), torch.sin(2 * x), torch.cos(2 * x)], dim=-1)
        return self.net(features).squeeze(-1)

