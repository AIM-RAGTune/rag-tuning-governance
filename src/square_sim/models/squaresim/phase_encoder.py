from __future__ import annotations

import torch
from torch import nn


class PhaseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, use_phase: bool = True):
        super().__init__()
        self.use_phase = use_phase
        self.register_buffer("mean", torch.zeros(input_dim), persistent=False)
        self.register_buffer("std", torch.ones(input_dim), persistent=False)
        encoded_dim = input_dim * 2 if use_phase else input_dim
        self.proj = nn.Sequential(nn.Linear(encoded_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    @torch.no_grad()
    def fit_scaler(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        if self.use_phase:
            z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return self.proj(z)

