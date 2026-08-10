from __future__ import annotations

import torch
from torch import nn


class GateInspiredFourierSeries(nn.Module):
    def __init__(self, input_dim: int, rank: int = 16):
        super().__init__()
        self.angle = nn.Linear(input_dim, rank)
        self.mix = nn.Sequential(nn.Linear(rank * 3, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.angle(x)
        entangle = torch.sin(theta) * torch.roll(torch.cos(theta), shifts=1, dims=-1)
        return self.mix(torch.cat([torch.sin(theta), torch.cos(theta), entangle], dim=-1)).squeeze(-1)


class GateInspiredVQCSurrogate(nn.Module):
    def __init__(self, input_dim: int, layers: int = 3, width: int = 16):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.readout = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = torch.sin(self.input(x))
        for layer in self.layers:
            rotated = torch.sin(layer(state)) + torch.cos(torch.roll(state, 1, dims=-1))
            state = 0.5 * state + 0.5 * rotated
        return self.readout(state).squeeze(-1)


def make_gate_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "gate_inspired_vqc_surrogate":
        return GateInspiredVQCSurrogate(input_dim)
    if model_name == "gate_inspired_fourier_series":
        return GateInspiredFourierSeries(input_dim)
    raise ValueError(f"Unknown gate-inspired baseline '{model_name}'.")

