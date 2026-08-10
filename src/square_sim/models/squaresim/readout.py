from __future__ import annotations

from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        hidden = max(16, min(128, input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)

