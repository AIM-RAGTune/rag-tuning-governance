from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 5
    best_loss: float = float("inf")
    bad_epochs: int = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - 1e-5:
            self.best_loss = loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

