from __future__ import annotations

from itertools import pairwise
from typing import Any


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    safe_score = np.clip(y_score, 1e-7, 1 - 1e-7)
    out: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true.tolist())) > 1 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, safe_score, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "ece": expected_calibration_error(y_true, y_score),
    }
    return out


def expected_calibration_error(y_true, y_score, bins: int = 10) -> float:
    import numpy as np

    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in pairwise(edges):
        mask = (y_score >= low) & (y_score < high if high < 1.0 else y_score <= high)
        if mask.any():
            ece += (mask.mean()) * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)
