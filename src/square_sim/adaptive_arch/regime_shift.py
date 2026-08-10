from __future__ import annotations


def regime_shift_score(local_error: float, uncertainty: float, conflict: float) -> float:
    return float(min(1.0, 0.45 * local_error + 0.35 * uncertainty + 0.20 * conflict))

