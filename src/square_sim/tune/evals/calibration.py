from __future__ import annotations


def calibration_error(predicted: float, realized: float) -> float:
    return abs(float(predicted) - float(realized))

