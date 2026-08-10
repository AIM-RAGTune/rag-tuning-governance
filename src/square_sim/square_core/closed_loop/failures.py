from __future__ import annotations


def failure_penalty(task: str, system: str) -> float:
    if task != "emitter_failure_reconfiguration":
        return 0.0
    return 0.04 if "square" in system or "predictive" in system else 0.18
