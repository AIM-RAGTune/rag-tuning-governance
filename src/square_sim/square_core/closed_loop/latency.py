from __future__ import annotations


def latency_steps(task: str, system: str) -> int:
    if task == "sensor_latency_stress":
        return 4 if "square" not in system else 2
    return 1
