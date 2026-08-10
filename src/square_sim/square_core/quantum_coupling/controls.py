from __future__ import annotations


def field_schedule(system: str, step: int, steps: int) -> float:
    if system == "uncontrolled_evolution":
        return 0.0
    if system == "static_field_control":
        return 0.35
    if system == "gate_sequence_baseline":
        return 0.65 if step in {steps // 3, 2 * steps // 3} else 0.0
    if system == "square_adaptive_field_control":
        return 0.25 + 0.35 * (step / max(steps - 1, 1))
    return 0.48
