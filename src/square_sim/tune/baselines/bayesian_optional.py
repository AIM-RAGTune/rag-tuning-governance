from __future__ import annotations


def availability() -> dict[str, str]:
    try:
        import optuna  # noqa: F401
    except Exception:
        return {"status": "skipped", "reason": "optuna is not installed; deterministic adaptive fallback is used by the simulator."}
    return {"status": "available"}

