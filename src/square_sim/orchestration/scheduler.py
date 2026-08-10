from __future__ import annotations


def scheduler_status() -> dict[str, str]:
    return {"status": "manual", "message": "Use cron/systemd or RQ Scheduler for recurring matrices."}

