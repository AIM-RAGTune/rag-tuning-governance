from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from square_sim.registry.db import connect


def create_job(database_url: str, queue: str, payload: dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with connect(database_url) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, queue, payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, "submitted", queue, json.dumps(payload, sort_keys=True), now, now),
        )
        conn.commit()
    return job_id


def list_jobs(database_url: str) -> list[dict[str, Any]]:
    with connect(database_url) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()]


def get_job(database_url: str, job_id: str) -> dict[str, Any] | None:
    with connect(database_url) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def set_job_status(database_url: str, job_id: str, status: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect(database_url) as conn:
        conn.execute("UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?", (status, now, job_id))
        conn.commit()

