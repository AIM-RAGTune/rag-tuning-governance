from __future__ import annotations

from typing import Any

from square_sim.config import Settings


def enqueue(settings: Settings, queue_name: str, fn_name: str, payload: dict[str, Any]) -> str:
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        raise RuntimeError("Redis/RQ dependencies are required for service queue mode. Run `uv sync`.") from exc
    redis = Redis.from_url(settings.redis_url)
    queue = Queue(queue_name, connection=redis)
    job = queue.enqueue(fn_name, payload)
    return str(job.id)

