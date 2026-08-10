from __future__ import annotations

from square_sim.config import Settings
from square_sim.orchestration.node_registry import heartbeat


def run_worker(settings: Settings, role: str, queues: list[str]) -> None:
    heartbeat(settings, role)
    try:
        from redis import Redis
        from rq import Worker
    except ImportError as exc:
        raise RuntimeError("Redis/RQ dependencies are required for worker mode. Run `uv sync`.") from exc
    redis = Redis.from_url(settings.redis_url)
    Worker(queues, connection=redis).work()

