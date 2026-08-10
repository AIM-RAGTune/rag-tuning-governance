from __future__ import annotations

from square_sim.registry.db import connect


def migrate(database_url: str) -> None:
    with connect(database_url):
        pass

