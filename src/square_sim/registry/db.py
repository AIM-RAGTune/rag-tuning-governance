from __future__ import annotations

import sqlite3
from pathlib import Path


def sqlite_path_from_url(database_url: str) -> Path:
    if database_url.startswith("sqlite:///"):
        raw = database_url.removeprefix("sqlite:///")
        return Path(raw)
    if database_url.startswith("postgres"):
        raise RuntimeError(
            "Postgres registry is intended for orchestrated deployment. Install SQLAlchemy/Alembic "
            "migrations for production Postgres; local CLI fallback uses SQLite."
        )
    return Path(database_url)


def connect(database_url: str) -> sqlite3.Connection:
    path = sqlite_path_from_url(database_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          dataset TEXT NOT NULL,
          dataset_version TEXT NOT NULL,
          split_id TEXT NOT NULL,
          target TEXT NOT NULL,
          model TEXT NOT NULL,
          seed INTEGER NOT NULL,
          config_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          run_path TEXT NOT NULL,
          metrics_path TEXT,
          predictions_path TEXT,
          explanation_path TEXT,
          started_at TEXT,
          ended_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          status TEXT NOT NULL,
          queue TEXT,
          payload TEXT,
          created_at TEXT,
          updated_at TEXT
        )
        """
    )
    conn.commit()

