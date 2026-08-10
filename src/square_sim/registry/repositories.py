from __future__ import annotations

from dataclasses import asdict

from square_sim.registry.db import connect
from square_sim.registry.models import RunRecord


class RunRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def upsert(self, record: RunRecord) -> None:
        payload = asdict(record)
        with connect(self.database_url) as conn:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, dataset, dataset_version, split_id, target, model, seed, config_hash,
                  status, run_path, metrics_path, predictions_path, explanation_path, started_at, ended_at
                ) VALUES (
                  :run_id, :dataset, :dataset_version, :split_id, :target, :model, :seed,
                  :config_hash, :status, :run_path, :metrics_path, :predictions_path,
                  :explanation_path, :started_at, :ended_at
                )
                ON CONFLICT(run_id) DO UPDATE SET
                  status=excluded.status,
                  metrics_path=excluded.metrics_path,
                  predictions_path=excluded.predictions_path,
                  explanation_path=excluded.explanation_path,
                  ended_at=excluded.ended_at
                """,
                payload,
            )
            conn.commit()

    def get(self, run_id: str) -> dict | None:
        with connect(self.database_url) as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list(self, dataset: str | None = None, target: str | None = None) -> list[dict]:
        sql = "SELECT * FROM runs"
        params = []
        where = []
        if dataset:
            where.append("dataset = ?")
            params.append(dataset)
        if target:
            where.append("target = ?")
            params.append(target)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC"
        with connect(self.database_url) as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def update_status(self, run_id: str, status: str, ended_at: str | None = None) -> None:
        with connect(self.database_url) as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = COALESCE(?, ended_at) WHERE run_id = ?",
                (status, ended_at, run_id),
            )
            conn.commit()

