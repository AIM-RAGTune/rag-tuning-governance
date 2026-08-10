from pathlib import Path

from square_sim.registry.models import RunRecord
from square_sim.registry.repositories import RunRepository


def test_run_registry_insert_update(tmp_path: Path):
    repo = RunRepository(f"sqlite:///{tmp_path / 'registry.sqlite3'}")
    record = RunRecord(
        run_id="20260101-000000-energy-target-logistic-abc",
        dataset="energy",
        dataset_version="v1",
        split_id="default",
        target="target",
        model="logistic_regression",
        seed=42,
        config_hash="abc",
        status="running",
        run_path=str(tmp_path),
    )
    repo.upsert(record)
    repo.update_status(record.run_id, "succeeded")
    assert repo.get(record.run_id)["status"] == "succeeded"

