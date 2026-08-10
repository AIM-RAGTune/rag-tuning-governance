from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from square_sim.config import Settings
from square_sim.data.split import create_split
from square_sim.data.synthetic import make_synthetic_dataset


def test_deterministic_split(tmp_path: Path):
    settings = Settings.from_env(tmp_path)
    make_synthetic_dataset(settings)
    first = create_split("energy", settings, split_id="default", seed=42, target="target", version="synthetic-v1")
    second = create_split("energy", settings, split_id="again", seed=42, target="target", version="synthetic-v1")
    assert first["row_counts"] == second["row_counts"]
    assert first["class_balance"] == second["class_balance"]

