import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from square_sim.training.metrics import binary_metrics


def test_metric_calculation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_metrics(y, s)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0

