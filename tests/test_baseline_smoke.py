import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from square_sim.models.baselines.sklearn_baselines import (
    make_sklearn_baseline,
    predict_proba_positive,
)


def test_baseline_smoke_training():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = make_sklearn_baseline("logistic_regression")
    model.fit(x, y)
    scores = predict_proba_positive(model, x)
    assert scores.shape == (80,)

