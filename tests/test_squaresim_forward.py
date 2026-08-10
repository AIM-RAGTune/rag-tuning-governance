import pytest

torch = pytest.importorskip("torch")

from square_sim.models.squaresim.model import SquareSimConfig, SQUARESimModel


def test_squaresim_forward_cpu():
    model = SQUARESimModel(SquareSimConfig(input_dim=5, grid_size=16, emitter_count=8, steps=2))
    x = torch.randn(4, 5)
    model.fit_scaler(x)
    out = model(x)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()

