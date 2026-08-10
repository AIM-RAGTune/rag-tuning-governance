from __future__ import annotations

from square_sim.tune.simulator.state import TuneState


def evaluate_state(state: TuneState) -> dict[str, float]:
    return dict(state.eval_vector)

