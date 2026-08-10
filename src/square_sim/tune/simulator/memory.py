from __future__ import annotations

from square_sim.tune.simulator.actions import CandidateAction
from square_sim.tune.simulator.state import TuneState


def remember_intervention(state: TuneState, action: CandidateAction, *, good: bool) -> None:
    key = "known_good" if good else "known_bad"
    state.memory_state.setdefault(key, []).append(action.to_dict())

