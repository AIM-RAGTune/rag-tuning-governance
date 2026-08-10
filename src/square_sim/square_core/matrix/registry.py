from __future__ import annotations

from square_sim.square_core.adaptive_arch.runner import simulate as adaptive_arch_simulate
from square_sim.square_core.closed_loop.runner import simulate as closed_loop_simulate
from square_sim.square_core.field_substrate.runner import simulate as field_substrate_simulate
from square_sim.square_core.quantum_coupling.runner import simulate as quantum_coupling_simulate
from square_sim.square_core.soliton.runner import simulate as soliton_simulate

TRACK_RUNNERS = {
    "adaptive_arch": adaptive_arch_simulate,
    "field_substrate": field_substrate_simulate,
    "closed_loop": closed_loop_simulate,
    "quantum_coupling": quantum_coupling_simulate,
    "soliton": soliton_simulate,
}
