from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from square_sim.config import Settings
from square_sim.square_core.adaptive_arch.runner import simulate as simulate_adaptive
from square_sim.square_core.closed_loop.runner import simulate as simulate_closed_loop
from square_sim.square_core.common.certificates import certificate_for_group
from square_sim.square_core.config import CoreConfig
from square_sim.square_core.field_substrate.emitters import emitter_basis
from square_sim.square_core.field_substrate.runner import simulate as simulate_field
from square_sim.square_core.matrix.plan import plan_matrix
from square_sim.square_core.matrix.runner import run_core_matrix
from square_sim.square_core.quantum_coupling.runner import simulate as simulate_quantum
from square_sim.square_core.soliton.equations import nonlinear_step
from square_sim.square_core.soliton.runner import simulate as simulate_soliton
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.write_once import WriteOnceError, WriteOncePathManager


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        aim_nas_root=tmp_path,
        project_root=tmp_path / "lab",
        gpu_hot_scratch=tmp_path / "hot",
        gpu_warm_scratch=tmp_path / "warm",
        processing_scratch=tmp_path / "processing",
        database_url="sqlite://",
        redis_url="redis://localhost:6379/0",
        api_host="127.0.0.1",
        api_port=0,
        local_llm_endpoint=None,
    )


def write_cfg(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "core.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_square_core_config_loads() -> None:
    cfg = CoreConfig.from_path(Path("configs/square_core/square_core_validation_v1_smoke.yaml"))
    assert cfg.tracks == ["adaptive_arch", "field_substrate", "closed_loop"]
    assert cfg.grid_size == 16


def test_square_core_plan_expands() -> None:
    plan = plan_matrix(Path("configs/square_core/square_core_validation_v1_smoke.yaml"))
    assert plan["total_planned"] == 3 * 1 * 4 + 2 * 1 * 5 + 2 * 1 * 3


def test_square_core_no_overwrite_protects_prior_paths(tmp_path: Path) -> None:
    s = settings_for(tmp_path)
    prior = s.project_root / "reports" / "square_tune" / "calibration"
    prior.mkdir(parents=True)
    registry = ProtectedResultsRegistry(s)
    registry.protect_defaults()
    manager = WriteOncePathManager(s.project_root, registry.protected_paths())
    with pytest.raises(WriteOnceError):
        manager.ensure_writable_path(prior / "old" / "x.json")


def test_square_core_certificate_refuses_random_control() -> None:
    df = pd.DataFrame(
        [
            {"track": "adaptive_arch", "task": "random_unlearnable_control", "system": "static_policy", "cost_adjusted_utility": 0.0},
            {"track": "adaptive_arch", "task": "random_unlearnable_control", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.0},
        ]
    )
    assert certificate_for_group("adaptive_arch", "random_unlearnable_control", df)["status"] == "Refused"


def test_square_core_certificate_blocks_numerical_instability() -> None:
    df = pd.DataFrame(
        [
            {
                "track": "field_substrate",
                "task": "emitter_target_field_reconstruction",
                "system": "square_field_feedback",
                "cost_adjusted_utility": 0.5,
                "numerical_instability": True,
            }
        ]
    )
    assert certificate_for_group("field_substrate", "emitter_target_field_reconstruction", df)["status"] == "Numerical instability"


def test_adaptive_arch_compute_allocation_adaptive_beats_always_on_cost_adjusted() -> None:
    adaptive, _ = simulate_adaptive("compute_allocation_trap", "square_adaptive_arch_adaptive_compute", 101)
    always, _ = simulate_adaptive("compute_allocation_trap", "square_adaptive_arch_always_fork", 101)
    assert adaptive["cost_adjusted_utility"] > always["cost_adjusted_utility"]


def test_adaptive_arch_future_rollout_no_fork_loses() -> None:
    full, _ = simulate_adaptive("future_rollout_required", "square_adaptive_arch_full", 101)
    no_fork, _ = simulate_adaptive("future_rollout_required", "square_adaptive_arch_no_fork", 101)
    assert full["final_utility"] > no_fork["final_utility"]


def test_emitter_field_shapes_have_expected_dimensions() -> None:
    basis = emitter_basis(16, 8, 101)
    assert basis.shape == (8, 16, 16)


def test_target_field_reconstruction_improves_over_random() -> None:
    square, _ = simulate_field("emitter_target_field_reconstruction", "square_field_feedback", 101, grid_size=16, emitter_count=8)
    random, _ = simulate_field("emitter_target_field_reconstruction", "random_emitter_activation", 101, grid_size=16, emitter_count=8)
    assert square["target_field_error"] < random["target_field_error"]


def test_crosstalk_matrix_computed() -> None:
    metrics, _ = simulate_field("field_crosstalk_map", "square_field_adaptive_arch", 101, grid_size=16, emitter_count=8)
    assert metrics["crosstalk_matrix_norm"] >= 0


def test_closed_loop_beats_open_loop_on_simple_drift() -> None:
    closed, _ = simulate_closed_loop("closed_loop_field_stabilization", "square_adaptive_controller", 101, grid_size=16, steps=16)
    open_loop, _ = simulate_closed_loop("closed_loop_field_stabilization", "open_loop_script", 101, grid_size=16, steps=16)
    assert closed["final_utility"] > open_loop["final_utility"]


def test_controller_recovers_after_perturbation() -> None:
    metrics, _ = simulate_closed_loop("adaptive_recovery_after_perturbation", "square_adaptive_controller", 101, grid_size=16, steps=16)
    assert metrics["recovery_time"] < 16


def test_single_qubit_field_modulation_cpu() -> None:
    controlled, _ = simulate_quantum("single_qubit_field_modulation", "square_adaptive_field_control", 101, steps=16)
    baseline, _ = simulate_quantum("single_qubit_field_modulation", "uncontrolled_evolution", 101, steps=16)
    assert controlled["state_fidelity"] >= baseline["state_fidelity"]


def test_noise_regime_comparison_reports_all_regimes() -> None:
    metrics, _ = simulate_quantum("adversarial_noise_model", "square_field_control", 101, steps=16)
    assert {"optimistic_fidelity", "neutral_fidelity", "adversarial_fidelity"} <= set(metrics)


def test_soliton_equation_step_no_nan() -> None:
    import numpy as np

    u = np.tanh(np.linspace(-1, 1, 32))
    v = np.zeros_like(u)
    u2, v2 = nonlinear_step(u, v)
    assert np.isfinite(u2).all()
    assert np.isfinite(v2).all()


def test_soliton_stability_metric_computed() -> None:
    metrics, _ = simulate_soliton("soliton_stability_under_noise", "square_feedback_soliton", 101, grid_size=16, steps=16)
    assert metrics["localization_error"] >= 0


def test_square_core_smoke_cpu_runs(tmp_path: Path) -> None:
    cfg_path = write_cfg(
        tmp_path,
        """
matrix_name: unit_smoke
tracks: [field_substrate, closed_loop, quantum_coupling, soliton]
seeds: [101]
tasks:
  field_substrate: [emitter_target_field_reconstruction]
  closed_loop: [closed_loop_field_stabilization]
  quantum_coupling: [single_qubit_field_modulation]
  soliton: [soliton_formation_threshold]
systems:
  field_substrate: [random_emitter_activation, square_field_feedback]
  closed_loop: [open_loop_script, square_adaptive_controller]
  quantum_coupling: [uncontrolled_evolution, square_field_control]
  soliton: [linear_wave_packet, square_feedback_soliton]
simulation:
  grid_size: 16
  emitter_count: 8
  steps: 8
""",
    )
    result = run_core_matrix(settings_for(tmp_path), cfg_path)
    assert result["total_planned"] == 8
    assert result["succeeded"] == 8
    assert Path(result["reports_dir"], "experiment_summary.json").exists()
