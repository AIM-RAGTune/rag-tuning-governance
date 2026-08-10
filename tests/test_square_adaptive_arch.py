from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.adaptive_arch.certificate import certificate_for_task
from square_sim.adaptive_arch.config import AdaptiveArchConfig
from square_sim.adaptive_arch.generators import generate_suite, generate_task, validate_suite
from square_sim.adaptive_arch.runner import run_benchmark
from square_sim.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(tmp_path / "nas")


def test_adaptive_arch_generator_reproducible_by_seed(tmp_path: Path) -> None:
    a = generate_task("local_regime_shift", tmp_path / "a", rows=96, seed=101)
    b = generate_task("local_regime_shift", tmp_path / "b", rows=96, seed=101)
    pd.testing.assert_frame_equal(
        pd.read_parquet(Path(a["path"]) / "data.parquet"),
        pd.read_parquet(Path(b["path"]) / "data.parquet"),
    )


def test_random_unlearnable_control_refuses() -> None:
    df = pd.DataFrame(
        [
            {"task": "random_unlearnable_control", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.0},
            {"task": "random_unlearnable_control", "system": "static_policy", "cost_adjusted_utility": 0.0},
        ]
    )
    assert certificate_for_task("random_unlearnable_control", df)["status"] == "Refused"


def test_linear_static_control_static_baseline_wins() -> None:
    df = pd.DataFrame(
        [
            {"task": "linear_static_control", "system": "linear_static_baseline", "cost_adjusted_utility": 0.12},
            {"task": "linear_static_control", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.08},
        ]
    )
    assert certificate_for_task("linear_static_control", df)["status"] == "Refused"


def test_component_support_requires_ablation_win() -> None:
    df = pd.DataFrame(
        [
            {"task": "merge_required_architecture", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.20},
            {"task": "merge_required_architecture", "system": "square_adaptive_arch_no_merge", "cost_adjusted_utility": 0.10},
            {"task": "merge_required_architecture", "system": "static_policy", "cost_adjusted_utility": 0.05},
        ]
    )
    cert = certificate_for_task("merge_required_architecture", df)
    assert cert["component_support"]["merge_reintegration"] == "supported"


def test_adaptive_arch_smoke_config_loads() -> None:
    cfg = AdaptiveArchConfig.from_path(Path("configs/adaptive_arch/square_adaptive_arch_v1_smoke.yaml"))
    assert "compute_allocation_trap" in cfg.tasks
    assert "square_adaptive_arch_full" in cfg.systems


def test_adaptive_arch_run_cpu_smoke(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    data_root = settings.project_root / "datasets" / "synthetic" / "square_adaptive_arch_v1"
    generate_suite(
        data_root,
        rows=128,
        seeds=[101],
        tasks=["compute_allocation_trap", "linear_static_control", "random_unlearnable_control"],
    )
    cfg = tmp_path / "arch_smoke.yaml"
    cfg.write_text(
        f"""experiment_name: test_arch_smoke
dataset_root: {data_root}
tasks: [compute_allocation_trap, linear_static_control, random_unlearnable_control]
seeds: [101]
systems: [static_policy, linear_static_baseline, square_adaptive_arch_full, square_adaptive_arch_no_fork, square_adaptive_arch_always_fork]
budget:
  max_rounds: 2
  num_branches: 2
  rollout_steps: 1
  max_response_surface_evaluations: 16
  max_candidate_actions: 16
  simulated_gpu_hour_budget: 4.0
""",
        encoding="utf-8",
    )
    summary = run_benchmark(settings, cfg, skip_completed=False)
    assert summary["succeeded"] == 15
    report_dir = settings.project_root / "reports" / "square_adaptive_arch" / "v1" / summary["experiment_id"]
    assert (report_dir / "metrics.parquet").exists()
    assert (report_dir / "no_overwrite_audit.json").exists()
    manifests = list((settings.project_root / "square_adaptive_arch_runs").glob("*/*/*/*/run_manifest.json"))
    assert manifests
    trace = list((settings.project_root / "square_adaptive_arch_runs").glob("*/*/*/*/adaptive_arch_diagnostics/architecture_trace.parquet"))
    assert trace


def test_validate_generated_tasks(tmp_path: Path) -> None:
    generate_suite(tmp_path, rows=64, seeds=[101], tasks=["future_rollout_required"])
    result = validate_suite(tmp_path)
    assert result["failed"] == 0
    assert result["dataset_versions"] == 1


def test_compute_allocation_trap_full_over_spends() -> None:
    df = pd.DataFrame(
        [
            {"task": "compute_allocation_trap", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.2},
            {"task": "compute_allocation_trap", "system": "square_adaptive_arch_always_fork", "cost_adjusted_utility": 0.1},
            {"task": "compute_allocation_trap", "system": "static_policy", "cost_adjusted_utility": 0.05},
        ]
    )
    cert = certificate_for_task("compute_allocation_trap", df)
    assert cert["component_support"]["adaptive_compute_allocation"] == "supported"


def _component_cert(task: str, ablation: str, component: str):
    df = pd.DataFrame(
        [
            {"task": task, "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.2},
            {"task": task, "system": ablation, "cost_adjusted_utility": 0.1},
            {"task": task, "system": "static_policy", "cost_adjusted_utility": 0.04},
        ]
    )
    return certificate_for_task(task, df)["component_support"][component]


def test_local_regime_shift_requires_reconfiguration() -> None:
    assert _component_cert("local_regime_shift", "square_adaptive_arch_no_local_reconfiguration", "local_reconfiguration") == "supported"


def test_future_rollout_required_no_fork_loses() -> None:
    assert _component_cert("future_rollout_required", "square_adaptive_arch_no_fork", "conditional_forking") == "supported"


def test_merge_required_no_merge_loses() -> None:
    assert _component_cert("merge_required_architecture", "square_adaptive_arch_no_merge", "merge_reintegration") == "supported"


def test_memory_prevents_repeated_failure_no_memory_loses() -> None:
    assert _component_cert("memory_prevents_repeated_failure", "square_adaptive_arch_no_memory", "architecture_memory") == "supported"


def test_dynamic_topology_static_topology_loses() -> None:
    assert _component_cert("dynamic_topology_routing", "square_adaptive_arch_static_topology", "dynamic_topology") == "supported"


def test_nonlinear_extrapolation_linear_rollout_loses() -> None:
    assert _component_cert("nonlinear_extrapolation_required", "square_adaptive_arch_linear_rollout", "nonlinear_rollout") == "supported"


def test_protect_known_good_global_update_regresses() -> None:
    assert _component_cert("protect_known_good_while_adapting", "square_adaptive_arch_no_regression_protection", "regression_protection") == "supported"


def test_compute_allocation_trap_no_fork_misses_hard_subset() -> None:
    df = pd.DataFrame(
        [
            {"task": "compute_allocation_trap", "system": "square_adaptive_arch_full", "cost_adjusted_utility": 0.2},
            {"task": "compute_allocation_trap", "system": "square_adaptive_arch_no_fork", "cost_adjusted_utility": 0.1},
            {"task": "compute_allocation_trap", "system": "static_policy", "cost_adjusted_utility": 0.05},
        ]
    )
    assert certificate_for_task("compute_allocation_trap", df)["component_support"]["conditional_forking"] == "supported"
