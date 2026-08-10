from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.config import Settings
from square_sim.tune.config import TuneBudget
from square_sim.tune.experiments.runner import run_tune_matrix
from square_sim.tune.external.acquire import acquire_external
from square_sim.tune.peft_optional.local_lora_smoke import run_peft_smoke
from square_sim.tune.reporting.certificate import certificate_for_dataset
from square_sim.tune.simulator.response_surface import ResponseSurface
from square_sim.tune.simulator.square_tune_optimizer import run_optimizer
from square_sim.tune.simulator.state import initial_state_from_frame
from square_sim.tune.synthetic.generators import generate_dataset, generate_suite
from square_sim.tune.synthetic.schemas import LATENT_COLUMNS
from square_sim.tune.synthetic.validators import validate_suite


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_env(tmp_path)


def test_synthetic_generation_reproducible_by_seed(tmp_path: Path) -> None:
    a = generate_dataset("synthetic_llm_failure_cluster_routing", tmp_path / "a", rows=128, seed=101)
    b = generate_dataset("synthetic_llm_failure_cluster_routing", tmp_path / "b", rows=128, seed=101)
    dfa = pd.read_parquet(Path(a["path"]) / "data.parquet")
    dfb = pd.read_parquet(Path(b["path"]) / "data.parquet")
    pd.testing.assert_frame_equal(dfa, dfb)


def test_synthetic_generation_different_seeds_different_data(tmp_path: Path) -> None:
    a = generate_dataset("synthetic_llm_failure_cluster_routing", tmp_path, rows=128, seed=101)
    b = generate_dataset("synthetic_llm_failure_cluster_routing", tmp_path, rows=128, seed=202)
    dfa = pd.read_parquet(Path(a["path"]) / "data.parquet")
    dfb = pd.read_parquet(Path(b["path"]) / "data.parquet")
    assert not dfa.equals(dfb)


def test_generator_manifest_card_expected_outcomes_written(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_merge_required", tmp_path, rows=64, seed=101)
    path = Path(row["path"])
    assert (path / "generator_manifest.json").exists()
    assert (path / "mechanism_card.md").exists()
    assert (path / "expected_outcomes.json").exists()
    assert (path / "checksums.sha256").exists()


def test_latent_columns_excluded_by_default(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_data_poison_regression", tmp_path, rows=64, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "data.parquet")
    assert not any(col in df.columns for col in LATENT_COLUMNS)


def test_suite_validation(tmp_path: Path) -> None:
    generate_suite(
        tmp_path,
        rows=64,
        seeds=[101],
        datasets=["synthetic_llm_linear_control", "synthetic_llm_random_label"],
    )
    report = validate_suite(tmp_path)
    assert report["failed"] == 0
    assert report["dataset_count"] == 2


def test_random_label_balanced_and_refused(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_random_label", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=2, num_branches=2, rollout_steps=1)
    rows = []
    for opt in ["random_search", "greedy_eval_improvement", "square_tune_full"]:
        result = run_optimizer(opt, df, mechanism_name="random_label", seed=101, budget=budget)
        metric = result.metrics
        metric.update({"optimizer_name": opt, "dataset_key": "synthetic_llm_random_label", "control_type": "refusal_control"})
        rows.append(metric)
    cert = certificate_for_dataset("synthetic_llm_random_label", pd.DataFrame(rows))
    assert cert["status"] in {"Refusal control passed", "Refusal control failed"}
    assert cert["status"] == "Refusal control passed"


def test_linear_control_solved_by_simple_baseline(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_linear_control", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=3, num_branches=3, rollout_steps=1)
    greedy = run_optimizer("greedy_eval_improvement", df, mechanism_name="linear_control", seed=101, budget=budget)
    full = run_optimizer("square_tune_full", df, mechanism_name="linear_control", seed=101, budget=budget)
    assert greedy.metrics["final_utility"] >= full.metrics["final_utility"] - 0.2


def test_response_surface_deterministic_with_seed(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_rag_policy_conflict", tmp_path, rows=64, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    state = initial_state_from_frame(df, 101)
    import numpy as np

    from square_sim.tune.simulator.actions import make_action_space

    action = make_action_space(list(state.failure_clusters), np.random.default_rng(1), count=1)[0]
    surface = ResponseSurface()
    a = surface.evaluate_action(state, action, mechanism_name="rag_policy_conflict", seed=101, round_idx=0)
    b = surface.evaluate_action(state, action, mechanism_name="rag_policy_conflict", seed=101, round_idx=0)
    assert a.realized_utility == b.realized_utility


def test_square_tune_full_runs_cpu_and_ablations_change_behavior(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_merge_required", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=3, num_branches=3, rollout_steps=2)
    full = run_optimizer("square_tune_full", df, mechanism_name="merge_required", seed=101, budget=budget)
    no_fork = run_optimizer("square_tune_no_fork", df, mechanism_name="merge_required", seed=101, budget=budget)
    linear = run_optimizer("square_tune_linear_rollout", df, mechanism_name="merge_required", seed=101, budget=budget)
    no_merge = run_optimizer("square_tune_no_merge", df, mechanism_name="merge_required", seed=101, budget=budget)
    assert full.trajectory.shape[0] == 3
    assert no_fork.branch_diagnostics["branch_id"].max() == 0
    assert linear.metrics["nonlinear_rollout"] is False
    assert no_merge.metrics["merge_enabled"] is False


def test_baseline_random_search_runs(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_tool_routing", tmp_path, rows=128, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("random_search", df, mechanism_name="tool_routing", seed=101, budget=TuneBudget(max_rounds=2))
    assert "final_utility" in result.metrics


def test_run_report_and_commercial_metrics_present(tmp_path: Path) -> None:
    root = tmp_path / "nas" / "SQUARE" / "source-validation-workspace"
    data_root = root / "datasets" / "synthetic" / "square_tune"
    generate_suite(data_root, rows=128, seeds=[101], datasets=["synthetic_llm_failure_cluster_routing"])
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(
        f"""experiment_name: test_square_tune
dataset_root: {data_root}
datasets: [synthetic_llm_failure_cluster_routing]
seeds: [101]
optimizers: [random_search, square_tune_full, square_tune_no_fork]
protocol_path: protocols/square_tune/synthetic_mechanism_protocol_v1.yaml
square_tune:
  max_rounds: 2
  num_branches: 2
  rollout_steps: 1
""",
        encoding="utf-8",
    )
    summary = run_tune_matrix(_settings(tmp_path / "nas"), cfg_path, skip_completed=False)
    assert summary["completed"] == 3
    metrics = root / "reports" / "square_tune" / "experiments" / summary["experiment_id"] / "metrics.parquet"
    assert metrics.exists()
    manifests = list((root / "tune_runs").glob("*/*/*/*/run_manifest.json"))
    assert manifests


def test_certificate_supports_merge_only_when_full_beats_no_merge() -> None:
    df = pd.DataFrame(
        [
            {"optimizer_name": "square_tune_full", "final_utility": 0.70, "control_type": "positive_control"},
            {"optimizer_name": "square_tune_no_merge", "final_utility": 0.60, "control_type": "positive_control"},
            {"optimizer_name": "random_search", "final_utility": 0.55, "control_type": "positive_control"},
            {"optimizer_name": "greedy_eval_improvement", "final_utility": 0.56, "control_type": "positive_control"},
        ]
    )
    cert = certificate_for_dataset("synthetic_llm_merge_required", df)
    assert cert["mechanism_support"]["merge_reintegration"] == "supported"


def test_external_acquire_graceful_without_config(tmp_path: Path) -> None:
    result = acquire_external(tmp_path / "missing.yaml")
    assert result["status"] == "skipped"


def test_peft_smoke_skips_when_dependencies_missing_or_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "peft.yaml"
    cfg.write_text("enable_real_peft: false\n", encoding="utf-8")
    result = run_peft_smoke(cfg)
    assert result["status"] == "skipped"
