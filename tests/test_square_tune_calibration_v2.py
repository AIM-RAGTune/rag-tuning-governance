from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.config import Settings
from square_sim.tune.config import TuneBudget, TuneExperimentConfig
from square_sim.tune.experiments.runner import run_tune_matrix
from square_sim.tune.reporting.calibration import (
    evaluate_calibration_gates,
    write_calibration_reports,
)
from square_sim.tune.reporting.certificate import certificate_for_dataset, write_certificates
from square_sim.tune.simulator.budget import BudgetLedger
from square_sim.tune.simulator.square_tune_optimizer import _merge_outcomes, run_optimizer
from square_sim.tune.simulator.state import CostState, TuneState
from square_sim.tune.synthetic.generators import generate_dataset, generate_suite


def _state(**overrides: float) -> TuneState:
    eval_vector = {
        "domain_accuracy": 0.50,
        "retrieval_faithfulness": 0.50,
        "instruction_following": 0.50,
        "style_match": 0.50,
        "safety": 0.75,
        "latency": 0.30,
        "cost": 0.30,
        "calibration": 0.50,
        "regression_score": 0.80,
    }
    eval_vector.update(overrides)
    return TuneState(
        eval_vector=eval_vector,
        failure_clusters={"regression_cluster": 0.7, "retrieval_miss_cluster": 0.6},
        data_pool_summary={},
        adapter_state={},
        prompt_policy_state={},
        rag_policy_state={},
        tool_policy_state={},
        cost_state=CostState(),
    )


def _metrics_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "seed": 101,
        "control_type": "positive_control",
        "protected_utility": 0.60,
        "cost_adjusted_improvement": 0.10,
        "regression_count": 0,
        "preserved_known_good_score": 0.90,
        "repeated_bad_action_count": 0,
        "response_surface_evaluations": 144,
        "candidate_actions_scored": 144,
        "simulated_gpu_hours": 1.0,
        "latent_columns_used": "",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_calibration_gate_random_label_pass() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_random_label", "optimizer_name": "square_tune_full", "final_utility": 0.49},
            {"dataset_key": "synthetic_llm_random_label", "optimizer_name": "random_search", "final_utility": 0.51},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.81},
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert next(g for g in gates["gates"] if g["gate_name"] == "random_label_refusal")["status"] == "passed"


def test_calibration_gate_random_label_fail_invalidates_certificates() -> None:
    gates = {"failed_gates": ["random_label_refusal"], "global_status": "failed", "gates": []}
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "random_search", "final_utility": 0.60},
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "square_tune_no_snapshot", "final_utility": 0.60},
        ]
    )
    cert = certificate_for_dataset("synthetic_llm_failure_cluster_routing", df, gates)
    assert cert["status"] == "Inconclusive pending calibration"


def test_linear_control_classical_gate_passes_when_linear_baseline_wins() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.83},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.84},
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert next(g for g in gates["gates"] if g["gate_name"] == "linear_control_classical_sanity")["status"] == "passed"


def test_linear_control_classical_gate_fails_when_square_dominates() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.90},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.80},
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert next(g for g in gates["gates"] if g["gate_name"] == "linear_control_classical_sanity")["status"] == "failed"


def test_linear_control_failure_downgrades_supported_to_inconclusive_pending_calibration() -> None:
    gates = {"failed_gates": ["linear_control_classical_sanity"], "global_status": "failed", "gates": []}
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "random_search", "final_utility": 0.60},
            {"dataset_key": "synthetic_llm_failure_cluster_routing", "optimizer_name": "square_tune_no_snapshot", "final_utility": 0.60},
        ]
    )
    assert certificate_for_dataset("synthetic_llm_failure_cluster_routing", df, gates)["status"] == "Inconclusive pending calibration"


def test_budget_ledger_equalizes_square_and_baselines() -> None:
    budget = TuneBudget(max_rounds=2, num_branches=3, rollout_steps=2, max_response_surface_evaluations=12)
    ledger = BudgetLedger.from_budget(budget)
    assert ledger.start_round()
    assert ledger.consume(evaluations=3, candidates=3, gpu_hours=0.1, branch_rollouts=3)
    assert ledger.to_dict()["response_surface_evaluations"] == 3


def test_budget_parity_gate_detects_unfair_budget() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.8, "response_surface_evaluations": 144},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.82, "response_surface_evaluations": 12},
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert next(g for g in gates["gates"] if g["gate_name"] == "budget_parity")["status"] == "failed"


def test_cost_adjusted_metric_computed(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_cost_tradeoff", tmp_path, rows=128, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("square_tune_full", df, mechanism_name="cost_tradeoff", seed=101, budget=TuneBudget(max_rounds=2))
    assert "cost_adjusted_improvement" in result.metrics


def test_linear_control_ground_truth_is_linear(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_linear_control", tmp_path, rows=256, seed=101, noise_level=0.0)
    df = pd.read_parquet(Path(row["path"]) / "data.parquet")
    corr = df[["target_utility", "feature_data_quality", "feature_regression_risk"]].corr()["target_utility"]
    assert corr["feature_data_quality"] > 0.70
    assert corr["feature_regression_risk"] < -0.25


def test_linear_control_simple_baseline_competitive(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_linear_control", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=3, num_branches=3, rollout_steps=1)
    linear = run_optimizer("linear_utility_optimizer", df, mechanism_name="linear_control", seed=101, budget=budget)
    full = run_optimizer("square_tune_full", df, mechanism_name="linear_control", seed=101, budget=budget)
    assert linear.metrics["final_utility"] >= full.metrics["final_utility"] - 0.01


def test_square_tune_no_privileged_linear_control_access(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_linear_control", tmp_path, rows=128, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "data.parquet")
    assert not any(col.startswith("latent_") for col in df.columns)


def test_merge_required_single_branch_insufficient(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_merge_required", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    no_merge = run_optimizer("square_tune_no_merge", df, mechanism_name="merge_required", seed=101, budget=TuneBudget(max_rounds=3))
    assert no_merge.metrics["merge_enabled"] is False


def test_pareto_merge_beats_no_merge_on_merge_required_smoke(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_merge_required", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=4, num_branches=4, rollout_steps=2)
    full = run_optimizer("square_tune_full", df, mechanism_name="merge_required", seed=101, budget=budget)
    no_merge = run_optimizer("square_tune_no_merge", df, mechanism_name="merge_required", seed=101, budget=budget)
    assert full.metrics["final_utility"] > no_merge.metrics["final_utility"]


def test_constraint_gated_merge_rejects_regressive_branch() -> None:
    from square_sim.tune.simulator.response_surface import SimulatedOutcome

    current = _state(safety=0.80)
    bad = _state(domain_accuracy=0.95, safety=0.60)
    good = _state(domain_accuracy=0.70, safety=0.82)
    outcomes = [
        SimulatedOutcome(bad, 0.9, 0.9, 0.9, 1, 0.1, {}, 0.1, "bad"),
        SimulatedOutcome(good, 0.7, 0.7, 0.1, 0, 0.1, {}, 0.1, "good"),
    ]
    merged, weights = _merge_outcomes(current, outcomes, [0.9, 0.7], merge_enabled=True, strategy="constraint_gated_pareto_merge")
    assert weights[0] == 0.0
    assert merged.eval_vector["safety"] >= 0.80


def test_metric_specific_merge_combines_non_conflicting_improvements() -> None:
    from square_sim.tune.simulator.response_surface import SimulatedOutcome

    current = _state()
    a = _state(domain_accuracy=0.75)
    b = _state(safety=0.85)
    merged, weights = _merge_outcomes(
        current,
        [
            SimulatedOutcome(a, 0.7, 0.7, 0.1, 0, 0.1, {}, 0.1, "a"),
            SimulatedOutcome(b, 0.7, 0.7, 0.1, 0, 0.1, {}, 0.1, "b"),
        ],
        [0.7, 0.7],
        merge_enabled=True,
        strategy="metric_specific_merge",
    )
    assert sum(weights) == 1.0
    assert merged.eval_vector["domain_accuracy"] >= 0.75
    assert merged.eval_vector["safety"] >= 0.85


def test_memory_required_full_must_beat_no_memory(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_repeated_regression_memory", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    full = run_optimizer("square_tune_full", df, mechanism_name="repeated_regression_memory", seed=101, budget=TuneBudget(max_rounds=3))
    no_memory = run_optimizer("square_tune_no_memory", df, mechanism_name="repeated_regression_memory", seed=101, budget=TuneBudget(max_rounds=3))
    assert full.metrics["repeated_bad_action_count"] < no_memory.metrics["repeated_bad_action_count"]


def test_repeated_regression_memory_records_bad_action(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_repeated_regression_memory", tmp_path, rows=256, seed=202)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("square_tune_full", df, mechanism_name="repeated_regression_memory", seed=202, budget=TuneBudget(max_rounds=4))
    assert result.metrics["known_bad_items_count"] >= 0
    assert "repeated_bad_action_count" in result.metrics


def test_no_memory_repeats_bad_action(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_repeated_regression_memory", tmp_path, rows=256, seed=303)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("square_tune_no_memory", df, mechanism_name="repeated_regression_memory", seed=303, budget=TuneBudget(max_rounds=3))
    assert result.metrics["repeated_bad_action_count"] >= 1


def test_regression_veto_full_beats_no_regression_sensor(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_regression_veto", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    budget = TuneBudget(max_rounds=3)
    full = run_optimizer("square_tune_full", df, mechanism_name="regression_veto", seed=101, budget=budget)
    no_reg = run_optimizer("square_tune_no_regression_sensor", df, mechanism_name="regression_veto", seed=101, budget=budget)
    assert full.metrics["protected_utility"] > no_reg.metrics["protected_utility"]


def test_regression_veto_rejects_high_raw_low_safety_branch() -> None:
    test_constraint_gated_merge_rejects_regressive_branch()


def test_no_regression_sensor_accepts_regressive_branch(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_regression_veto", tmp_path, rows=256, seed=202)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("square_tune_no_regression_sensor", df, mechanism_name="regression_veto", seed=202, budget=TuneBudget(max_rounds=3))
    assert result.metrics["regression_count"] >= 2


def test_regression_certificate_requires_protected_metric_improvement() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_regression_veto", "optimizer_name": "square_tune_full", "final_utility": 0.70, "protected_utility": 0.40},
            {"dataset_key": "synthetic_llm_regression_veto", "optimizer_name": "square_tune_no_regression_sensor", "final_utility": 0.69, "protected_utility": 0.50},
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert "regression_awareness" in gates["failed_gates"]


def test_cost_tradeoff_cost_aware_beats_no_cost_sensor_on_cost_adjusted_utility(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_cost_tradeoff", tmp_path, rows=256, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    full = run_optimizer("square_tune_full", df, mechanism_name="cost_tradeoff", seed=101, budget=TuneBudget(max_rounds=3))
    no_cost = run_optimizer("square_tune_no_cost_sensor", df, mechanism_name="cost_tradeoff", seed=101, budget=TuneBudget(max_rounds=3))
    assert full.metrics["cost_adjusted_improvement"] > no_cost.metrics["cost_adjusted_improvement"]


def test_cost_tradeoff_raw_best_not_cost_best(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_cost_tradeoff", tmp_path, rows=256, seed=202)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    no_cost = run_optimizer("square_tune_no_cost_sensor", df, mechanism_name="cost_tradeoff", seed=202, budget=TuneBudget(max_rounds=3))
    full = run_optimizer("square_tune_full", df, mechanism_name="cost_tradeoff", seed=202, budget=TuneBudget(max_rounds=3))
    assert full.metrics["cost_adjusted_improvement"] >= no_cost.metrics["cost_adjusted_improvement"]


def test_cost_aware_selects_lower_cost_branch(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_cost_tradeoff", tmp_path, rows=128, seed=303)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    full = run_optimizer("square_tune_full", df, mechanism_name="cost_tradeoff", seed=303, budget=TuneBudget(max_rounds=2))
    assert full.metrics["simulated_gpu_hours"] > 0


def test_no_cost_sensor_overpends_budget(tmp_path: Path) -> None:
    row = generate_dataset("synthetic_llm_cost_tradeoff", tmp_path, rows=128, seed=404)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    result = run_optimizer("square_tune_no_cost_sensor", df, mechanism_name="cost_tradeoff", seed=404, budget=TuneBudget(max_rounds=2))
    assert result.metrics["cost_adjusted_improvement"] < result.metrics["final_utility"]


def test_oracle_leakage_gate_detects_latent_columns() -> None:
    df = _metrics_frame(
        [
            {
                "dataset_key": "synthetic_llm_merge_required",
                "optimizer_name": "greedy_oracle_feature_baseline",
                "final_utility": 0.8,
                "latent_columns_used": "latent_merge_group",
            }
        ]
    )
    gates = evaluate_calibration_gates(df)
    assert "oracle_leakage" in gates["failed_gates"]


def test_calibration_v2_smoke_config_loads() -> None:
    cfg = TuneExperimentConfig.from_path(Path("configs/tune/square_tune_calibration_v2_smoke.yaml"))
    assert "synthetic_llm_linear_control" in cfg.datasets
    assert "linear_utility_optimizer" in cfg.optimizers


def test_calibration_v2_controls_only_matrix_expands() -> None:
    cfg = TuneExperimentConfig.from_path(Path("configs/tune/square_tune_calibration_v2_controls_only.yaml"))
    planned = [(d, s, o) for d in cfg.datasets for s in cfg.seeds for o in cfg.optimizers]
    assert len(planned) == 6 * 3 * 13


def test_calibration_report_written(tmp_path: Path) -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.82},
        ]
    )
    out = write_calibration_reports(tmp_path, "exp", df)
    assert Path(out["output_dir"], "calibration_gate_report.md").exists()


def test_calibration_certificate_written(tmp_path: Path) -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.80, "control_type": "classical_control"},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.82, "control_type": "classical_control"},
        ]
    )
    index = write_certificates(tmp_path, "exp", df)
    assert index["certificate_count"] == 1
    assert (tmp_path / "certificate_index.md").exists()


def test_run_all_v2_stops_if_gates_fail() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.90},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.80},
        ]
    )
    assert evaluate_calibration_gates(df)["global_status"] == "failed"


def test_run_all_v2_continues_if_gates_pass() -> None:
    df = _metrics_frame(
        [
            {"dataset_key": "synthetic_llm_random_label", "optimizer_name": "square_tune_full", "final_utility": 0.49},
            {"dataset_key": "synthetic_llm_random_label", "optimizer_name": "random_search", "final_utility": 0.51},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_linear_control", "optimizer_name": "linear_utility_optimizer", "final_utility": 0.82},
            {"dataset_key": "synthetic_llm_merge_required", "optimizer_name": "square_tune_full", "final_utility": 0.80},
            {"dataset_key": "synthetic_llm_merge_required", "optimizer_name": "square_tune_no_merge", "final_utility": 0.70},
            {"dataset_key": "synthetic_llm_repeated_regression_memory", "optimizer_name": "square_tune_full", "final_utility": 0.80, "preserved_known_good_score": 0.90, "repeated_bad_action_count": 0},
            {"dataset_key": "synthetic_llm_repeated_regression_memory", "optimizer_name": "square_tune_no_memory", "final_utility": 0.70, "preserved_known_good_score": 0.70, "repeated_bad_action_count": 2},
            {"dataset_key": "synthetic_llm_regression_veto", "optimizer_name": "square_tune_full", "final_utility": 0.80, "protected_utility": 0.80, "regression_count": 0},
            {"dataset_key": "synthetic_llm_regression_veto", "optimizer_name": "square_tune_no_regression_sensor", "final_utility": 0.70, "protected_utility": 0.60, "regression_count": 2},
            {"dataset_key": "synthetic_llm_cost_tradeoff", "optimizer_name": "square_tune_full", "final_utility": 0.80, "cost_adjusted_improvement": 0.20},
            {"dataset_key": "synthetic_llm_cost_tradeoff", "optimizer_name": "square_tune_no_cost_sensor", "final_utility": 0.70, "cost_adjusted_improvement": 0.10},
        ]
    )
    assert evaluate_calibration_gates(df)["global_status"] == "passed"


def test_calibration_run_matrix_smoke(tmp_path: Path) -> None:
    root = tmp_path / "nas" / "SQUARE" / "source-validation-workspace"
    data_root = root / "datasets" / "synthetic" / "square_tune_calibration_v2"
    generate_suite(data_root, rows=64, seeds=[101], datasets=["synthetic_llm_linear_control"])
    cfg_path = tmp_path / "calibration_smoke.yaml"
    cfg_path.write_text(
        f"""experiment_name: calibration_test
dataset_root: {data_root}
protocol_path: protocols/square_tune/synthetic_calibration_protocol_v2.yaml
datasets: [synthetic_llm_linear_control]
seeds: [101]
optimizers: [linear_utility_optimizer, square_tune_full]
square_tune:
  max_rounds: 2
  num_branches: 2
  rollout_steps: 1
  max_response_surface_evaluations: 16
  max_candidate_actions: 16
""",
        encoding="utf-8",
    )
    summary = run_tune_matrix(Settings.from_env(tmp_path / "nas"), cfg_path, skip_completed=False)
    assert summary["completed"] == 2
