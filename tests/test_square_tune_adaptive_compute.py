from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.tune.config import TuneBudget
from square_sim.tune.external.certificate import certificate_for_scenario
from square_sim.tune.simulator.adaptive_compute import ComputeGatePolicy, fork_roi, summarize_compute_gate
from square_sim.tune.simulator.square_tune_optimizer import run_optimizer
from square_sim.tune.simulator.state import initial_state_from_frame
from square_sim.tune.synthetic.generators import generate_dataset


def _state(tmp_path: Path, dataset: str = "synthetic_llm_rag_policy_conflict"):
    row = generate_dataset(dataset, tmp_path, rows=192, seed=101)
    df = pd.read_parquet(Path(row["path"]) / "train.parquet")
    return df, initial_state_from_frame(df, 101)


def test_adaptive_compute_policy_low_uncertainty_uses_cheap_search(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    state.failure_clusters = {"easy": 0.02}
    state.eval_vector["retrieval_faithfulness"] = 0.95
    decision = ComputeGatePolicy("square_tune_adaptive_compute").decide(
        state=state,
        round_idx=1,
        selected_region="easy",
        scenario_family="rag_policy_conflict",
        max_rounds=8,
        num_branches=6,
    )
    assert decision.decision == "cheap_local_search"
    assert not decision.fork_invoked


def test_adaptive_compute_policy_high_uncertainty_invokes_fork(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    state.failure_clusters = {"hard": 0.95}
    decision = ComputeGatePolicy("square_tune_adaptive_compute_no_conflict_gate").decide(
        state=state,
        round_idx=1,
        selected_region="hard",
        scenario_family="rag_policy_conflict",
        max_rounds=8,
        num_branches=6,
    )
    assert decision.fork_invoked


def test_adaptive_compute_policy_conflict_invokes_merge(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    state.failure_clusters = {"conflict": 0.8}
    state.eval_vector["domain_accuracy"] = 0.95
    state.eval_vector["safety"] = 0.35
    decision = ComputeGatePolicy("square_tune_adaptive_compute").decide(
        state=state,
        round_idx=1,
        selected_region="conflict",
        scenario_family="rag_policy_conflict",
        max_rounds=8,
        num_branches=6,
    )
    assert decision.merge_invoked


def test_adaptive_compute_policy_budget_pressure_suppresses_fork(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    state.failure_clusters = {"late": 0.75}
    decision = ComputeGatePolicy("square_tune_adaptive_compute_no_conflict_gate").decide(
        state=state,
        round_idx=7,
        selected_region="late",
        scenario_family="rag_policy_conflict",
        max_rounds=8,
        num_branches=6,
    )
    assert decision.decision in {"cheap_local_search", "single_branch_rollout", "regression_repair"}


def test_adaptive_compute_policy_regression_risk_invokes_regression_repair(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    state.failure_clusters = {"risk": 0.75}
    state.eval_vector["regression_score"] = 0.1
    decision = ComputeGatePolicy("square_tune_adaptive_compute").decide(
        state=state,
        round_idx=1,
        selected_region="risk",
        scenario_family="prompt_regression",
        max_rounds=8,
        num_branches=6,
    )
    assert decision.regression_repair_invoked


def test_compute_gate_decision_serialization(tmp_path: Path) -> None:
    _, state = _state(tmp_path)
    decision = ComputeGatePolicy("square_tune_adaptive_compute").decide(
        state=state,
        round_idx=0,
        selected_region=next(iter(state.failure_clusters)),
        scenario_family="rag_policy_conflict",
        max_rounds=8,
        num_branches=6,
    )
    payload = decision.to_dict()
    assert payload["round_idx"] == 0
    assert "expected_value_of_fork" in payload


def test_fork_roi_positive_when_gain_exceeds_cost() -> None:
    assert fork_roi(0.2, 0.1) > 0


def test_fork_roi_negative_when_cost_exceeds_gain() -> None:
    assert fork_roi(-0.02, 0.1) < 0


def test_adaptive_compute_not_always_fork_by_default(tmp_path: Path) -> None:
    df, _ = _state(tmp_path)
    result = run_optimizer(
        "square_tune_adaptive_compute",
        df,
        mechanism_name="rag_policy_conflict",
        seed=101,
        budget=TuneBudget(max_rounds=4, num_branches=4, rollout_steps=2),
    )
    assert 0.0 < result.metrics["fork_invocation_rate"] < 1.0
    assert not result.adaptive_diagnostics.empty


def test_adaptive_compute_variant_no_uncertainty_gate_changes_behavior(tmp_path: Path) -> None:
    df, _ = _state(tmp_path)
    budget = TuneBudget(max_rounds=4, num_branches=4, rollout_steps=2)
    base = run_optimizer("square_tune_adaptive_compute", df, mechanism_name="rag_policy_conflict", seed=101, budget=budget)
    variant = run_optimizer(
        "square_tune_adaptive_compute_no_uncertainty_gate",
        df,
        mechanism_name="rag_policy_conflict",
        seed=101,
        budget=budget,
    )
    assert not base.adaptive_diagnostics["decision"].equals(variant.adaptive_diagnostics["decision"])


def test_degenerate_flags() -> None:
    no_fork = summarize_compute_gate([{"decision": "cheap_local_search", "fork_invoked": False} for _ in range(4)])
    full = summarize_compute_gate([{"decision": "multi_branch_fork", "fork_invoked": True, "fork_roi": 1.0} for _ in range(4)])
    assert "behaves_like_no_fork" in no_fork["degenerate_behavior_flag"]
    assert "behaves_like_full" in full["degenerate_behavior_flag"]


def test_adaptive_certificate_candidate_when_no_fork_still_best() -> None:
    rows = pd.DataFrame(
        [
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "greedy_regression_aware", "cost_adjusted_improvement": 0.01, "final_utility": 0.8, "response_surface_evaluations": 144, "source_appropriate": True, "source_license_status": "captured"},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_full", "cost_adjusted_improvement": 0.04, "final_utility": 0.93, "response_surface_evaluations": 144, "source_appropriate": True, "source_license_status": "captured"},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_no_fork", "cost_adjusted_improvement": 0.09, "final_utility": 0.91, "response_surface_evaluations": 144, "source_appropriate": True, "source_license_status": "captured"},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_adaptive_compute", "cost_adjusted_improvement": 0.06, "final_utility": 0.925, "response_surface_evaluations": 144, "source_appropriate": True, "source_license_status": "captured", "fork_invocation_rate": 0.3, "positive_fork_roi_rate": 0.7},
        ]
    )
    cert = certificate_for_scenario("rag_policy_optimization", rows, calibration={"status": "passed"})
    assert cert["status"] == "Candidate adaptive external signal"


def test_adaptive_config_loads() -> None:
    path = Path("configs/tune/external_transfer/external_transfer_v2_adaptive_compute_smoke.yaml")
    assert "square_tune_adaptive_compute" in path.read_text(encoding="utf-8")
