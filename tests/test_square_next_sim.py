from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from square_sim.config import Settings
from square_sim.next_sim.adaptive_escalation.policies import decide_escalation
from square_sim.next_sim.certificates import evaluate_track_certificate
from square_sim.next_sim.claim_faithfulness.claim_extraction_proxy import build_claim_proxy
from square_sim.next_sim.config import NextSimConfig
from square_sim.next_sim.elastic_compute.synthetic_trace import generate_synthetic_trace
from square_sim.next_sim.protected_results import protect_prior
from square_sim.next_sim.publication_bundle import create_publication_bundle
from square_sim.next_sim.rag_hard_subset.subset_detection import (
    detect_hard_subsets,
    subset_availability,
)
from square_sim.next_sim.runner import _fixture_rag_frame, evaluate_run, plan_matrix, run_matrix
from square_sim.next_sim.square_core_v2.closed_loop_v2 import closed_loop_metrics
from square_sim.next_sim.square_core_v2.field_substrate_v2 import field_metrics
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.write_once import WriteOnceError, WriteOncePathManager


def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(tmp_path)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_next_sim_config_loads() -> None:
    cfg = NextSimConfig.from_path(Path("configs/next_sim/square_next_sim_v1_smoke.yaml"))
    assert "rag_hard_subset_v1" in cfg.tracks
    assert cfg.planned_runs()


def test_next_sim_plan_expands() -> None:
    plan = plan_matrix(Path("configs/next_sim/square_next_sim_v1_smoke.yaml"))
    assert plan["planned"] == 21


def test_next_sim_protected_paths_block_write(tmp_path: Path) -> None:
    s = settings(tmp_path)
    protected = s.project_root / "reports" / "square_tune" / "matched_cost_rag" / "v1" / "square_tune_matched_cost_rag_v1_full_matrix_20260802-232631-5449febfb3"
    protected.mkdir(parents=True)
    protect_prior(s)
    manager = WriteOncePathManager(tmp_path, ProtectedResultsRegistry(s).protected_paths())
    with pytest.raises(WriteOnceError):
        manager.ensure_writable_path(protected / "new.json")


def test_next_sim_smoke_matrix_runs_on_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = settings(tmp_path)
    monkeypatch.setattr("square_sim.next_sim.runner.latest_scenario_manifest", lambda _settings: None)
    config = write_config(
        tmp_path,
        """
matrix_name: smoke_fixture
seeds: [101]
tracks: [elastic_compute_policy_v1]
systems_by_track:
  elastic_compute_policy_v1: [static_threshold_policy, square_tune_adaptive_compute]
""",
    )
    result = run_matrix(s, config)
    assert result["planned"] == 2
    assert result["succeeded"] == 2
    assert Path(result["reports_dir"], "executive_summary.md").exists()


def test_rag_hard_subset_membership_rules() -> None:
    frame = _fixture_rag_frame(101, rows=200)
    masks, profile = detect_hard_subsets(frame)
    assert profile["available"] is True
    assert masks["hard_composite"].sum() > 0


def test_rag_hard_subset_unavailable_if_metadata_missing() -> None:
    frame = pd.DataFrame({"query": ["a"]})
    assert subset_availability(frame)["available"] is False


def test_no_fork_robustness_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("square_sim.next_sim.runner.latest_scenario_manifest", lambda _settings: None)
    metrics = evaluate_run(settings(tmp_path), "no_fork_robustness_v1", "square_tune_no_fork", 101)
    assert "distance_to_oracle" in metrics


def test_adaptive_escalation_tier_logic() -> None:
    decision = decide_escalation(uncertainty=0.9, retrieval_conflict=0.86, hallucination_risk=0.8, budget_pressure=0.1)
    assert decision.tier == 3
    low = decide_escalation(uncertainty=0.2, retrieval_conflict=0.2, hallucination_risk=0.2, budget_pressure=0.2)
    assert low.tier == 1


def test_claim_level_proxy_pipeline() -> None:
    claims = build_claim_proxy(_fixture_rag_frame(101, rows=10))
    assert {"claim_id", "unsupported_claim_risk", "high_risk_claim"}.issubset(claims.columns)


def test_elastic_compute_synthetic_trace_generation() -> None:
    trace = generate_synthetic_trace(seed=101, rows=50)
    assert len(trace) == 50
    assert trace["demand"].between(0, 1).all()


def test_elastic_compute_metrics(tmp_path: Path) -> None:
    metrics = evaluate_run(settings(tmp_path), "elastic_compute_policy_v1", "square_tune_adaptive_compute", 101)
    assert metrics["SLO_violation_rate"] < 0.2


def test_square_core_v2_field_substrate_config() -> None:
    metrics = field_metrics("square_field_crosstalk_aware", 101)
    assert metrics["target_field_error"] < field_metrics("random_emitter_activation", 101)["target_field_error"]


def test_square_core_v2_closed_loop_config() -> None:
    metrics = closed_loop_metrics("square_adaptive_controller_with_topology", 101)
    assert metrics["recovery_time"] < closed_loop_metrics("open_loop_script", 101)["recovery_time"]


def test_certificate_refused_when_no_fork_beats_adaptive_everywhere() -> None:
    frame = pd.DataFrame(
        [
            {"track": "rag_hard_subset_v1", "system": "square_tune_no_fork", "cost_adjusted_utility": 0.5, "hard_subset_performance": 0.8},
            {"track": "rag_hard_subset_v1", "system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.4, "hard_subset_performance": 0.7},
        ]
    )
    assert evaluate_track_certificate("rag_hard_subset_v1", frame)["status"] == "Negative result"


def test_certificate_candidate_when_adaptive_wins_hard_subset() -> None:
    frame = pd.DataFrame(
        [
            {"track": "rag_hard_subset_v1", "system": "square_tune_no_fork", "cost_adjusted_utility": 0.5, "hard_subset_performance": 0.7},
            {"track": "rag_hard_subset_v1", "system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.4, "hard_subset_performance": 0.8},
        ]
    )
    assert evaluate_track_certificate("rag_hard_subset_v1", frame)["status"] == "Candidate signal"


def test_certificate_supported_when_escalation_beats_no_fork_and_full() -> None:
    frame = pd.DataFrame(
        [
            {"track": "adaptive_escalation_v2", "system": "square_tune_no_fork", "cost_adjusted_utility": 0.5},
            {"track": "adaptive_escalation_v2", "system": "square_tune_full", "cost_adjusted_utility": 0.4},
            {"track": "adaptive_escalation_v2", "system": "square_tune_hard_subset_escalation", "cost_adjusted_utility": 0.6},
        ]
    )
    assert evaluate_track_certificate("adaptive_escalation_v2", frame)["status"] == "Signal supported"


def test_publication_bundle_excludes_raw_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = settings(tmp_path)
    monkeypatch.setattr("square_sim.next_sim.runner.latest_scenario_manifest", lambda _settings: None)
    config = write_config(tmp_path, "matrix_name: pub\nseeds: [101]\ntracks: [elastic_compute_policy_v1]\nsystems_by_track:\n  elastic_compute_policy_v1: [static_threshold_policy]\n")
    result = run_matrix(s, config)
    bundle = create_publication_bundle(s, result["experiment_id"], tmp_path / "bundle")
    assert bundle["raw_data_included"] is False


def test_no_overwrite_audit_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = settings(tmp_path)
    monkeypatch.setattr("square_sim.next_sim.runner.latest_scenario_manifest", lambda _settings: None)
    config = write_config(tmp_path, "matrix_name: audit\nseeds: [101]\ntracks: [elastic_compute_policy_v1]\nsystems_by_track:\n  elastic_compute_policy_v1: [static_threshold_policy]\n")
    result = run_matrix(s, config)
    assert Path(result["reports_dir"], "no_overwrite_audit.json").exists()


def test_negative_result_preserved() -> None:
    frame = pd.DataFrame(
        [
            {"track": "rag_hard_subset_v1", "system": "square_tune_no_fork", "cost_adjusted_utility": 1.0, "hard_subset_performance": 1.0},
            {"track": "rag_hard_subset_v1", "system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.1, "hard_subset_performance": 0.1},
        ]
    )
    cert = evaluate_track_certificate("rag_hard_subset_v1", frame)
    assert cert["status"] == "Negative result"
