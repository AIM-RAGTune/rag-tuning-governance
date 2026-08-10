from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_matched_cost.certificates import evaluate_certificate
from square_sim.square_tune_matched_cost.config import MatchedCostRAGConfig
from square_sim.square_tune_matched_cost.datasets import (
    fixture_rag_dataset,
    ingest_matched_cost_rag,
    normalize_rag_frame,
)
from square_sim.square_tune_matched_cost.gating_baselines import (
    random_gating_mask,
    retrieval_confidence_gating_mask,
    uncertainty_gating_mask,
)
from square_sim.square_tune_matched_cost.matched_cost import evaluate_system, expensive_compute_rate
from square_sim.square_tune_matched_cost.metrics import budget_deviation_pct, cost_adjusted_utility
from square_sim.square_tune_matched_cost.paths import publication_root, reports_root
from square_sim.square_tune_matched_cost.policy_space import RAG_POLICY_SPACE
from square_sim.square_tune_matched_cost.publication_bundle import create_publication_bundle
from square_sim.square_tune_matched_cost.runner import protect_prior, run_matrix
from square_sim.square_tune_matched_cost.scenario_compile import (
    compile_matched_cost_scenario,
    duplicate_query_leakage,
)
from square_sim.square_tune_matched_cost.statistics import aggregate_statistics, utility_sensitivity
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import write_json, write_text
from square_sim.utils.write_once import WriteOncePathManager


def _settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("AIM_NAS_ROOT", str(tmp_path / "nas"))
    monkeypatch.setenv("SQUARESIM_DATA_ROOT", str(tmp_path / "lab"))
    return Settings.from_env()


def test_matched_cost_config_loads() -> None:
    cfg = MatchedCostRAGConfig.from_path(Path("configs/matched_cost_rag/square_tune_matched_cost_rag_v1_full_matrix.yaml"))
    assert len(cfg.planned_runs()) == 90
    assert "random_gating_matched_cost" in cfg.systems


def test_protected_prior_paths_block_overwrite(tmp_path: Path, monkeypatch) -> None:
    s = _settings(tmp_path, monkeypatch)
    prior = s.project_root / "reports" / "square_tune" / "calibration"
    prior.mkdir(parents=True)
    protect_prior(s)
    manager = WriteOncePathManager(prior, ProtectedResultsRegistry(s).protected_paths())
    try:
        manager.ensure_writable_path(prior / "x")
    except Exception as exc:
        assert "protected" in str(exc)
    else:
        raise AssertionError("protected path write was not blocked")


def test_fixture_rag_dataset_normalizes(tmp_path: Path) -> None:
    source = pd.DataFrame({"question": ["q"], "answer": ["a"], "context": ["c"], "faithfulness": [0.8]})
    frame = normalize_rag_frame(source, source_path=tmp_path / "source.parquet", source_dataset="ragbench")
    assert {"example_id", "query", "base_quality", "uncertainty"}.issubset(frame.columns)


def test_scenario_compile_train_val_test(tmp_path: Path, monkeypatch) -> None:
    s = _settings(tmp_path, monkeypatch)
    ingest_matched_cost_rag(s, max_rows=120)
    manifest = compile_matched_cost_scenario(s)
    assert manifest["train_count"] > manifest["validation_count"] > 0
    assert manifest["test_count"] > 0


def test_duplicate_query_leakage_detection() -> None:
    splits = {
        "train": pd.DataFrame({"query": ["a", "b"]}),
        "test": pd.DataFrame({"query": ["b", "c"]}),
    }
    assert duplicate_query_leakage(splits)["leakage_count"] == 1


def test_rag_policy_space_valid() -> None:
    assert "top_k" in RAG_POLICY_SPACE
    assert "verification_policy" in RAG_POLICY_SPACE


def test_budget_ledger_matches_costs() -> None:
    assert budget_deviation_pct(0.205, 0.20) <= 2.5


def test_random_gating_matches_invocation_rate() -> None:
    frame = fixture_rag_dataset(100)
    assert abs(random_gating_mask(frame, 0.2, 101).mean() - 0.2) < 0.01


def test_uncertainty_gating_matches_invocation_rate() -> None:
    frame = fixture_rag_dataset(100)
    assert abs(uncertainty_gating_mask(frame, 0.2).mean() - 0.2) < 0.01


def test_retrieval_confidence_gating_matches_invocation_rate() -> None:
    frame = fixture_rag_dataset(100)
    assert abs(retrieval_confidence_gating_mask(frame, 0.2).mean() - 0.2) < 0.01


def test_hpo_baseline_respects_budget() -> None:
    frame = fixture_rag_dataset(160)
    rate = expensive_compute_rate(frame)
    result = evaluate_system(system="optuna_tpe_matched_budget_optional", seed=101, validation=frame, test=frame)
    assert abs(result.metrics["expensive_compute_invocation_rate"] - rate) <= 0.02


def test_cost_adjusted_utility_computed() -> None:
    value = cost_adjusted_utility(quality=0.8, cost=0.3, latency=0.2, regression=0.1, weights={"quality": 1, "cost": 0.25, "latency": 0.1, "regression": 0.5})
    assert value == 0.655


def test_utility_sensitivity_runs() -> None:
    frame = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "held_out_test_raw_quality": 0.8, "total_cost_proxy": 0.4, "simulated_latency_cost": 0.2, "regression_count": 0.1},
            {"system": "random_gating_matched_cost", "held_out_test_raw_quality": 0.7, "total_cost_proxy": 0.4, "simulated_latency_cost": 0.2, "regression_count": 0.1},
        ]
    )
    assert utility_sensitivity(frame)["weight_setting"].nunique() >= 5


def test_bootstrap_intervals_generated() -> None:
    frame = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "seed": 101, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "total_cost_proxy": 0.4, "expensive_compute_invocation_rate": 0.2, "positive_expensive_compute_roi_rate": 1.0},
            {"system": "random_gating_matched_cost", "seed": 101, "held_out_test_cost_adjusted_utility": 0.4, "held_out_test_raw_quality": 0.7, "total_cost_proxy": 0.4, "expensive_compute_invocation_rate": 0.2, "positive_expensive_compute_roi_rate": 1.0},
        ]
    )
    assert not aggregate_statistics(frame, bootstrap_samples=10)["bootstrap_intervals"].empty


def test_certificate_refuses_when_random_gating_wins() -> None:
    metrics = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.4, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "random_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "uncertainty_threshold_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.3, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "retrieval_confidence_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.3, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "square_tune_no_fork", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.3, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
        ]
    )
    assert evaluate_certificate(metrics, pd.DataFrame(), no_overwrite_status="append_only_confirmed")["status"] == "Negative result"


def test_certificate_refuses_when_uncertainty_gating_wins() -> None:
    metrics = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.4, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "random_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.3, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "uncertainty_threshold_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
        ]
    )
    assert evaluate_certificate(metrics, pd.DataFrame(), no_overwrite_status="append_only_confirmed")["status"] == "Negative result"


def test_certificate_budget_confounded_on_budget_mismatch() -> None:
    metrics = pd.DataFrame([{"system": "square_tune_adaptive_compute", "real_data_used": True, "budget_confounded_flag": True}])
    assert evaluate_certificate(metrics, pd.DataFrame(), no_overwrite_status="append_only_confirmed")["status"] == "Budget confounded"


def test_certificate_candidate_on_mixed_results() -> None:
    metrics = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.6, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "random_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "uncertainty_threshold_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "retrieval_confidence_gating_matched_cost", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
            {"system": "square_tune_no_fork", "seed": 101, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False},
        ]
    )
    cert = evaluate_certificate(metrics, pd.DataFrame(), no_overwrite_status="append_only_confirmed")
    assert cert["status"] in {"Candidate signal", "Inconclusive", "Utility fragile"}


def test_certificate_supported_on_clear_matched_cost_win() -> None:
    rows = []
    for seed in [101, 202, 303, 404, 505]:
        rows.append({"system": "square_tune_adaptive_compute", "seed": seed, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.8, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False})
        for system in ["random_gating_matched_cost", "uncertainty_threshold_gating_matched_cost", "retrieval_confidence_gating_matched_cost", "square_tune_no_fork"]:
            rows.append({"system": system, "seed": seed, "real_data_used": True, "held_out_test_cost_adjusted_utility": 0.5, "held_out_test_raw_quality": 0.8, "budget_confounded_flag": False})
    sensitivity = pd.DataFrame([{"weight_setting": f"w{i}", "system": "square_tune_adaptive_compute", "rank": 1} for i in range(4)])
    assert evaluate_certificate(pd.DataFrame(rows), sensitivity, no_overwrite_status="append_only_confirmed")["status"] == "Signal supported"


def test_publication_bundle_excludes_raw_data(tmp_path: Path, monkeypatch) -> None:
    s = _settings(tmp_path, monkeypatch)
    exp = "square_tune_matched_cost_rag_v1_fixture_20260802-000000-deadbeef"
    report = reports_root(s) / exp
    cert = s.project_root / "certificates" / "square_tune" / "matched_cost_rag" / "v1" / exp
    report.mkdir(parents=True)
    cert.mkdir(parents=True)
    write_text(report / "executive_summary.md", "summary")
    write_json(cert / "certificate.json", {"status": "Inconclusive"})
    bundle = create_publication_bundle(s, exp, publication_root(s) / exp)
    assert bundle["raw_data_excluded"]


def test_no_positive_certificate_without_real_data() -> None:
    metrics = pd.DataFrame([{"system": "square_tune_adaptive_compute", "real_data_used": False, "budget_confounded_flag": False}])
    assert evaluate_certificate(metrics, pd.DataFrame(), no_overwrite_status="append_only_confirmed")["status"] == "Data unavailable"


def test_smoke_run_completes(tmp_path: Path, monkeypatch) -> None:
    s = _settings(tmp_path, monkeypatch)
    ingest_matched_cost_rag(s, max_rows=120)
    compile_matched_cost_scenario(s, max_rows=120)
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(
        """
matrix_name: smoke_fixture
seeds: [101]
systems:
  - static_default_rag_policy
  - square_tune_no_fork
  - square_tune_adaptive_compute
  - random_gating_matched_cost
  - uncertainty_threshold_gating_matched_cost
simulation:
  max_rows: 120
  bootstrap_samples: 10
real_data_required: false
""",
        encoding="utf-8",
    )
    result = run_matrix(s, cfg)
    assert result["succeeded"] == 5
    assert Path(result["reports_dir"]).joinpath("executive_summary.md").exists()
