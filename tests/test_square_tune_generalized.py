from __future__ import annotations

from pathlib import Path

import pandas as pd

from square_sim.config import Settings
from square_sim.square_tune_generalized.config import GeneralizedConfig
from square_sim.square_tune_generalized.datasets.classical_ml_hybrid import (
    generate_ml_to_llm_hybrid,
)
from square_sim.square_tune_generalized.datasets.elastic_compute import (
    generate_elastic_compute_trace,
)
from square_sim.square_tune_generalized.datasets.licenses import license_record
from square_sim.square_tune_generalized.datasets.manifests import write_dataset_artifacts
from square_sim.square_tune_generalized.datasets.patient_flow import (
    generate_patient_flow_synthetic_proxy,
    import_mimic_patient_flow,
)
from square_sim.square_tune_generalized.datasets.rag import generate_rag_proxy
from square_sim.square_tune_generalized.evaluation.compute_metrics import elastic_compute_metrics
from square_sim.square_tune_generalized.evaluation.hybrid_metrics import ml_to_llm_metrics
from square_sim.square_tune_generalized.evaluation.operations_metrics import patient_flow_metrics
from square_sim.square_tune_generalized.evaluation.rag_metrics import rag_proxy_metrics
from square_sim.square_tune_generalized.reporting.certificates import (
    certificate_for_group,
    write_generalized_certificates,
)
from square_sim.square_tune_generalized.reporting.publication_bundle import (
    create_publication_bundle,
)
from square_sim.square_tune_generalized.scenarios.compiler import compile_generalized_scenarios
from square_sim.square_tune_generalized.scenarios.rag_policy import RAG_POLICY_SPACE
from square_sim.square_tune_generalized.simulation.response_surfaces import (
    simulate_generalized_system,
)
from square_sim.square_tune_generalized.simulation.runner import run_generalized_matrix
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text


def _settings(tmp_path: Path, monkeypatch) -> Settings:
    monkeypatch.setenv("AIM_NAS_ROOT", str(tmp_path / "nas"))
    monkeypatch.setenv("SQUARESIM_DATA_ROOT", str(tmp_path / "lab"))
    return Settings.from_env()


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "smoke.yaml"
    path.write_text(
        """
matrix_name: smoke
tracks: [rag, patient_flow, elastic_compute, ml_to_llm]
seeds: [101]
scenarios:
  rag: [rag_policy_optimization]
  patient_flow: [ed_boarding_risk_explanation]
  elastic_compute: [autoscaling_threshold_policy]
  ml_to_llm: [prediction_plus_explanation]
systems:
  - static_default_policy
  - classical_only_baseline
  - threshold_policy_baseline
  - square_tune_no_fork
  - square_tune_no_cost_sensor
  - square_tune_adaptive_compute
simulation:
  rows_per_track: 80
  scenario_max_rows: 60
""",
        encoding="utf-8",
    )
    return path


def _write_track_dataset(root: Path, track: str, frame: pd.DataFrame) -> None:
    write_dataset_artifacts(
        root=root,
        dataset_key=f"{track}_fixture",
        track=track,
        rows=frame,
        license_metadata=license_record(
            dataset_key=f"{track}_fixture",
            source_type="fixture",
            license_status="captured",
            publication_safe=True,
        ),
        source_note="fixture",
    )


def _prepare_all(tmp_path: Path, monkeypatch) -> tuple[Settings, Path]:
    settings = _settings(tmp_path, monkeypatch)
    root = settings.project_root / "datasets" / "generalized" / "v1"
    _write_track_dataset(root, "rag", generate_rag_proxy(80))
    _write_track_dataset(root, "patient_flow", generate_patient_flow_synthetic_proxy(80))
    _write_track_dataset(root, "elastic_compute", generate_elastic_compute_trace(80))
    _write_track_dataset(root, "ml_to_llm", generate_ml_to_llm_hybrid(80))
    cfg = _config(tmp_path)
    compile_generalized_scenarios(
        cfg,
        dataset_root=root,
        output_root=settings.project_root / "scenarios" / "generalized" / "v1",
    )
    return settings, cfg


def test_generalized_config_loads(tmp_path: Path) -> None:
    cfg = GeneralizedConfig.from_path(_config(tmp_path))
    assert cfg.matrix_name == "smoke"
    assert len(cfg.planned_runs()) == 24


def test_freeze_evidence_manifest_written(tmp_path: Path) -> None:
    out = tmp_path / "evidence"
    out.mkdir()
    write_json(out / "evidence_manifest.json", {"prior_experiment_ids": ["x"]})
    assert read_json(out / "evidence_manifest.json")["prior_experiment_ids"] == ["x"]


def test_no_overwrite_protects_prior_results(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    prior = settings.project_root / "reports" / "square_core" / "v1" / "square_core_validation_v1_full_matrix_20260802-152150-d2d7bb0cc3"
    prior.mkdir(parents=True)
    registry = ProtectedResultsRegistry(settings)
    registry.protect_defaults()
    assert prior in registry.protected_paths()


def test_rag_policy_space_valid() -> None:
    assert "top_k" in RAG_POLICY_SPACE
    assert True in RAG_POLICY_SPACE["claim_level_check_enabled"]


def test_patient_flow_synthetic_proxy_generation() -> None:
    frame = generate_patient_flow_synthetic_proxy(25, 101)
    assert "boarding_risk" in frame
    assert frame["source_dataset"].eq("patient_flow_synthetic_proxy_v1").all()


def test_mimic_patient_flow_fixture_import(tmp_path: Path) -> None:
    mimic = tmp_path / "mimic-iv-ed"
    mimic.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "stay_id": [10, 20, 30],
            "intime": ["2160-01-01 00:00:00", "2160-01-01 00:30:00", "2160-01-01 03:00:00"],
            "outtime": ["2160-01-01 06:00:00", "2160-01-01 02:30:00", "2160-01-01 05:00:00"],
            "disposition": ["ADMITTED", "HOME", "TRANSFER"],
        }
    ).to_csv(mimic / "edstays.csv", index=False)
    pd.DataFrame({"subject_id": [1, 2, 3], "stay_id": [10, 20, 30], "acuity": [2, 4, 3]}).to_csv(
        mimic / "triage.csv",
        index=False,
    )
    frame = import_mimic_patient_flow(mimic)
    assert len(frame) == 3
    assert frame["source_dataset"].eq("mimic_iv_ed").all()
    assert frame["row_id"].str.startswith("mimic-").all()
    assert "subject_id" not in frame.columns


def test_patient_flow_forbids_clinical_claims_in_reports() -> None:
    cert = certificate_for_group("patient_flow", "ed_boarding_risk_explanation", pd.DataFrame())
    assert "not diagnosis" in cert["caveats"][0]


def test_elastic_compute_synthetic_trace_generation() -> None:
    frame = generate_elastic_compute_trace(20, 101)
    assert {"queue_time", "slo_violation", "cost"}.issubset(frame.columns)


def test_ml_to_llm_hybrid_fixture_compiles(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    root = settings.project_root / "datasets" / "generalized" / "v1"
    _write_track_dataset(root, "ml_to_llm", generate_ml_to_llm_hybrid(40))
    cfg = tmp_path / "hybrid.yaml"
    cfg.write_text("matrix_name: hybrid\ntracks: [ml_to_llm]\nseeds: [101]\nsimulation:\n  scenario_max_rows: 20\n", encoding="utf-8")
    report = compile_generalized_scenarios(cfg, dataset_root=root, output_root=settings.project_root / "scenarios" / "generalized" / "v1")
    assert report["scenario_count"] == 6


def test_scenario_card_written(tmp_path: Path, monkeypatch) -> None:
    settings, cfg = _prepare_all(tmp_path, monkeypatch)
    compile_generalized_scenarios(cfg, dataset_root=settings.project_root / "datasets" / "generalized" / "v1", output_root=settings.project_root / "scenarios2")
    assert list((settings.project_root / "scenarios2").glob("*/*/*/scenario_card.md"))


def test_license_summary_written(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    manifest = write_dataset_artifacts(
        root=root,
        dataset_key="rag_fixture",
        track="rag",
        rows=generate_rag_proxy(10),
        license_metadata=license_record(dataset_key="rag_fixture", source_type="fixture", license_status="captured", publication_safe=True),
        source_note="fixture",
    )
    assert Path(manifest["data_path"]).parent.joinpath("license_summary.json").exists()


def test_budget_parity_enforced() -> None:
    frame = pd.DataFrame(
        [
            {"system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.8, "budget_parity_ok": False},
            {"system": "static_default_policy", "cost_adjusted_utility": 0.1, "budget_parity_ok": False},
        ]
    )
    assert certificate_for_group("rag", "rag_policy_optimization", frame)["status"] == "Budget confounded"


def test_generalized_smoke_matrix_expands(tmp_path: Path) -> None:
    assert len(GeneralizedConfig.from_path(_config(tmp_path)).planned_runs()) == 24


def test_generalized_rag_robustness_config_expands() -> None:
    cfg = GeneralizedConfig.from_path(
        Path("configs/generalized/square_tune_generalized_v1_rag_robustness_cost_sensitivity.yaml")
    )
    assert cfg.stress_profiles == [
        "nominal",
        "high_cost_penalty",
        "strict_regression_gate",
        "low_uncertainty",
        "high_uncertainty",
        "tight_budget",
    ]
    assert len(cfg.planned_runs()) == 2100


def test_stress_profile_changes_cost_adjusted_utility() -> None:
    frame = generate_rag_proxy(100, 101)
    nominal, _ = simulate_generalized_system(
        "rag",
        "rag_policy_optimization",
        "square_tune_adaptive_compute",
        101,
        frame,
        stress_profile="nominal",
    )
    tight, _ = simulate_generalized_system(
        "rag",
        "rag_policy_optimization",
        "square_tune_adaptive_compute",
        101,
        frame,
        stress_profile="tight_budget",
    )
    assert tight["stress_profile"] == "tight_budget"
    assert tight["cost_adjusted_utility"] != nominal["cost_adjusted_utility"]


def test_stress_profile_written_to_run_manifest(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    root = settings.project_root / "datasets" / "generalized" / "v1"
    _write_track_dataset(root, "rag", generate_rag_proxy(80))
    cfg = tmp_path / "stress.yaml"
    cfg.write_text(
        """
matrix_name: stress_fixture
tracks: [rag]
seeds: [101]
scenarios:
  rag: [rag_policy_optimization]
systems:
  - static_default_policy
  - square_tune_adaptive_compute
stress_profiles:
  - nominal
  - tight_budget
simulation:
  scenario_max_rows: 40
""",
        encoding="utf-8",
    )
    compile_generalized_scenarios(
        cfg,
        dataset_root=root,
        output_root=settings.project_root / "scenarios" / "generalized" / "v1",
    )
    result = run_generalized_matrix(settings, cfg)
    manifests = list((Path(result["artifacts_dir"]) / "runs").glob("*/run_manifest.json"))
    stress_profiles = {read_json(path)["stress_profile"] for path in manifests}
    assert stress_profiles == {"nominal", "tight_budget"}


def test_certificate_groups_by_stress_profile(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "track": "rag",
                "scenario": "rag_policy_optimization",
                "stress_profile": "nominal",
                "system": "static_default_policy",
                "cost_adjusted_utility": 0.2,
                "budget_parity_ok": True,
            },
            {
                "track": "rag",
                "scenario": "rag_policy_optimization",
                "stress_profile": "nominal",
                "system": "square_tune_adaptive_compute",
                "cost_adjusted_utility": 0.7,
                "budget_parity_ok": True,
            },
            {
                "track": "rag",
                "scenario": "rag_policy_optimization",
                "stress_profile": "tight_budget",
                "system": "static_default_policy",
                "cost_adjusted_utility": 0.6,
                "budget_parity_ok": True,
            },
            {
                "track": "rag",
                "scenario": "rag_policy_optimization",
                "stress_profile": "tight_budget",
                "system": "square_tune_adaptive_compute",
                "cost_adjusted_utility": 0.4,
                "budget_parity_ok": True,
            },
        ]
    )
    index = write_generalized_certificates(tmp_path / "certs", "stress-exp", metrics)
    assert len(index["certificates"]) == 2
    assert (tmp_path / "certs" / "rag" / "rag_policy_optimization" / "nominal" / "certificate.json").exists()


def test_rag_metrics_proxy_computed() -> None:
    metrics = rag_proxy_metrics(generate_rag_proxy(20))
    assert metrics["faithfulness_proxy"] > 0


def test_patient_flow_metrics_computed() -> None:
    metrics = patient_flow_metrics(generate_patient_flow_synthetic_proxy(20))
    assert metrics["predictive_quality"] > 0


def test_elastic_compute_metrics_computed() -> None:
    metrics = elastic_compute_metrics(generate_elastic_compute_trace(20))
    assert "SLO_violation_rate" in metrics


def test_ml_to_llm_metrics_computed() -> None:
    metrics = ml_to_llm_metrics(generate_ml_to_llm_hybrid(20))
    assert metrics["predictive_performance"] > 0


def test_certificate_refuses_static_control() -> None:
    frame = pd.DataFrame(
        [
            {"system": "classical_only_baseline", "cost_adjusted_utility": 0.8, "budget_parity_ok": True},
            {"system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.5, "budget_parity_ok": True},
        ]
    )
    assert certificate_for_group("ml_to_llm", "prediction_only_baseline", frame)["status"] == "Refused"


def test_certificate_candidate_when_ablation_wins() -> None:
    frame = pd.DataFrame(
        [
            {"system": "static_default_policy", "cost_adjusted_utility": 0.2, "budget_parity_ok": True},
            {"system": "square_tune_no_fork", "cost_adjusted_utility": 0.7, "budget_parity_ok": True},
            {"system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.6, "budget_parity_ok": True},
        ]
    )
    assert certificate_for_group("rag", "rag_policy_optimization", frame)["status"] == "Candidate signal"


def test_certificate_supported_when_adaptive_compute_beats_baselines_and_ablations() -> None:
    frame = pd.DataFrame(
        [
            {"system": "static_default_policy", "cost_adjusted_utility": 0.2, "budget_parity_ok": True},
            {"system": "square_tune_no_fork", "cost_adjusted_utility": 0.4, "budget_parity_ok": True},
            {"system": "square_tune_adaptive_compute", "cost_adjusted_utility": 0.7, "budget_parity_ok": True},
        ]
    )
    assert certificate_for_group("rag", "rag_policy_optimization", frame)["status"] == "Signal supported"


def test_publication_bundle_excludes_restricted_raw_data(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    exp = "square_tune_generalized_v1_fixture_20260802-000000-deadbeef"
    report = settings.project_root / "reports" / "generalized" / exp
    cert = settings.project_root / "certificates" / "generalized" / exp
    report.mkdir(parents=True)
    cert.mkdir(parents=True)
    write_text(report / "generalized_benchmark_summary.md", "summary")
    write_json(report / "generalized_benchmark_summary.json", {"experiment_id": exp})
    pd.DataFrame([{"x": 1}]).to_parquet(report / "aggregate_metrics.parquet", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(report / "aggregate_metrics.csv", index=False)
    write_json(cert / "certificate_index.json", {"certificates": []})
    bundle = create_publication_bundle(settings, exp, tmp_path / "bundle")
    assert bundle["restricted_raw_data_excluded"]


def test_publication_bundle_contains_reproducibility_manifest(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    exp = "square_tune_generalized_v1_fixture_20260802-000001-deadbeef"
    (settings.project_root / "reports" / "generalized" / exp).mkdir(parents=True)
    (settings.project_root / "certificates" / "generalized" / exp).mkdir(parents=True)
    write_json(settings.project_root / "reports" / "generalized" / exp / "generalized_benchmark_summary.json", {})
    create_publication_bundle(settings, exp, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "reproducibility_manifest.json").exists()


def test_reports_include_caveats(tmp_path: Path, monkeypatch) -> None:
    settings, cfg = _prepare_all(tmp_path, monkeypatch)
    result = run_generalized_matrix(settings, cfg)
    report = Path(result["reports_dir"]) / "generalized_benchmark_summary.md"
    assert "does not prove SQUARE hardware" in report.read_text(encoding="utf-8")


def test_generalized_smoke_run_cpu(tmp_path: Path, monkeypatch) -> None:
    settings, cfg = _prepare_all(tmp_path, monkeypatch)
    result = run_generalized_matrix(settings, cfg)
    assert result["succeeded"] == result["total_planned"]
