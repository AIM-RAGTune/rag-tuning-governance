from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from square_sim.config import Settings
from square_sim.tune.external.acquire import import_manual_dataset
from square_sim.tune.external.audit import data_audit, scenario_audit
from square_sim.tune.external.certificate import write_external_certificates
from square_sim.tune.external.diagnostics import diagnose_cost
from square_sim.tune.external.licenses import license_metadata, scan_pii_phi_texts
from square_sim.tune.external.normalize import normalize_dataset_path
from square_sim.tune.external.paths import external_root
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.tune.external.runner import run_all_external_v1, run_external_matrix
from square_sim.tune.external.scenarios import compile_scenarios
from square_sim.utils.files import read_json
from square_sim.utils.write_once import WriteOnceError, WriteOncePathManager


def _settings(tmp_path: Path) -> Settings:
    project = tmp_path / "project"
    return Settings(
        aim_nas_root=tmp_path,
        project_root=project,
        gpu_hot_scratch=tmp_path / "scratch" / "hot",
        gpu_warm_scratch=tmp_path / "scratch" / "warm",
        processing_scratch=tmp_path / "scratch" / "processing",
        database_url=f"sqlite:///{tmp_path / 'registry.sqlite3'}",
        redis_url="redis://localhost:6379/0",
        api_host="127.0.0.1",
        api_port=8087,
        local_llm_endpoint=None,
    )


def _fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt": [f"Instruction {i}" for i in range(12)],
            "question": [f"Question {i}" for i in range(12)],
            "context": [f"Context {i}" for i in range(12)],
            "response": [f"Answer {i}" for i in range(12)],
            "score": [0.2 + 0.05 * (i % 10) for i in range(12)],
            "category": ["routing", "retrieval", "preference"] * 4,
        }
    )


def _write_fixture_csv(tmp_path: Path, name: str = "fixture.csv") -> Path:
    path = tmp_path / name
    _fixture_frame().to_csv(path, index=False)
    return path


def _write_license(tmp_path: Path) -> Path:
    path = tmp_path / "license.txt"
    path.write_text("cc-by-4.0\n", encoding="utf-8")
    return path


def _make_external_dataset(tmp_path: Path, dataset_key: str, families: list[str]) -> tuple[Settings, Path]:
    s = _settings(tmp_path)
    source = _write_fixture_csv(tmp_path, f"{dataset_key}.csv")
    license_note = _write_license(tmp_path)
    import_manual_dataset(
        dataset_key,
        source,
        output_root=external_root(s),
        license_note=license_note,
        scenario_families=families,
    )
    return s, external_root(s)


def _scenario_config(tmp_path: Path, root: Path, families: list[str]) -> Path:
    config = tmp_path / "external.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "scenario_root": str(root / "scenarios"),
                "scenario_families": families,
                "seeds": [101],
                "optimizers": [
                    "random_search",
                    "greedy_regression_aware",
                    "square_tune_full",
                    "square_tune_no_snapshot",
                    "square_tune_no_merge",
                    "square_tune_no_cost_sensor",
                ],
                "square_tune": {
                    "max_rounds": 2,
                    "num_branches": 2,
                    "rollout_steps": 1,
                    "max_response_surface_evaluations": 12,
                    "max_candidate_actions": 12,
                    "simulated_gpu_hour_budget": 2.0,
                    "budget_ledger_enabled": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config


def _source_appropriate_config(tmp_path: Path, root: Path, families: list[str]) -> Path:
    path = _scenario_config(tmp_path, root, families)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["source_appropriate"] = True
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_passed_calibration(s: Settings) -> Path:
    path = (
        s.project_root
        / "certificates"
        / "square_tune"
        / "calibration"
        / "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd"
        / "certificate_index.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"experiment_id":"square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd","calibration_gates":{"global_status":"passed"}}',
        encoding="utf-8",
    )
    return path


def test_write_once_manifest_refuses_overwrite(tmp_path: Path) -> None:
    manager = WriteOncePathManager(tmp_path, [])
    manager.write_json(tmp_path / "manifest.json", {"ok": True})
    with pytest.raises(WriteOnceError):
        manager.write_json(tmp_path / "manifest.json", {"ok": False})


def test_protected_results_registry_blocks_prior_calibration_write(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    prior = s.project_root / "reports" / "square_tune" / "calibration"
    prior.mkdir(parents=True)
    registry = ProtectedResultsRegistry(s)
    registry.protect_defaults()
    with pytest.raises(WriteOnceError):
        registry.assert_not_protected(prior / "old" / "x.json")


def test_external_experiment_id_unique_when_existing_path_present(tmp_path: Path) -> None:
    manager = WriteOncePathManager(tmp_path, [])
    first = manager.create_experiment_dir("square_tune_external_v1_smoke", {"a": 1})
    second = manager.create_experiment_dir("square_tune_external_v1_smoke", {"a": 1})
    assert first[0] != second[0]
    assert first[1].exists()
    assert second[1].exists()


def test_license_unknown_blocks_unless_allowed() -> None:
    blocked = license_metadata({"key": "x", "license_required": True}, allow_unknown_license=False)
    allowed = license_metadata({"key": "x", "license_required": True}, allow_unknown_license=True)
    assert blocked["license_status"] == "missing"
    assert allowed["license_status"] == "unknown"


def test_pii_phi_scan_flags_email_phone_ssn_like_patterns() -> None:
    result = scan_pii_phi_texts(["Email user@example.com or call 555-123-4567. SSN 123-45-6789."])
    assert result["email_like_count"] == 1
    assert result["phone_like_count"] == 1
    assert result["ssn_like_count"] == 1
    assert result["warnings"]


@pytest.mark.parametrize(
    ("dataset_key", "family"),
    [
        ("ifeval", "prompt_regression_optimization"),
        ("ragbench", "rag_policy_optimization"),
        ("helpsteer2", "data_curation_preference_optimization"),
        ("dolly15k", "data_curation_preference_optimization"),
        ("bfcl", "tool_routing_policy_optimization"),
    ],
)
def test_normalize_external_fixtures(tmp_path: Path, dataset_key: str, family: str) -> None:
    source = _write_fixture_csv(tmp_path)
    manifest = normalize_dataset_path(
        source,
        tmp_path / "normalized",
        dataset_key=dataset_key,
        scenario_families=[family],
        license_status="captured",
    )
    assert manifest["row_count"] == 12
    assert Path(manifest["normalized_path"]).exists()
    frame = pd.read_parquet(manifest["normalized_path"])
    assert {"row_id", "input_text", "scenario_family", "row_checksum"}.issubset(frame.columns)


def test_manual_import_manifest_created(tmp_path: Path) -> None:
    s, root = _make_external_dataset(tmp_path, "ragbench", ["rag_policy_optimization"])
    manifests = list((root / "normalized").glob("ragbench/*/external_dataset_manifest.json"))
    assert manifests
    assert read_json(manifests[0])["dataset_key"] == "ragbench"
    assert s.project_root.exists()


@pytest.mark.parametrize(
    "family",
    [
        "rag_policy_optimization",
        "prompt_regression_optimization",
        "data_curation_preference_optimization",
        "tool_routing_policy_optimization",
    ],
)
def test_compile_external_scenarios(tmp_path: Path, family: str) -> None:
    _, root = _make_external_dataset(tmp_path, "fixture", [family])
    config = _scenario_config(tmp_path, root, [family])
    result = compile_scenarios(config)
    assert result["status"] == "completed"
    assert result["results"][0]["scenario_family"] == family
    assert list((root / "scenarios" / family).glob("*/scenario_manifest.json"))


def test_external_transfer_simulator_runs_cpu(tmp_path: Path) -> None:
    s, root = _make_external_dataset(tmp_path, "ragbench", ["rag_policy_optimization"])
    config = _scenario_config(tmp_path, root, ["rag_policy_optimization"])
    compile_scenarios(config)
    _write_passed_calibration(s)
    summary = run_external_matrix(s, config, device="cpu", smoke=True)
    assert summary["succeeded"] > 0
    assert summary["failed"] == 0
    assert (Path(summary["reports_dir"]) / "external_transfer_summary.md").exists()
    assert (Path(summary["reports_dir"]) / "no_overwrite_audit.json").exists()


def test_external_certificate_requires_calibration_reference(tmp_path: Path) -> None:
    s, root = _make_external_dataset(tmp_path, "ragbench", ["rag_policy_optimization"])
    config = _scenario_config(tmp_path, root, ["rag_policy_optimization"])
    compile_scenarios(config)
    summary = run_external_matrix(s, config, device="cpu", smoke=True)
    from square_sim.tune.external.runner import generate_external_certificate

    certs = generate_external_certificate(s, str(summary["experiment_id"]))
    statuses = {cert["status"] for cert in certs["certificates"]}
    assert "Calibration prerequisite missing" in statuses


def test_external_certificate_license_caveat(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    metrics = pd.DataFrame(
        [
            {
                "scenario_family": "rag_policy_optimization",
                "optimizer_name": "square_tune_full",
                "cost_adjusted_improvement": 0.2,
                "response_surface_evaluations": 12,
                "actual_response_surface_evaluations": 8,
                "simulated_gpu_hours": 0.2,
                "source_license_status": "unknown",
                "seed": 101,
            },
            {
                "scenario_family": "rag_policy_optimization",
                "optimizer_name": "random_search",
                "cost_adjusted_improvement": 0.1,
                "response_surface_evaluations": 12,
                "actual_response_surface_evaluations": 8,
                "simulated_gpu_hours": 0.2,
                "source_license_status": "unknown",
                "seed": 101,
            },
        ]
    )
    _write_passed_calibration(s)
    certs = write_external_certificates(s.project_root, "external_fixture", metrics)
    assert certs["certificates"][0]["status"] == "License restricted / internal only"


def test_external_certificate_budget_confounded_when_budget_fails(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _write_passed_calibration(s)
    metrics = pd.DataFrame(
        [
            {
                "scenario_family": "rag_policy_optimization",
                "optimizer_name": "square_tune_full",
                "cost_adjusted_improvement": 0.2,
                "response_surface_evaluations": 24,
                "actual_response_surface_evaluations": 16,
                "simulated_gpu_hours": 0.4,
                "source_license_status": "captured",
                "seed": 101,
            },
            {
                "scenario_family": "rag_policy_optimization",
                "optimizer_name": "random_search",
                "cost_adjusted_improvement": 0.1,
                "response_surface_evaluations": 12,
                "actual_response_surface_evaluations": 8,
                "simulated_gpu_hours": 0.2,
                "source_license_status": "captured",
                "seed": 101,
            },
        ]
    )
    certs = write_external_certificates(s.project_root, "external_budget", metrics)
    assert certs["certificates"][0]["status"] == "Budget confounded"


def test_run_all_v1_stops_if_smoke_fails_without_scenarios(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    result = run_all_external_v1(
        s,
        protect_prior=True,
        acquire=False,
        compile=False,
        run_smoke_first=True,
        run_full_if_smoke_passes=True,
        generate_reports_flag=False,
        generate_certificates=False,
        resume=True,
        skip_completed=True,
    )
    assert result["smoke_summary"]["failed"] == 1
    assert result["full_summary"] is None


def test_run_all_v1_does_not_write_into_calibration_dirs(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    prior = s.project_root / "reports" / "square_tune" / "calibration" / "old"
    prior.mkdir(parents=True)
    marker = prior / "marker.txt"
    marker.write_text("old", encoding="utf-8")
    run_all_external_v1(
        s,
        protect_prior=True,
        acquire=False,
        compile=False,
        run_smoke_first=False,
        run_full_if_smoke_passes=False,
        generate_reports_flag=False,
        generate_certificates=False,
        resume=True,
        skip_completed=True,
    )
    assert marker.read_text(encoding="utf-8") == "old"


def test_full_dataset_config_loads() -> None:
    path = Path("configs/tune/external_transfer/datasets_full_v1.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"ragbench", "ifeval", "helpsteer2", "dolly15k", "bfcl"}.issubset(cfg["datasets"])


def test_scenario_compiler_source_appropriate_rag(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    root = external_root(s)
    source = _write_fixture_csv(tmp_path, "ragbench.csv")
    license_note = _write_license(tmp_path)
    import_manual_dataset("ragbench", source, output_root=root, license_note=license_note, scenario_families=["rag_policy_optimization"])
    import_manual_dataset("ifeval", source, output_root=root, license_note=license_note, scenario_families=["prompt_regression_optimization"])
    cfg = _source_appropriate_config(tmp_path, root, ["rag_policy_optimization"])
    result = compile_scenarios(cfg, source_appropriate=True)
    scenario_path = Path(result["scenario_root"]) / "rag_policy_optimization" / result["results"][0]["scenario_id"] / "scenario.parquet"
    frame = pd.read_parquet(scenario_path)
    assert set(frame["source_dataset"]) == {"ragbench"}


def test_scenario_compiler_source_appropriate_prompt(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    root = external_root(s)
    source = _write_fixture_csv(tmp_path, "fixture.csv")
    license_note = _write_license(tmp_path)
    import_manual_dataset("ragbench", source, output_root=root, license_note=license_note, scenario_families=["rag_policy_optimization"])
    import_manual_dataset("ifeval", source, output_root=root, license_note=license_note, scenario_families=["prompt_regression_optimization"])
    cfg = _source_appropriate_config(tmp_path, root, ["prompt_regression_optimization"])
    result = compile_scenarios(cfg, source_appropriate=True)
    scenario_path = Path(result["scenario_root"]) / "prompt_regression_optimization" / result["results"][0]["scenario_id"] / "scenario.parquet"
    frame = pd.read_parquet(scenario_path)
    assert set(frame["source_dataset"]) == {"ifeval"}


def test_scenario_audit_flags_pooled_all_dataset_scenario(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    root = external_root(s)
    source = _write_fixture_csv(tmp_path, "fixture.csv")
    license_note = _write_license(tmp_path)
    import_manual_dataset("ragbench", source, output_root=root, license_note=license_note, scenario_families=["rag_policy_optimization"])
    import_manual_dataset("ifeval", source, output_root=root, license_note=license_note, scenario_families=["prompt_regression_optimization"])
    cfg = _scenario_config(tmp_path, root, ["rag_policy_optimization"])
    compile_scenarios(cfg, source_appropriate=False)
    audit = scenario_audit(root / "scenarios")
    assert any(row["warnings"] for row in audit["scenarios"])


def test_data_audit_flags_small_required_dataset(tmp_path: Path) -> None:
    _, root = _make_external_dataset(tmp_path, "ragbench", ["rag_policy_optimization"])
    audit = data_audit(root)
    row = audit["datasets"][0]
    assert row["status"] == "warning"
    assert "below suggested threshold" in " ".join(row["warnings"])


def test_ragtruth_manual_pending_record(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/tune/external_transfer/datasets_full_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["datasets"]["ragtruth"]["source_type"] == "github_or_manual"
    assert "import-manual" in cfg["datasets"]["ragtruth"]["manual_import_command"]


def test_cost_diagnostic_detects_no_cost_sensor_win(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    exp = "external_diag"
    reports = s.project_root / "reports" / "square_tune" / "external_transfer" / exp
    reports.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_full", "cost_adjusted_improvement": 0.1, "final_utility": 0.9},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_no_cost_sensor", "cost_adjusted_improvement": 0.2, "final_utility": 0.8},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_no_merge", "cost_adjusted_improvement": 0.05, "final_utility": 0.7},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "random_search", "cost_adjusted_improvement": 0.01, "final_utility": 0.6},
        ]
    )
    metrics.to_parquet(reports / "metrics.parquet", index=False)
    # Also write manifests-compatible metrics through external run loader fallback by monkeypatching with a direct file copy.
    from square_sim.tune.external import diagnostics as diag

    original = diag.load_external_metrics
    diag.load_external_metrics = lambda settings, experiment_id: metrics
    try:
        result = diagnose_cost(s, exp)
    finally:
        diag.load_external_metrics = original
    assert result["records"][0]["no_cost_sensor_beats_full"]


def test_external_certificate_requires_source_appropriate_scenario(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _write_passed_calibration(s)
    metrics = pd.DataFrame(
        [
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "square_tune_full", "cost_adjusted_improvement": 0.2, "response_surface_evaluations": 12, "actual_response_surface_evaluations": 8, "simulated_gpu_hours": 0.2, "source_license_status": "captured", "source_appropriate": False, "seed": 101},
            {"scenario_family": "rag_policy_optimization", "optimizer_name": "random_search", "cost_adjusted_improvement": 0.1, "response_surface_evaluations": 12, "actual_response_surface_evaluations": 8, "simulated_gpu_hours": 0.2, "source_license_status": "captured", "source_appropriate": False, "seed": 101},
        ]
    )
    certs = write_external_certificates(s.project_root, "external_source_bad", metrics)
    assert certs["certificates"][0]["status"] == "Inconclusive"
