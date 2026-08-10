from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from square_sim.config import Settings
from square_sim.data.ensure_splits import build_ensure_requests, ensure_splits
from square_sim.data.features import FeaturePolicy, select_features
from square_sim.data.resolver import default_split_id, resolve_dataset_input, resolved_split_id
from square_sim.orchestration.matrix import expand_matrix
from square_sim.reporting.bootstrap_compare import align_predictions, paired_bootstrap_record
from square_sim.reporting.certificate import classify_certificate
from square_sim.utils.files import write_json


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("AIM_PROJECT_ROOT", str(tmp_path / "SQUARE" / "source-validation-workspace"))
    return Settings.from_env(tmp_path)


def _catalog_fixture(settings: Settings, dataset: str = "energy", version: str = "energy-abc123") -> None:
    root = settings.project_root
    processed = root / "datasets" / "processed" / dataset / version
    processed.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": [f"g{i % 3}" for i in range(20)],
            "call_failure": [i % 2 for i in range(20)],
            "target": [i % 2 for i in range(20)],
            "target_real": [(i + 1) % 2 for i in range(20)],
            "in_pocket": [1 if i < 5 else 0 for i in range(20)],
        }
    )
    df.to_parquet(processed / "data.parquet", index=False)
    schema = {
        "dataset_key": dataset,
        "dataset_version_id": version,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {"name": "feature_a", "dtype": "int64", "role": "feature"},
            {"name": "feature_b", "dtype": "str", "role": "feature"},
            {"name": "call_failure", "dtype": "int64", "role": "feature"},
            {"name": "target", "dtype": "int64", "role": "target"},
            {"name": "target_real", "dtype": "int64", "role": "target_real"},
            {"name": "in_pocket", "dtype": "int64", "role": "pocket_flag"},
        ],
        "warnings": ["Potential leakage-like column name: call_failure"],
    }
    write_json(processed / "schema.json", schema)
    write_json(processed / "profile.json", {"warnings": []})
    write_json(processed / "validation_report.json", {"warnings": []})
    for target in ["target", "target_real", "in_pocket"]:
        split_id = default_split_id(target, 42)
        split = root / "datasets" / "splits" / dataset / version / split_id
        split.mkdir(parents=True, exist_ok=True)
        df.iloc[:12].to_parquet(split / "train.parquet", index=False)
        df.iloc[12:16].to_parquet(split / "val.parquet", index=False)
        df.iloc[16:].to_parquet(split / "test.parquet", index=False)
        write_json(
            split / "split_manifest.json",
            {
                "dataset_key": dataset,
                "source_dataset_version": version,
                "split_id": split_id,
                "target": target,
                "seed": 42,
                "method": "stratified",
                "class_balance": {},
            },
        )
    catalog_dir = root / "datasets" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        catalog_dir / "dataset_catalog.json",
        {
            "datasets": [
                {
                    "dataset_key": dataset,
                    "display_name": "Synthetic Energy",
                    "kaggle_slug": "example/energy",
                    "download_id": "download1",
                    "dataset_version_id": version,
                    "processed_parquet_path": str(processed / "data.parquet"),
                    "schema_path": str(processed / "schema.json"),
                    "profile_path": str(processed / "profile.json"),
                    "validation_report_path": str(processed / "validation_report.json"),
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "target_columns_present": ["target", "target_real", "in_pocket"],
                    "status": "ready",
                    "warnings": ["Potential leakage-like column name: call_failure"],
                }
            ]
        },
    )


def test_split_id_resolution_default():
    assert resolved_split_id("default", "target", 42)[0] == "default_target_target_seed_42"


def test_split_id_resolution_target_real():
    assert resolved_split_id("default", "target_real", 42)[0] == "default_target_target_real_seed_42"


def test_split_id_resolution_in_pocket():
    assert resolved_split_id("default", "in_pocket", 42)[0] == "default_target_in_pocket_seed_42"


def test_catalog_version_pinning_and_leakage_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, monkeypatch)
    _catalog_fixture(settings)
    resolved = resolve_dataset_input(settings, "energy", "target", seed=42, split_id="default")
    assert resolved.dataset_version_id == "energy-abc123"
    assert resolved.split_id == "default_target_target_seed_42"
    assert "call_failure" in json.dumps(resolved.leakage_warnings)


def test_ensure_splits_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, monkeypatch)
    _catalog_fixture(settings)
    missing = settings.project_root / "datasets" / "splits" / "energy" / "energy-abc123" / default_split_id("target_real", 99)
    assert not missing.exists()
    payload = ensure_splits(
        settings,
        build_ensure_requests(datasets=["energy"], targets=["target_real"], seed=99),
        create=True,
        dry_run=True,
    )
    assert payload["rows"][0]["status"] == "would_create"
    assert not missing.exists()


def test_feature_policy_excludes_targets():
    schema = {
        "columns": [
            {"name": "x", "role": "feature"},
            {"name": "target", "role": "target"},
            {"name": "target_real", "role": "target_real"},
            {"name": "in_pocket", "role": "pocket_flag"},
        ]
    }
    selection = select_features(["x", "target", "target_real", "in_pocket"], schema, "target", FeaturePolicy())
    assert selection.selected_features == ["x"]
    assert set(selection.excluded_features) == {"target", "target_real", "in_pocket"}


def test_certificate_downgrades_on_unacknowledged_leakage():
    rows = [
        {"model": "logistic_regression", "roc_auc": 0.70},
        {"model": "fourier_mlp", "roc_auc": 0.71},
        {"model": "squaresim_full", "roc_auc": 0.90},
        {"model": "squaresim_no_feedback", "roc_auc": 0.80},
        {"model": "squaresim_no_nonlinear", "roc_auc": 0.81},
    ]
    status, reason = classify_certificate(
        "target",
        rows,
        {"square_vs_classical": {"ci95": [0.02, 0.20]}},
        leakage_warnings=["Potential leakage-like column name: call_failure"],
    )
    assert status == "Inconclusive"
    assert "Leakage" in reason


def test_prediction_row_id_alignment(tmp_path: Path):
    pytest.importorskip("sklearn")
    a = pd.DataFrame({"row_id": ["a", "b"], "y_true": [0, 1], "y_score": [0.2, 0.8]})
    b = pd.DataFrame({"row_id": ["a", "b"], "y_true": [0, 1], "y_score": [0.1, 0.7]})
    pa = tmp_path / "a.parquet"
    pb = tmp_path / "b.parquet"
    a.to_parquet(pa, index=False)
    b.to_parquet(pb, index=False)
    assert len(align_predictions(pa, pb)) == 2
    records = paired_bootstrap_record(
        dataset="energy",
        target="target",
        split_id="s",
        model_a="squaresim_full",
        model_b="logistic_regression",
        predictions_a=pa,
        predictions_b=pb,
        samples=10,
    )
    assert {r["metric"] for r in records} >= {"roc_auc", "pr_auc"}
    b["row_id"] = ["a", "c"]
    b.to_parquet(pb, index=False)
    with pytest.raises(ValueError, match="row_id mismatch"):
        align_predictions(pa, pb)


def test_matrix_plan_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("yaml")
    settings = _settings(tmp_path, monkeypatch)
    _catalog_fixture(settings)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
experiment_name: synthetic_matrix
datasets: [energy]
targets: [target, target_real, in_pocket]
models: [logistic_regression, squaresim_full]
split:
  seed: 42
resources:
  device: cpu
feature_policy:
  exclude_roles: [target, target_real, pocket_flag]
""",
        encoding="utf-8",
    )
    _experiment_id, planned = expand_matrix(settings, cfg)
    assert len(planned) == 6
    assert {p.split_id for p in planned} == {
        "default_target_target_seed_42",
        "default_target_target_real_seed_42",
        "default_target_in_pocket_seed_42",
    }
