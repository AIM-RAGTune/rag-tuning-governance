from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from square_sim.config import Settings
from square_sim.data.synthetic_snapshot import make_snapshot_diagnostics
from square_sim.models.squaresim.branch_scoring import BranchScorer, BranchScoringConfig
from square_sim.models.squaresim.branching import BranchGenerator, BranchingConfig
from square_sim.models.squaresim.merge import BranchMerger, MergeConfig
from square_sim.models.squaresim.model import make_squaresim_model
from square_sim.models.squaresim.region_selectors import FixedZoneSelector, TopEnergySelector
from square_sim.models.squaresim.rollout import RolloutConfig, SnapshotRolloutEngine
from square_sim.models.squaresim.snapshot import SnapshotState
from square_sim.models.squaresim.snapshot_engine import SnapshotCaptureConfig, SnapshotCaptureEngine
from square_sim.reporting.certificate import ontology_component_support
from square_sim.training.train import _apply_target_view, run_single_model


def _settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("AIM_PROJECT_ROOT", str(tmp_path / "SQUARE" / "source-validation-workspace"))
    monkeypatch.setenv("SQUARESIM_DATABASE_URL", f"sqlite:///{tmp_path / 'registry.sqlite3'}")
    return Settings.from_env(tmp_path)


def _snapshot_state() -> SnapshotState:
    field = torch.randn(3, 2, 1, 8, 8)
    return SnapshotState(
        field_patch=field,
        emitter_activations=torch.randn(3, 8),
        step_index=1,
        region_descriptor={"mode": "test"},
        snapshot_id="test-snapshot",
        encoded_features=torch.randn(3, 16),
        zone_readout=torch.randn(3, 12),
        memory_state=torch.randn(3, 12),
    )


def test_snapshot_state_clone_no_mutation():
    state = _snapshot_state()
    cloned = state.clone_for_branching()
    cloned.field_patch.add_(10.0)
    assert not torch.allclose(state.field_patch, cloned.field_patch)


def test_region_selector_fixed_zone_shapes():
    selector = FixedZoneSelector(grid_size=16, num_regions=2)
    field = torch.randn(4, 1, 16, 16)
    selection = selector.select(field)
    assert selection.masks.shape == (4, 2, 16, 16)
    assert selection.region_scores.shape == (4, 2)


def test_region_selector_top_energy_selects_high_energy_region():
    selector = TopEnergySelector(patch_size=4, num_regions=1)
    field = torch.zeros(1, 1, 16, 16)
    field[:, :, 10:14, 9:13] = 5.0
    selection = selector.select(field)
    y0, x0, y1, x1 = selection.crop_coords[0, 0].tolist()
    assert y0 <= 10 <= y1
    assert x0 <= 9 <= x1


def test_snapshot_capture_contains_required_context():
    engine = SnapshotCaptureEngine(
        SnapshotCaptureConfig(region_selector="top_energy", num_regions=1, patch_size=(4, 4)),
        grid_size=16,
        channels=1,
    )
    snapshot = engine.capture(
        field=torch.randn(2, 1, 16, 16),
        emitter_activations=torch.randn(2, 8),
        zone_readout=torch.randn(2, 24),
        memory_state=torch.randn(2, 24),
        encoded_features=torch.randn(2, 16),
        step_index=1,
    )
    assert snapshot.field_patch.shape == (2, 1, 1, 4, 4)
    assert snapshot.zone_readout is not None
    assert snapshot.memory_state is not None
    assert snapshot.local_energy is not None


def test_branch_generator_adds_branch_dimension_and_identity():
    state = _snapshot_state()
    generator = BranchGenerator(BranchingConfig(num_branches=4), encoded_dim=16, emitter_count=8)
    branched = generator(state)
    assert branched.field_patch.shape == (3, 2, 4, 1, 8, 8)
    assert torch.allclose(branched.field_patch[:, :, 0], state.field_patch)


def test_rollout_linear_vs_nonlinear_different_and_no_nan():
    state = BranchGenerator(BranchingConfig(num_branches=2), encoded_dim=16, emitter_count=8)(_snapshot_state())
    linear = SnapshotRolloutEngine(RolloutConfig(dynamics="linear_field", steps=2), channels=1)(state)
    nonlinear = SnapshotRolloutEngine(RolloutConfig(dynamics="nonlinear_field", steps=2), channels=1)(state)
    assert torch.isfinite(nonlinear.field_patch).all()
    assert not torch.allclose(linear.field_patch, nonlinear.field_patch)


def test_branch_scorer_outputs_weights_sum_to_one():
    state = BranchGenerator(BranchingConfig(num_branches=3), encoded_dim=16, emitter_count=8)(_snapshot_state())
    scores = BranchScorer(BranchScoringConfig(), encoded_dim=16, channels=1)(state)
    assert scores.weights.shape == (3, 2, 3)
    assert torch.allclose(scores.weights.sum(dim=-1), torch.ones(3, 2), atol=1e-5)


def test_branch_merger_weighted_readout():
    state = BranchGenerator(BranchingConfig(num_branches=2), encoded_dim=16, emitter_count=8)(_snapshot_state())
    scores = BranchScorer(BranchScoringConfig(), encoded_dim=16, channels=1)(state)
    result = BranchMerger(MergeConfig())(
        main_field=torch.randn(3, 1, 8, 8),
        rolled_branches=state,
        branch_scores=scores,
    )
    assert result.readout_features.shape == (3, 8)


def test_snapshot_model_forward_cpu():
    model = make_squaresim_model("squaresim_snapshot_rollout", input_dim=5)
    x = torch.randn(4, 5)
    model.fit_scaler(x)
    out = model(x)
    assert out.shape == (4,)
    assert model.last_diagnostics["snapshot_count"] >= 1


def test_snapshot_ablation_flags_change_behavior():
    x = torch.randn(4, 5)
    no_fork = make_squaresim_model("squaresim_snapshot_no_fork", input_dim=5)
    no_fork.fit_scaler(x)
    no_fork(x)
    no_merge = make_squaresim_model("squaresim_snapshot_no_merge", input_dim=5)
    no_merge.fit_scaler(x)
    no_merge(x)
    assert no_fork.last_diagnostics["num_branches"] == 1
    assert no_merge.last_diagnostics["merge_contribution_mean"] == 0.0


def test_synthetic_snapshot_dataset_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, monkeypatch)
    payload = make_snapshot_diagnostics(settings, tmp_path / "synthetic", rows=200, seed=42)
    assert len(payload["created"]) == 6
    assert Path(payload["catalog"]).exists()


def test_snapshot_diagnostics_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path, monkeypatch)
    make_snapshot_diagnostics(settings, tmp_path / "synthetic", rows=220, seed=42)
    result = run_single_model(
        settings,
        "synthetic_snapshot_field_overlap_future",
        "target",
        "squaresim_snapshot_rollout",
        device="cpu",
        max_epochs=1,
        batch_size=64,
    )
    run_dir = Path(result["run_path"])
    assert (run_dir / "snapshot_diagnostics" / "snapshot_summary.json").exists()
    assert (run_dir / "snapshot_diagnostics" / "branch_statistics.parquet").exists()


def test_certificate_snapshot_component_support():
    support = ontology_component_support(
        [
            {"model": "squaresim_snapshot_rollout", "roc_auc": 0.90},
            {"model": "squaresim_snapshot_no_fork", "roc_auc": 0.80},
            {"model": "squaresim_snapshot_linear_rollout", "roc_auc": 0.85},
            {"model": "squaresim_snapshot_no_merge", "roc_auc": 0.70},
        ]
    )
    assert support["snapshot_forking"] == "supported"
    assert support["nonlinear_rollout"] == "supported"
    assert support["merge_reintegration"] == "supported"


def test_evaluation_mask_and_delta_label_generation():
    train = pd.DataFrame({"target": [0, 1, 1], "target_real": [0, 0, 1], "in_pocket": [1, 0, 1]})
    val = train.copy()
    test = train.copy()
    _tr, _va, masked, label, manifest = _apply_target_view(train, val, test, "target_pocket_only")
    assert label == "target"
    assert manifest["evaluation_mask"] == "in_pocket == 1"
    assert len(masked) == 2
    tr, _va, _te, label, manifest = _apply_target_view(train, val, test, "delta_label_all_rows")
    assert label == "__delta_label"
    assert tr["__delta_label"].tolist() == [0, 1, 0]


def test_snapshot_cuda_unavailable_message():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")
    assert not torch.cuda.is_available()
