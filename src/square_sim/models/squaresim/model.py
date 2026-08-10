from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from square_sim.models.squaresim.branch_scoring import BranchScorer, BranchScoringConfig
from square_sim.models.squaresim.branching import BranchGenerator, BranchingConfig
from square_sim.models.squaresim.dynamics import NonlinearFieldDynamics
from square_sim.models.squaresim.emitters import EmitterLayer
from square_sim.models.squaresim.feedback import FeedbackController
from square_sim.models.squaresim.field_grid import ScalarFieldGrid
from square_sim.models.squaresim.merge import BranchMerger, MergeConfig
from square_sim.models.squaresim.phase_encoder import PhaseEncoder
from square_sim.models.squaresim.readout import ClassifierHead
from square_sim.models.squaresim.rollout import RolloutConfig, SnapshotRolloutEngine
from square_sim.models.squaresim.snapshot_engine import SnapshotCaptureConfig, SnapshotCaptureEngine
from square_sim.models.squaresim.zones import ZoneMemory, ZoneReadout


@dataclass(frozen=True)
class SquareSimConfig:
    input_dim: int
    emitter_count: int = 32
    grid_size: int = 32
    steps: int = 4
    channels: int = 1
    encoded_dim: int = 64
    feedback_enabled: bool = True
    nonlinear_enabled: bool = True
    memory_enabled: bool = True
    overlap_zones_enabled: bool = True
    static_emitters: bool = False
    phase_enabled: bool = True
    linear_field_only: bool = False
    snapshot_enabled: bool = False
    snapshot_capture: SnapshotCaptureConfig | None = None
    branching: BranchingConfig | None = None
    rollout: RolloutConfig | None = None
    branch_scoring: BranchScoringConfig | None = None
    merge: MergeConfig | None = None
    max_estimated_snapshot_memory_mb: float = 1024.0


class SQUARESimModel(nn.Module):
    def __init__(self, config: SquareSimConfig):
        super().__init__()
        self.config = config
        self.encoder = PhaseEncoder(config.input_dim, config.encoded_dim, config.phase_enabled)
        self.emitters = EmitterLayer(config.encoded_dim, config.emitter_count, config.static_emitters)
        self.grid = ScalarFieldGrid(config.emitter_count, config.grid_size, config.channels)
        self.dynamics = NonlinearFieldDynamics(
            nonlinear_enabled=config.nonlinear_enabled,
            linear_only=config.linear_field_only,
        )
        self.readout = ZoneReadout(config.grid_size, config.channels, config.overlap_zones_enabled)
        self.memory = ZoneMemory(self.readout.output_dim)
        self.feedback = FeedbackController(config.encoded_dim, self.readout.output_dim, config.emitter_count)
        head_dim = self.readout.output_dim + config.encoded_dim
        if config.memory_enabled:
            head_dim += self.readout.output_dim
        self.head = ClassifierHead(head_dim)

    @torch.no_grad()
    def fit_scaler(self, x: torch.Tensor) -> None:
        self.encoder.fit_scaler(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        emitters = self.emitters(encoded)
        drive = self.grid(emitters)
        field = drive
        memory_state = None
        zone_readout = self.readout(field)
        for _ in range(self.config.steps):
            if self.config.feedback_enabled:
                emitters = emitters + self.feedback(encoded, zone_readout)
            drive = self.grid(emitters)
            field = self.dynamics(field, drive)
            zone_readout = self.readout(field)
            if self.config.memory_enabled:
                memory_state = self.memory(memory_state, zone_readout)
        parts = [zone_readout, encoded]
        if self.config.memory_enabled and memory_state is not None:
            parts.append(memory_state)
        return self.head(torch.cat(parts, dim=-1))


class SquareSimSnapshotRolloutClassifier(nn.Module):
    def __init__(self, config: SquareSimConfig):
        super().__init__()
        self.config = config
        self.encoder = PhaseEncoder(config.input_dim, config.encoded_dim, config.phase_enabled)
        self.emitters = EmitterLayer(config.encoded_dim, config.emitter_count, config.static_emitters)
        self.grid = ScalarFieldGrid(config.emitter_count, config.grid_size, config.channels)
        self.dynamics = NonlinearFieldDynamics(
            nonlinear_enabled=config.nonlinear_enabled,
            linear_only=config.linear_field_only,
        )
        self.readout = ZoneReadout(config.grid_size, config.channels, config.overlap_zones_enabled)
        self.memory = ZoneMemory(self.readout.output_dim)
        self.feedback = FeedbackController(config.encoded_dim, self.readout.output_dim, config.emitter_count)
        capture_cfg = config.snapshot_capture or SnapshotCaptureConfig()
        branching_cfg = config.branching or BranchingConfig()
        rollout_cfg = config.rollout or RolloutConfig()
        scoring_cfg = config.branch_scoring or BranchScoringConfig()
        merge_cfg = config.merge or MergeConfig()
        self.snapshot_engine = SnapshotCaptureEngine(
            capture_cfg,
            grid_size=config.grid_size,
            channels=config.channels,
        )
        self.branch_generator = BranchGenerator(
            branching_cfg,
            encoded_dim=config.encoded_dim,
            emitter_count=config.emitter_count,
        )
        self.rollout_engine = SnapshotRolloutEngine(rollout_cfg, channels=config.channels)
        self.branch_scorer = BranchScorer(scoring_cfg, encoded_dim=config.encoded_dim, channels=config.channels)
        self.branch_merger = BranchMerger(merge_cfg)
        self.snapshot_feature_dim = capture_cfg.num_regions * config.channels * 4
        if capture_cfg.region_selector == "whole_field":
            self.snapshot_feature_dim = config.channels * 4
        head_dim = self.readout.output_dim + config.encoded_dim + self.snapshot_feature_dim
        if config.memory_enabled:
            head_dim += self.readout.output_dim
        self.head = ClassifierHead(head_dim)
        self.last_diagnostics: dict[str, object] = {}

    @torch.no_grad()
    def fit_scaler(self, x: torch.Tensor) -> None:
        self.encoder.fit_scaler(x)

    def _empty_snapshot_features(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.snapshot_feature_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        emitters = self.emitters(encoded)
        drive = self.grid(emitters)
        field = drive
        memory_state = None
        zone_readout = self.readout(field)
        snapshot_outputs = []
        diagnostics: dict[str, object] = {
            "snapshot_enabled": True,
            "snapshot_count": 0,
            "rollout_stability_warning_count": 0,
            "estimated_branch_tensor_mb": 0.0,
            "branch_entropy_values": [],
            "branch_weight_max_values": [],
            "merge_contributions": [],
        }
        final_step = max(self.config.steps - 1, 0)
        for step in range(self.config.steps):
            if self.config.feedback_enabled:
                emitters = emitters + self.feedback(encoded, zone_readout)
            drive = self.grid(emitters)
            field = self.dynamics(field, drive)
            zone_readout = self.readout(field)
            if self.config.memory_enabled:
                memory_state = self.memory(memory_state, zone_readout)
            if self.snapshot_engine.config.should_capture(step, final_step):
                snapshot = self.snapshot_engine.capture(
                    field=field,
                    emitter_activations=emitters,
                    step_index=step,
                    zone_masks=self.readout.masks,
                    zone_readout=zone_readout,
                    memory_state=memory_state,
                    encoded_features=encoded,
                )
                branches = self.branch_generator(snapshot)
                memory_mb = branches.estimate_memory_bytes() / (1024 * 1024)
                diagnostics["estimated_branch_tensor_mb"] = float(diagnostics["estimated_branch_tensor_mb"]) + memory_mb
                diagnostics["num_branches"] = branches.diagnostics.get("num_branches")
                if memory_mb > self.config.max_estimated_snapshot_memory_mb:
                    diagnostics.setdefault("warnings", [])
                    diagnostics["warnings"].append(
                        f"Estimated snapshot branch memory {memory_mb:.1f} MB exceeds configured threshold."
                    )
                rolled = self.rollout_engine(branches)
                scores = self.branch_scorer(rolled)
                merged = self.branch_merger(
                    main_field=field,
                    rolled_branches=rolled,
                    branch_scores=scores,
                    zone_readout=zone_readout,
                    memory_state=memory_state,
                )
                if merged.field_update is not None:
                    field = merged.field_update
                    zone_readout = self.readout(field)
                snapshot_outputs.append(merged.readout_features)
                diagnostics["snapshot_count"] = int(diagnostics["snapshot_count"]) + 1
                diagnostics["rollout_stability_warning_count"] = int(
                    diagnostics["rollout_stability_warning_count"]
                ) + int(rolled.diagnostics.get("rollout_stability_warning_count", 0))
                diagnostics["branch_entropy_values"].append(scores.diagnostics["branch_entropy_mean"])
                diagnostics["branch_weight_max_values"].append(scores.diagnostics["branch_weight_max_mean"])
                diagnostics["merge_contributions"].append(merged.diagnostics["merge_contribution"])
        if snapshot_outputs:
            snapshot_features = torch.stack(snapshot_outputs, dim=0).mean(dim=0)
        else:
            snapshot_features = self._empty_snapshot_features(x.shape[0], x.device, encoded.dtype)
        parts = [zone_readout, encoded, snapshot_features]
        if self.config.memory_enabled and memory_state is not None:
            parts.append(memory_state)
        if diagnostics["branch_entropy_values"]:
            diagnostics["branch_entropy_mean"] = float(
                sum(float(v) for v in diagnostics["branch_entropy_values"]) / len(diagnostics["branch_entropy_values"])
            )
            diagnostics["branch_weight_max_mean"] = float(
                sum(float(v) for v in diagnostics["branch_weight_max_values"]) / len(diagnostics["branch_weight_max_values"])
            )
            diagnostics["merge_contribution_mean"] = float(
                sum(float(v) for v in diagnostics["merge_contributions"]) / len(diagnostics["merge_contributions"])
            )
        self.last_diagnostics = diagnostics
        return self.head(torch.cat(parts, dim=-1))


def config_for_model_name(model_name: str, input_dim: int) -> SquareSimConfig:
    flags = {
        "squaresim_full": {},
        "squaresim_no_feedback": {"feedback_enabled": False},
        "squaresim_no_nonlinear": {"nonlinear_enabled": False},
        "squaresim_no_memory": {"memory_enabled": False},
        "squaresim_no_overlap_zones": {"overlap_zones_enabled": False},
        "squaresim_static_emitters": {"static_emitters": True},
        "squaresim_no_phase": {"phase_enabled": False},
        "squaresim_linear_field_only": {
            "feedback_enabled": False,
            "nonlinear_enabled": False,
            "memory_enabled": False,
            "linear_field_only": True,
        },
        "squaresim_snapshot_rollout": {
            "snapshot_enabled": True,
        },
        "squaresim_snapshot_no_fork": {
            "snapshot_enabled": True,
            "branching": BranchingConfig(enabled=True, num_branches=1, include_identity_branch=True),
        },
        "squaresim_snapshot_linear_rollout": {
            "snapshot_enabled": True,
            "rollout": RolloutConfig(dynamics="linear_field", nonlinear_coeff=0.0),
        },
        "squaresim_snapshot_no_merge": {
            "snapshot_enabled": True,
            "merge": MergeConfig(enabled=False, strategy="no_merge"),
        },
        "squaresim_snapshot_random_branch": {
            "snapshot_enabled": True,
            "branching": BranchingConfig(random_branch=True),
            "branch_scoring": BranchScoringConfig(mode="random_score"),
            "merge": MergeConfig(strategy="random_merge"),
        },
        "squaresim_snapshot_no_feedback": {
            "snapshot_enabled": True,
            "feedback_enabled": False,
            "rollout": RolloutConfig(use_feedback=False),
        },
        "squaresim_snapshot_whole_field": {
            "snapshot_enabled": True,
            "snapshot_capture": SnapshotCaptureConfig(region_selector="whole_field", num_regions=1),
        },
        "squaresim_snapshot_local_only": {
            "snapshot_enabled": True,
            "snapshot_capture": SnapshotCaptureConfig(include_global_context=False),
        },
        "squaresim_snapshot_no_memory": {
            "snapshot_enabled": True,
            "memory_enabled": False,
            "snapshot_capture": SnapshotCaptureConfig(include_memory_state=False),
        },
        "squaresim_snapshot_no_nonlinear": {
            "snapshot_enabled": True,
            "nonlinear_enabled": False,
            "linear_field_only": True,
            "rollout": RolloutConfig(dynamics="linear_field", nonlinear_coeff=0.0),
        },
        "squaresim_snapshot_no_phase": {
            "snapshot_enabled": True,
            "phase_enabled": False,
        },
    }
    if model_name not in flags:
        raise ValueError(f"Unknown SQUARESim model '{model_name}'.")
    return SquareSimConfig(input_dim=input_dim, **flags[model_name])


def make_squaresim_model(model_name: str, input_dim: int) -> nn.Module:
    config = config_for_model_name(model_name, input_dim)
    if config.snapshot_enabled:
        return SquareSimSnapshotRolloutClassifier(config)
    return SQUARESimModel(config)
