from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.features import FeaturePolicy, FeatureSelection, select_features
from square_sim.data.resolver import ResolvedDatasetInput, resolve_dataset_input
from square_sim.models.ablations import SQUARESIM_MODELS
from square_sim.models.baselines.boosted_trees import make_optional_booster
from square_sim.models.baselines.sklearn_baselines import (
    make_sklearn_baseline,
    predict_proba_positive,
)
from square_sim.paths import LabPaths
from square_sim.registry.models import RunRecord
from square_sim.registry.repositories import RunRepository
from square_sim.reporting.explain_run import generate_run_explanation
from square_sim.reporting.plots import write_binary_plots
from square_sim.system.environment_snapshot import environment_snapshot
from square_sim.system.gpu import gpu_info
from square_sim.system.hardware import hardware_snapshot
from square_sim.system.node_identity import get_node_identity
from square_sim.training.callbacks import EarlyStopping
from square_sim.training.evaluate import sigmoid_scores
from square_sim.training.losses import positive_class_weight
from square_sim.training.metrics import binary_metrics
from square_sim.training.resource_meter import ResourceMeter, parameter_count
from square_sim.utils.files import read_json, write_json
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.seed import set_seed

SKLEARN_MODELS = {"logistic_regression", "random_forest", "hist_gradient_boosting", "mlp"}
OPTIONAL_BOOSTERS = {"xgboost_optional", "lightgbm_optional"}
TORCH_BASELINES = {"fourier_mlp", "gate_inspired_vqc_surrogate", "gate_inspired_fourier_series"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_run_id(dataset: str, target: str, model: str, config: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{dataset[:8]}-{target[:16]}-{model[:28]}-{stable_hash(config, 8)}"


def _coerce_binary(series):
    import numpy as np

    y = series.to_numpy()
    uniques = sorted([v for v in set(y.tolist()) if not (isinstance(v, float) and np.isnan(v))])
    if len(uniques) <= 2:
        mapping = {uniques[0]: 0, uniques[-1]: 1} if len(uniques) == 2 else {uniques[0]: int(uniques[0])}
        return np.asarray([mapping.get(v, 0) for v in y], dtype=int), None
    median = float(np.nanmedian(y.astype(float)))
    return (y.astype(float) > median).astype(int), f"Target had {len(uniques)} values; binarized at median {median}."


def _frame_to_xy(
    train_df,
    val_df,
    test_df,
    target: str,
    feature_selection: FeatureSelection,
):
    import pandas as pd

    features = feature_selection.selected_features
    x_train = pd.get_dummies(train_df[features], dummy_na=True)
    x_val = pd.get_dummies(val_df[features], dummy_na=True)
    x_test = pd.get_dummies(test_df[features], dummy_na=True)
    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
    x_train = x_train.fillna(x_train.median(numeric_only=True)).fillna(0)
    x_val = x_val.fillna(x_train.median(numeric_only=True)).fillna(0)
    x_test = x_test.fillna(x_train.median(numeric_only=True)).fillna(0)
    y_train, train_warning = _coerce_binary(train_df[target])
    y_val, val_warning = _coerce_binary(val_df[target])
    y_test, test_warning = _coerce_binary(test_df[target])
    warnings = [w for w in [train_warning, val_warning, test_warning] if w]
    return x_train, x_val, x_test, y_train, y_val, y_test, warnings


def _load_split(
    settings: Settings,
    dataset: str,
    split_id: str,
    target: str,
    version: str | None = None,
    seed: int = 42,
) -> tuple[ResolvedDatasetInput, Any, Any, Any]:
    import pandas as pd

    resolved = resolve_dataset_input(settings, dataset, target, seed=seed, split_id=split_id, dataset_version_id=version)
    train_df = pd.read_parquet(resolved.train_path)
    val_df = pd.read_parquet(resolved.val_path)
    test_df = pd.read_parquet(resolved.test_path)
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' is absent from split files.")
    return resolved, train_df, val_df, test_df


def _target_view(target: str) -> dict[str, Any]:
    modes = {
        "target_all_rows": {"label": "target", "mask": None, "derived": None},
        "target_pocket_only": {"label": "target", "mask": ("in_pocket", 1), "derived": None},
        "target_non_pocket_only": {"label": "target", "mask": ("in_pocket", 0), "derived": None},
        "delta_label_all_rows": {
            "label": "__delta_label",
            "mask": None,
            "derived": {"columns": ["target", "target_real"], "op": "not_equal"},
        },
        "delta_label_pocket_only": {
            "label": "__delta_label",
            "mask": ("in_pocket", 1),
            "derived": {"columns": ["target", "target_real"], "op": "not_equal"},
        },
        "target_real_all_rows": {"label": "target_real", "mask": None, "derived": None},
        "in_pocket_detection": {"label": "in_pocket", "mask": None, "derived": None},
    }
    return modes.get(target, {"label": target, "mask": None, "derived": None})


def _split_target_for_view(target: str) -> str:
    view = _target_view(target)
    label = str(view["label"])
    if label == "__delta_label":
        return "target"
    return label


def _apply_target_view(train_df, val_df, test_df, target: str):
    view = _target_view(target)
    frames = [train_df.copy(), val_df.copy(), test_df.copy()]
    derived = view.get("derived")
    if derived:
        left, right = derived["columns"]
        for frame in frames:
            if left not in frame.columns or right not in frame.columns:
                raise ValueError(f"Derived target {target} requires columns {left} and {right}.")
            frame["__delta_label"] = (frame[left] != frame[right]).astype(int)
    mask = view.get("mask")
    original_test_rows = len(frames[2])
    if mask:
        column, value = mask
        for idx, frame in enumerate(frames):
            if column not in frame.columns:
                raise ValueError(f"Evaluation mask for {target} requires column {column}.")
            frames[idx] = frame[frame[column] == value].copy()
        if any(len(frame) == 0 for frame in frames):
            raise ValueError(f"Evaluation mask for {target} produced an empty split.")
    return (
        frames[0],
        frames[1],
        frames[2],
        str(view["label"]),
        {
            "target_name": target,
            "target_source_columns": derived["columns"] if derived else [str(view["label"])],
            "evaluation_mask": None if mask is None else f"{mask[0]} == {mask[1]}",
            "original_test_row_count": original_test_rows,
            "effective_test_row_count": len(frames[2]),
            "mask_row_count": len(frames[2]),
            "derived_target_definition": derived,
        },
    )


def _run_sklearn(model_name: str, x_train, y_train, x_test, seed: int):
    model = make_sklearn_baseline(model_name, seed)
    model.fit(x_train, y_train)
    return predict_proba_positive(model, x_test), model


def _run_optional_booster(model_name: str, x_train, y_train, x_test, seed: int):
    model = make_optional_booster(model_name, seed)
    if model is None:
        return None, None
    model.fit(x_train, y_train)
    return predict_proba_positive(model, x_test), model


def _torch_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def model_family(model_name: str) -> str:
    if model_name in SKLEARN_MODELS or model_name in OPTIONAL_BOOSTERS:
        return "classical"
    if model_name in TORCH_BASELINES:
        return "gate_inspired"
    if model_name in SQUARESIM_MODELS:
        return "squaresim"
    return "unknown"


def make_run_fingerprint(payload: dict[str, Any]) -> str:
    return stable_hash(payload, 16)


def _prediction_row_ids(test_df, target: str) -> tuple[list[str], str]:
    import pandas as pd

    if "row_id" in test_df.columns:
        return [str(v) for v in test_df["row_id"].tolist()], "existing_row_id"
    excluded = [c for c in ["target", "target_real", "in_pocket", target, "__delta_label"] if c in test_df.columns]
    source = test_df.drop(columns=excluded, errors="ignore")
    try:
        hashes = pd.util.hash_pandas_object(source.astype("string").fillna("<NA>"), index=False)
        return [f"row-{int(v):016x}" for v in hashes.tolist()], "hash_excluding_spectra_targets"
    except Exception:
        return [f"order-{i:08d}" for i in range(len(test_df))], "order_based"


def _run_torch_model(
    model_name: str,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    seed: int,
    device: str,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from square_sim.models.baselines.fourier_mlp import FourierMLP
    from square_sim.models.baselines.gate_inspired import make_gate_model
    from square_sim.models.squaresim.model import make_squaresim_model

    set_seed(seed)
    device = _torch_device(device)
    xtr = torch.tensor(x_train.to_numpy(dtype="float32"), dtype=torch.float32)
    xva = torch.tensor(x_val.to_numpy(dtype="float32"), dtype=torch.float32)
    xte = torch.tensor(x_test.to_numpy(dtype="float32"), dtype=torch.float32)
    ytr = torch.tensor(y_train.astype("float32"), dtype=torch.float32)
    yva = torch.tensor(y_val.astype("float32"), dtype=torch.float32)
    if model_name == "fourier_mlp":
        model = FourierMLP(xtr.shape[1])
    elif model_name in {"gate_inspired_vqc_surrogate", "gate_inspired_fourier_series"}:
        model = make_gate_model(model_name, xtr.shape[1])
    else:
        model = make_squaresim_model(model_name, xtr.shape[1])
        model.fit_scaler(xtr)
    model.to(device)
    pos_weight = torch.tensor([positive_class_weight(y_train)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    stopper = EarlyStopping(patience=4)
    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xva.to(device)), yva.to(device)).item()
        if stopper.step(val_loss):
            break
    model.eval()
    with torch.no_grad():
        logits = model(xte.to(device)).detach().cpu().numpy()
    return sigmoid_scores(logits), model


def _write_snapshot_diagnostics(run_dir: Path, run_id: str, model: Any) -> tuple[dict[str, Any], str | None]:
    diagnostics = getattr(model, "last_diagnostics", None)
    if not diagnostics or not diagnostics.get("snapshot_enabled"):
        return {}, None
    import pandas as pd

    output_dir = run_dir / "snapshot_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "enabled": True,
        "snapshot_count": diagnostics.get("snapshot_count", 0),
        "mean_snapshots_per_batch": diagnostics.get("snapshot_count", 0),
        "branch_entropy_mean": diagnostics.get("branch_entropy_mean"),
        "branch_weight_max_mean": diagnostics.get("branch_weight_max_mean"),
        "merge_contribution_mean": diagnostics.get("merge_contribution_mean"),
        "rollout_stability_warning_count": diagnostics.get("rollout_stability_warning_count", 0),
        "tensor_memory_estimate_mb": diagnostics.get("estimated_branch_tensor_mb", 0.0),
        "warnings": diagnostics.get("warnings", []),
    }
    write_json(output_dir / "snapshot_summary.json", summary)
    rows = []
    entropy_values = diagnostics.get("branch_entropy_values", []) or []
    weight_values = diagnostics.get("branch_weight_max_values", []) or []
    merge_values = diagnostics.get("merge_contributions", []) or []
    for idx, entropy in enumerate(entropy_values):
        rows.append(
            {
                "run_id": run_id,
                "diagnostic_sample_id": idx,
                "region_id": None,
                "branch_id": None,
                "branch_score": None,
                "branch_weight": weight_values[idx] if idx < len(weight_values) else None,
                "energy_before": None,
                "energy_after": None,
                "gradient_before": None,
                "gradient_after": None,
                "branch_entropy": entropy,
                "merge_contribution": merge_values[idx] if idx < len(merge_values) else None,
            }
        )
    pd.DataFrame(rows or [{"run_id": run_id, "diagnostic_sample_id": 0}]).to_parquet(
        output_dir / "branch_statistics.parquet",
        index=False,
    )
    write_json(output_dir / "rollout_stability.json", {"warning_count": summary["rollout_stability_warning_count"]})
    return summary, str(output_dir)


def run_single_model(
    settings: Settings,
    dataset: str,
    target: str,
    model_name: str,
    split_id: str = "default",
    seed: int = 42,
    device: str = "cpu",
    max_epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    dataset_version: str | None = None,
    config_payload: dict[str, Any] | None = None,
    feature_policy: dict[str, Any] | FeaturePolicy | None = None,
    experiment_id: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    import pandas as pd

    set_seed(seed)
    config_payload = config_payload or {
        "dataset": dataset,
        "target": target,
        "model": model_name,
        "split_id": split_id,
        "seed": seed,
    }
    split_target = _split_target_for_view(target)
    resolved, train_df, val_df, test_df = _load_split(
        settings, dataset, split_id, split_target, dataset_version, seed=seed
    )
    train_df, val_df, test_df, target_column, target_view_manifest = _apply_target_view(
        train_df, val_df, test_df, target
    )
    policy = feature_policy if isinstance(feature_policy, FeaturePolicy) else FeaturePolicy.from_config(feature_policy)
    feature_selection = select_features(
        list(train_df.columns),
        resolved.schema,
        target_column,
        policy,
        leakage_warnings=resolved.leakage_warnings,
    )
    x_train, x_val, x_test, y_train, y_val, y_test, warnings = _frame_to_xy(
        train_df, val_df, test_df, target_column, feature_selection
    )
    warnings = list(dict.fromkeys(warnings + resolved.warnings + feature_selection.leakage_warnings))
    env = environment_snapshot(Path.cwd())
    run_fingerprint = make_run_fingerprint(
        {
            "dataset_key": dataset,
            "dataset_version_id": resolved.dataset_version_id,
            "split_id": resolved.split_id,
            "target": target,
            "target_column": target_column,
            "target_view": target_view_manifest,
            "model": model_name,
            "seed": seed,
            "feature_policy": policy.to_dict(),
            "model_config": {"model": model_name},
            "training": {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "code_git_commit": env.get("git_commit"),
        }
    )
    run_id = make_run_id(dataset, target, model_name, config_payload)
    lab = LabPaths.from_settings(settings)
    run_dir = lab.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    start = now_iso()
    record = RunRecord(
        run_id=run_id,
        dataset=dataset,
        dataset_version=resolved.dataset_version_id,
        split_id=resolved.split_id,
        target=target,
        model=model_name,
        seed=seed,
        config_hash=stable_hash(config_payload),
        status="running",
        run_path=str(run_dir),
        started_at=start,
    )
    repo = RunRepository(settings.database_url)
    repo.upsert(record)
    meter = ResourceMeter(device)
    meter.start()
    skipped = False
    model: Any = None
    if model_name in SKLEARN_MODELS:
        y_score, model = _run_sklearn(model_name, x_train, y_train, x_test, seed)
    elif model_name in OPTIONAL_BOOSTERS:
        y_score, model = _run_optional_booster(model_name, x_train, y_train, x_test, seed)
        if y_score is None:
            skipped = True
            y_score = [0.5] * len(y_test)
    elif model_name in TORCH_BASELINES or model_name in SQUARESIM_MODELS:
        y_score, model = _run_torch_model(
            model_name, x_train, y_train, x_val, y_val, x_test, seed, device, max_epochs, batch_size, learning_rate
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'.")
    resources = meter.stop()
    snapshot_diagnostics, snapshot_diagnostics_path = _write_snapshot_diagnostics(run_dir, run_id, model)
    metrics = binary_metrics(y_test, y_score)
    metrics.update(
        {
            "run_id": run_id,
            "dataset": dataset,
            "dataset_version": resolved.dataset_version_id,
            "dataset_version_id": resolved.dataset_version_id,
            "split_id": resolved.split_id,
            "target": target,
            "model": model_name,
            "model_family": model_family(model_name),
            "seed": seed,
            "skipped": skipped,
            "warnings": warnings,
            "parameter_count": parameter_count(model),
            "inference_ms_per_1000_rows": None,
            "snapshot_count_mean": snapshot_diagnostics.get("snapshot_count"),
            "branch_entropy_mean": snapshot_diagnostics.get("branch_entropy_mean"),
            "branch_weight_max_mean": snapshot_diagnostics.get("branch_weight_max_mean"),
            "rollout_stability_warning_count": snapshot_diagnostics.get("rollout_stability_warning_count"),
            "estimated_branch_tensor_mb": snapshot_diagnostics.get("tensor_memory_estimate_mb"),
            "peak_gpu_memory_mb": resources.get("peak_gpu_memory_mb"),
            **resources,
        }
    )
    if resources["train_seconds"] and metrics.get("roc_auc") is not None:
        metrics["auc_per_train_minute"] = metrics["roc_auc"] / max(resources["train_seconds"] / 60.0, 1e-9)
    if metrics.get("parameter_count"):
        metrics["auc_per_million_parameters"] = metrics.get("roc_auc", 0.0) / (metrics["parameter_count"] / 1_000_000)
    write_json(run_dir / "metrics.json", metrics)
    row_ids, row_id_strategy = _prediction_row_ids(test_df, target_column)
    predictions_path = run_dir / "predictions.parquet"
    predictions = pd.DataFrame(
        {
            "row_id": row_ids,
            "y_true": y_test,
            "y_score": y_score,
            "y_pred": [int(float(score) >= 0.5) for score in y_score],
            "dataset_key": dataset,
            "dataset_version_id": resolved.dataset_version_id,
            "split_id": resolved.split_id,
            "target": target,
            "target_name": target,
            "target_column": target_column,
            "model": model_name,
            "run_id": run_id,
        }
    )
    predictions.to_parquet(predictions_path, index=False)
    prediction_manifest = {
        "row_id_strategy": row_id_strategy,
        "prediction_path": str(predictions_path),
        "count": len(predictions),
        "checksum": sha256_file(predictions_path),
        "target": target,
        "run_id": run_id,
    }
    write_json(run_dir / "prediction_manifest.json", prediction_manifest)
    feature_manifest = feature_selection.to_manifest(
        dataset_version_id=resolved.dataset_version_id,
        split_id=resolved.split_id,
        target=target,
    )
    write_json(run_dir / "feature_manifest.json", feature_manifest)
    plot_paths = write_binary_plots(y_test, y_score, run_dir / "plots")
    end_time = now_iso()
    eligibility_notes = []
    if feature_selection.unacknowledged_leakage_warnings:
        eligibility_notes.append(
            "Unacknowledged leakage warnings prevent Simulation-supported advantage under the default policy."
        )
    manifest = {
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "experiment_id": experiment_id,
        "status": "succeeded",
        "dataset_key": dataset,
        "dataset": dataset,
        "dataset_version_id": resolved.dataset_version_id,
        "dataset_version": resolved.dataset_version_id,
        "dataset_catalog_path": str(resolved.dataset_catalog_path) if resolved.dataset_catalog_path else None,
        "split_id": resolved.split_id,
        "split_manifest_path": str(resolved.split_manifest_path),
        "train_path": str(resolved.train_path),
        "val_path": str(resolved.val_path),
        "test_path": str(resolved.test_path),
        "target": target,
        "target_name": target,
        "target_column": target_column,
        "target_source_columns": target_view_manifest["target_source_columns"],
        "evaluation_mask": target_view_manifest["evaluation_mask"],
        "mask_row_count": target_view_manifest["mask_row_count"],
        "original_test_row_count": target_view_manifest["original_test_row_count"],
        "effective_test_row_count": target_view_manifest["effective_test_row_count"],
        "derived_target_definition": target_view_manifest["derived_target_definition"],
        "model_name": model_name,
        "model": model_name,
        "model_family": model_family(model_name),
        "ablation_flags": model_name if model_name.startswith("squaresim_") else None,
        "seed": seed,
        "feature_manifest_path": str(run_dir / "feature_manifest.json"),
        "config_path": config_path,
        "config_hash": stable_hash(config_payload),
        "code_git_commit": env.get("git_commit"),
        "git_commit_hash": env.get("git_commit"),
        "node_identity": get_node_identity().__dict__,
        "node_hostname": get_node_identity().hostname,
        "gpu_info": gpu_info(),
        "cuda_info": gpu_info().get("torch", {}),
        "environment_path": str(run_dir / "environment.json"),
        "hardware_path": str(run_dir / "hardware.json"),
        "dataset_manifest": str(resolved.split_manifest_path),
        "source_manifest_path": str(resolved.source_manifest_path) if resolved.source_manifest_path else None,
        "metrics_path": str(run_dir / "metrics.json"),
        "predictions_path": str(predictions_path),
        "prediction_manifest_path": str(run_dir / "prediction_manifest.json"),
        "plots": plot_paths,
        "started_at_utc": start,
        "ended_at_utc": end_time,
        "start_time": start,
        "end_time": end_time,
        "checkpoint": None,
        "leakage_warnings": feature_selection.leakage_warnings,
        "certificate_eligibility_notes": eligibility_notes,
        "warnings": warnings,
        "snapshot_diagnostics": snapshot_diagnostics,
        "snapshot_diagnostics_path": snapshot_diagnostics_path,
    }
    write_json(run_dir / "config.yaml", config_payload)
    write_json(run_dir / "environment.json", env)
    write_json(run_dir / "hardware.json", hardware_snapshot([settings.project_root]))
    write_json(run_dir / "dataset_manifest.json", read_json(resolved.split_manifest_path))
    write_json(run_dir / "run_manifest.json", manifest)
    generate_run_explanation(
        run_dir / "explanation.md",
        {
            "run_id": run_id,
            "dataset": dataset,
            "dataset_version": resolved.dataset_version_id,
            "source": str(
                settings.project_root / "datasets" / "processed" / dataset / resolved.dataset_version_id
            ),
            "target": target,
            "split_id": resolved.split_id,
            "split_folder": str(resolved.split_dir),
            "selected_features_count": len(feature_selection.selected_features),
            "excluded_columns": feature_selection.excluded_features,
            "leakage_warnings": feature_selection.leakage_warnings,
            "predictions_path": str(predictions_path),
            "part_of_matrix": bool(experiment_id),
            "model": model_name,
            "metrics": metrics,
            "resources": resources,
            "snapshot": snapshot_diagnostics,
            "snapshot_diagnostics_path": snapshot_diagnostics_path,
            "status": "Inconclusive",
        },
    )
    record.status = "succeeded"
    record.metrics_path = str(run_dir / "metrics.json")
    record.predictions_path = str(run_dir / "predictions.parquet")
    record.explanation_path = str(run_dir / "explanation.md")
    record.ended_at = manifest["end_time"]
    repo.upsert(record)
    try:
        import gc

        import torch

        del model
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return metrics | {"run_path": str(run_dir), "run_fingerprint": run_fingerprint}


def run_experiment_config(config_path: Path, settings: Settings) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    datasets = cfg.get("datasets") or [cfg["dataset"]]
    targets = cfg.get("targets") or [cfg.get("target", "target")]
    training = cfg.get("training", {})
    resources = cfg.get("resources", {})
    results = []
    for dataset in datasets:
        for target in targets:
            for model_name in cfg["models"]:
                result = run_single_model(
                    settings=settings,
                    dataset=dataset,
                    target=target,
                    model_name=model_name,
                    split_id=cfg.get("split_id", "default"),
                    seed=int(cfg.get("seed", 42)),
                    device=resources.get("device", "cpu"),
                    max_epochs=int(training.get("max_epochs", 5)),
                    batch_size=int(training.get("batch_size", 512)),
                    learning_rate=float(training.get("learning_rate", 0.001)),
                    feature_policy=cfg.get("feature_policy"),
                    experiment_id=cfg.get("experiment_id"),
                    config_path=str(config_path),
                    config_payload=cfg | {"current_dataset": dataset, "current_target": target, "current_model": model_name},
                )
                results.append(result)
    return results
