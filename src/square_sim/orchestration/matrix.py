from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.data.ensure_splits import build_ensure_requests, ensure_splits
from square_sim.data.features import FeaturePolicy, select_features
from square_sim.data.resolver import resolve_dataset_input, resolved_split_id
from square_sim.models.ablations import SQUARESIM_MODELS
from square_sim.system.environment_snapshot import environment_snapshot
from square_sim.training.train import (
    OPTIONAL_BOOSTERS,
    SKLEARN_MODELS,
    TORCH_BASELINES,
    make_run_fingerprint,
    run_single_model,
)
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import stable_hash

ALL_MODELS = set(SKLEARN_MODELS) | set(OPTIONAL_BOOSTERS) | set(TORCH_BASELINES) | set(SQUARESIM_MODELS)


@dataclass(frozen=True)
class PlannedRun:
    experiment_id: str
    dataset: str
    dataset_version_id: str | None
    target: str
    split_id: str
    model: str
    seed: int
    device: str
    status: str
    run_fingerprint: str | None
    warnings: list[str]
    command: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_matrix_config(config_path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def experiment_id_for(config_path: Path, cfg: dict[str, Any]) -> str:
    name = cfg.get("experiment_name") or config_path.stem
    return f"{name}-{stable_hash(cfg, 10)}"


def _as_list(value: Any, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def normalized_matrix_parts(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[int], str]:
    datasets = _as_list(cfg.get("datasets"), [cfg.get("dataset", "energy")])
    targets = _as_list(cfg.get("targets"), [cfg.get("target", "target")])
    models = _as_list(cfg.get("models"), [])
    split_cfg = cfg.get("split", {})
    seeds = [int(v) for v in _as_list(cfg.get("seeds"), [split_cfg.get("seed", cfg.get("seed", 42))])]
    split_id = str(split_cfg.get("split_id", cfg.get("split_id", "default")))
    return datasets, targets, models, seeds, split_id


def _completed_by_fingerprint(settings: Settings, fingerprint: str) -> dict[str, Any] | None:
    for manifest_path in (settings.project_root / "runs").glob("*/*/*/*/run_manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if (
            manifest.get("run_fingerprint") == fingerprint
            and manifest.get("status") == "succeeded"
            and Path(str(manifest.get("metrics_path"))).exists()
            and Path(str(manifest.get("predictions_path"))).exists()
        ):
            return manifest
    return None


def _device_valid(device: str) -> tuple[bool, str | None]:
    if not device.startswith("cuda"):
        return True, None
    try:
        import torch

        if torch.cuda.is_available():
            return True, None
    except Exception as exc:
        return False, f"Could not import/check torch CUDA: {exc}"
    return False, f"Requested device {device} but torch.cuda.is_available() is false."


def expand_matrix(
    settings: Settings,
    config_path: Path,
    *,
    datasets_filter: list[str] | None = None,
    targets_filter: list[str] | None = None,
    models_filter: list[str] | None = None,
    skip_completed: bool = False,
) -> tuple[str, list[PlannedRun]]:
    cfg = load_matrix_config(config_path)
    experiment_id = experiment_id_for(config_path, cfg)
    datasets, targets, models, seeds, requested_split_id = normalized_matrix_parts(cfg)
    if datasets_filter:
        datasets = [d for d in datasets if d in datasets_filter]
    if targets_filter:
        targets = [t for t in targets if t in targets_filter]
    if models_filter:
        models = [m for m in models if m in models_filter]
    resources = cfg.get("resources", {})
    device = str(resources.get("device", "cpu"))
    training = cfg.get("training", {})
    policy = FeaturePolicy.from_config(cfg.get("feature_policy"))
    env = environment_snapshot(Path.cwd())
    planned: list[PlannedRun] = []

    for dataset in datasets:
        for target in targets:
            from square_sim.training.train import _split_target_for_view

            split_target = _split_target_for_view(target)
            for seed in seeds:
                split_id, split_warning = resolved_split_id(requested_split_id, split_target, seed)
                for model in models:
                    warnings: list[str] = []
                    status = "pending"
                    version_id: str | None = None
                    fingerprint: str | None = None
                    planned_split_id = split_id
                    if model not in ALL_MODELS:
                        status = "blocked"
                        warnings.append(f"Unknown model: {model}")
                    device_ok, device_warning = _device_valid(device)
                    if not device_ok:
                        status = "blocked"
                        warnings.append(device_warning or f"Invalid device: {device}")
                    try:
                        resolved = resolve_dataset_input(settings, dataset, split_target, seed=seed, split_id=requested_split_id)
                        version_id = resolved.dataset_version_id
                        planned_split_id = resolved.split_id
                        if split_warning:
                            warnings.append(split_warning)
                        warnings.extend(resolved.leakage_warnings)
                        import pandas as pd

                        train_df = pd.read_parquet(resolved.train_path)
                        selection = select_features(
                            list(train_df.columns),
                            resolved.schema,
                            split_target,
                            policy,
                            leakage_warnings=resolved.leakage_warnings,
                        )
                        warnings.extend(selection.unacknowledged_leakage_warnings)
                        fingerprint = make_run_fingerprint(
                            {
                                "dataset_key": dataset,
                                "dataset_version_id": version_id,
                                "split_id": resolved.split_id,
                                "target": target,
                                "model": model,
                                "seed": seed,
                                "feature_policy": policy.to_dict(),
                                "model_config": {"model": model},
                                "training": {
                                    "max_epochs": int(training.get("max_epochs", 5)),
                                    "batch_size": int(training.get("batch_size", 512)),
                                    "learning_rate": float(training.get("learning_rate", 0.001)),
                                },
                                "code_git_commit": env.get("git_commit"),
                            }
                        )
                        if skip_completed and _completed_by_fingerprint(settings, fingerprint):
                            status = "completed"
                    except Exception as exc:
                        if status != "blocked":
                            status = "missing split" if "Missing split" in str(exc) else "blocked"
                        warnings.append(str(exc))
                    command = (
                        "square-sim run "
                        f"--dataset {dataset} --target {target} --model {model} "
                        f"--split-id {planned_split_id} --device {device}"
                    )
                    planned.append(
                        PlannedRun(
                            experiment_id=experiment_id,
                            dataset=dataset,
                            dataset_version_id=version_id,
                            target=target,
                            split_id=planned_split_id,
                            model=model,
                            seed=seed,
                            device=device,
                            status=status,
                            run_fingerprint=fingerprint,
                            warnings=list(dict.fromkeys(warnings)),
                            command=command,
                        )
                    )
    return experiment_id, planned


def write_plan(settings: Settings, experiment_id: str, planned: list[PlannedRun]) -> dict[str, Any]:
    output_dir = settings.project_root / "reports" / "preflight" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run.to_dict() for run in planned]
    payload = {
        "experiment_id": experiment_id,
        "planned_runs": len(rows),
        "datasets": sorted({r["dataset"] for r in rows}),
        "targets": sorted({r["target"] for r in rows}),
        "models": sorted({r["model"] for r in rows}),
        "rows": rows,
    }
    write_json(output_dir / "matrix_preview.json", payload)
    lines = ["# Matrix Preview", "", f"Experiment ID: `{experiment_id}`", f"Planned runs: {len(rows)}", ""]
    for row in rows:
        warning = f" warnings={len(row['warnings'])}" if row["warnings"] else ""
        lines.append(
            f"- {row['dataset']} / {row['target']} / {row['model']} / "
            f"{row['split_id']}: {row['status']}{warning}"
        )
    write_text(output_dir / "matrix_preview.md", "\n".join(lines) + "\n")
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "matrix_preview.csv", index=False)
        frame.to_parquet(output_dir / "matrix_preview.parquet", index=False)
    except Exception:
        pass
    return payload | {"output_dir": str(output_dir)}


def preflight(
    settings: Settings,
    config_path: Path,
    *,
    ensure_missing_splits: bool = False,
) -> dict[str, Any]:
    cfg = load_matrix_config(config_path)
    datasets, targets, _models, seeds, _split_id = normalized_matrix_parts(cfg)
    ensure_report = None
    if ensure_missing_splits:
        from square_sim.training.train import _split_target_for_view

        split_targets = list(dict.fromkeys(_split_target_for_view(target) for target in targets))
        ensure_report = ensure_splits(
            settings,
            build_ensure_requests(datasets=datasets, targets=split_targets, seed=seeds[0]),
            create=True,
            dry_run=False,
        )
    experiment_id, planned = expand_matrix(settings, config_path)
    plan_payload = write_plan(settings, experiment_id, planned)
    blocked = [run.to_dict() for run in planned if run.status in {"blocked", "missing split"}]
    output_dir = Path(plan_payload["output_dir"])
    payload = {
        "experiment_id": experiment_id,
        "status": "blocked" if blocked else "ok",
        "blocked": blocked,
        "ensure_splits": ensure_report,
        "plan": plan_payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "preflight_report.json", payload)
    lines = ["# Preflight Report", "", f"Experiment ID: `{experiment_id}`", f"Status: {payload['status']}", ""]
    if blocked:
        lines.append("## Blockers")
        for row in blocked:
            lines.append(f"- {row['dataset']} / {row['target']} / {row['model']}: {row['warnings']}")
    else:
        lines.append("No blocking issues were found.")
    write_text(output_dir / "preflight_report.md", "\n".join(lines) + "\n")
    return payload


def run_matrix(
    settings: Settings,
    config_path: Path,
    *,
    resume: bool = True,
    skip_completed: bool = True,
    only_missing: bool = False,
    retry_failed: bool = False,
    max_runs: int | None = None,
    datasets_filter: list[str] | None = None,
    targets_filter: list[str] | None = None,
    models_filter: list[str] | None = None,
    device_override: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cfg = load_matrix_config(config_path)
    if device_override:
        cfg.setdefault("resources", {})["device"] = device_override
    experiment_id = experiment_id_for(config_path, cfg)
    _eid, planned = expand_matrix(
        settings,
        config_path,
        datasets_filter=datasets_filter,
        targets_filter=targets_filter,
        models_filter=models_filter,
        skip_completed=skip_completed,
    )
    if max_runs is not None:
        planned = planned[:max_runs]
    if dry_run:
        return write_plan(settings, experiment_id, planned)

    training = cfg.get("training", {})
    resources = cfg.get("resources", {})
    run_policy = cfg.get("run_policy", {})
    fail_fast = bool(run_policy.get("fail_fast", False))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped = 0
    completed = 0
    for item in planned:
        if item.status == "completed" and skip_completed:
            skipped += 1
            results.append(item.to_dict() | {"result": "skipped_completed"})
            continue
        if item.status in {"blocked", "missing split"}:
            failures.append(item.to_dict())
            if fail_fast:
                break
            continue
        try:
            result = run_single_model(
                settings=settings,
                dataset=item.dataset,
                target=item.target,
                model_name=item.model,
                split_id=item.split_id,
                seed=item.seed,
                device=str(resources.get("device", "cpu")),
                max_epochs=int(training.get("max_epochs", 5)),
                batch_size=int(training.get("batch_size", 512)),
                learning_rate=float(training.get("learning_rate", 0.001)),
                feature_policy=cfg.get("feature_policy"),
                experiment_id=experiment_id,
                config_path=str(config_path),
                config_payload=cfg
                | {"current_dataset": item.dataset, "current_target": item.target, "current_model": item.model},
            )
            completed += 1
            results.append(item.to_dict() | {"result": "completed", "run_path": result.get("run_path")})
        except Exception as exc:
            failure = item.to_dict() | {"error": str(exc)}
            failures.append(failure)
            results.append(failure | {"result": "failed"})
            if fail_fast:
                break

    output_dir = settings.project_root / "reports" / "experiments" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": experiment_id,
        "config_path": str(config_path),
        "total_planned": len(planned),
        "completed": completed,
        "skipped": skipped,
        "failed": len(failures),
        "blocked": len([f for f in failures if f.get("status") in {"blocked", "missing split"}]),
        "results": results,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "experiment_summary.json", summary)
    write_json(output_dir / "failures.json", failures)
    lines = [
        "# Experiment Summary",
        "",
        f"Experiment ID: `{experiment_id}`",
        f"Completed: {completed}",
        f"Skipped: {skipped}",
        f"Failed/blocking: {len(failures)}",
        "",
    ]
    for failure in failures:
        lines.append(f"- {failure['dataset']} / {failure['target']} / {failure['model']}: {failure.get('error') or failure.get('warnings')}")
    write_text(output_dir / "experiment_summary.md", "\n".join(lines) + "\n")
    try:
        import pandas as pd

        pd.DataFrame(results).to_parquet(output_dir / "completed_runs.parquet", index=False)
    except Exception:
        pass
    return summary | {"output_dir": str(output_dir)}
