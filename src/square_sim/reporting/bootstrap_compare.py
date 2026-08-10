from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from square_sim.config import Settings
from square_sim.reporting.certificate import CLASSICAL_MODELS, GATE_MODELS
from square_sim.utils.files import read_json, write_json, write_text


def _score(row: dict[str, Any]) -> float:
    value = row.get("roc_auc")
    return float(value) if value is not None else float(row.get("accuracy", 0.0))


def _prediction_frame(path: Path):
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {"row_id", "y_true", "y_score"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Predictions are missing required columns: {missing}")
    return frame[["row_id", "y_true", "y_score"]].copy()


def align_predictions(path_a: Path, path_b: Path):
    left = _prediction_frame(path_a)
    right = _prediction_frame(path_b)
    joined = left.merge(right, on="row_id", suffixes=("_a", "_b"))
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError(
            f"Prediction row_id mismatch: left={len(left)}, right={len(right)}, aligned={len(joined)}"
        )
    if not (joined["y_true_a"].to_numpy() == joined["y_true_b"].to_numpy()).all():
        raise ValueError("Prediction y_true values do not align after row_id merge.")
    return joined


def _metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    def roc(y, s):
        return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")

    def pr(y, s):
        return float(average_precision_score(y, s))

    def ll(y, s):
        return float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1]))

    def brier(y, s):
        return float(brier_score_loss(y, s))

    return {"roc_auc": roc, "pr_auc": pr, "log_loss": ll, "brier_score": brier}


def paired_bootstrap_record(
    *,
    dataset: str,
    target: str,
    split_id: str,
    model_a: str,
    model_b: str,
    predictions_a: Path,
    predictions_b: Path,
    samples: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    joined = align_predictions(predictions_a, predictions_b)
    rng = np.random.default_rng(seed)
    y = joined["y_true_a"].to_numpy(dtype=int)
    a = joined["y_score_a"].to_numpy(dtype=float)
    b = joined["y_score_b"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for metric, func in _metric_functions().items():
        deltas: list[float] = []
        for _ in range(samples):
            idx = rng.integers(0, len(y), len(y))
            try:
                delta = func(y[idx], a[idx]) - func(y[idx], b[idx])
            except Exception:
                continue
            if np.isfinite(delta):
                deltas.append(float(delta))
        if not deltas:
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "split_id": split_id,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "status": "failed",
                    "reason": "No finite bootstrap samples.",
                    "row_count": len(y),
                    "aligned_row_count": len(joined),
                }
            )
            continue
        arr = np.asarray(deltas)
        lower, upper = np.quantile(arr, [0.025, 0.975])
        sign_probability = float((arr > 0).mean())
        status = "supports_a" if lower > 0 else "supports_b" if upper < 0 else "inconclusive"
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "split_id": split_id,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "delta_mean": float(arr.mean()),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "bootstrap_samples": samples,
                "seed": seed,
                "row_count": len(y),
                "aligned_row_count": len(joined),
                "sign_probability": sign_probability,
                "status": status,
            }
        )
    return rows


def _run_manifests(settings: Settings, experiment_id: str | None = None) -> list[dict[str, Any]]:
    manifests = []
    for path in (settings.project_root / "runs").glob("*/*/*/*/run_manifest.json"):
        try:
            manifest = read_json(path)
        except Exception:
            continue
        if experiment_id and manifest.get("experiment_id") != experiment_id:
            continue
        if manifest.get("status") == "succeeded":
            manifest["_manifest_path"] = str(path)
            manifests.append(manifest)
    return manifests


def _metric_rows(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for manifest in manifests:
        metrics_path = manifest.get("metrics_path")
        if metrics_path and Path(metrics_path).exists():
            row = read_json(Path(metrics_path))
            row["_manifest"] = manifest
            rows.append(row)
    return rows


def _best(rows: list[dict[str, Any]], models: set[str]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("model") in models]
    return max(candidates, key=_score, default=None)


def generate_bootstrap_comparisons(
    settings: Settings,
    experiment_id: str,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    manifests = _run_manifests(settings, experiment_id)
    metrics = _metric_rows(manifests)
    groups = sorted({(r.get("dataset"), r.get("target"), r.get("split_id")) for r in metrics})
    all_records: list[dict[str, Any]] = []
    outputs: list[str] = []

    for dataset, target, split_id in groups:
        rows = [r for r in metrics if (r.get("dataset"), r.get("target"), r.get("split_id")) == (dataset, target, split_id)]
        full = next((r for r in rows if r.get("model") == "squaresim_full"), None)
        if not full:
            continue
        comparisons: list[tuple[dict[str, Any], str]] = []
        best_classical = _best(rows, CLASSICAL_MODELS)
        best_gate = _best(rows, GATE_MODELS)
        if best_classical:
            comparisons.append((best_classical, "best_classical"))
        if best_gate:
            comparisons.append((best_gate, "best_gate_inspired"))
        for ablation in [
            "squaresim_no_feedback",
            "squaresim_no_nonlinear",
            "squaresim_no_memory",
            "squaresim_no_overlap_zones",
            "squaresim_static_emitters",
            "squaresim_no_phase",
        ]:
            row = next((r for r in rows if r.get("model") == ablation), None)
            if row:
                comparisons.append((row, ablation))
        full_predictions = Path(full["_manifest"]["predictions_path"])
        group_records: list[dict[str, Any]] = []
        for other, label in comparisons:
            try:
                records = paired_bootstrap_record(
                    dataset=str(dataset),
                    target=str(target),
                    split_id=str(split_id),
                    model_a="squaresim_full",
                    model_b=str(other["model"]),
                    predictions_a=full_predictions,
                    predictions_b=Path(other["_manifest"]["predictions_path"]),
                    samples=samples,
                    seed=seed,
                )
                for record in records:
                    record["comparison"] = label
                group_records.extend(records)
            except Exception as exc:
                group_records.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "split_id": split_id,
                        "model_a": "squaresim_full",
                        "model_b": other.get("model"),
                        "comparison": label,
                        "metric": "all",
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
        snapshot = next((r for r in rows if r.get("model") == "squaresim_snapshot_rollout"), None)
        if snapshot:
            snapshot_comparisons: list[tuple[dict[str, Any], str]] = [(full, "squaresim_full")]
            if best_classical:
                snapshot_comparisons.append((best_classical, "snapshot_vs_best_classical"))
            if best_gate:
                snapshot_comparisons.append((best_gate, "snapshot_vs_best_gate_inspired"))
            for ablation in [
                "squaresim_snapshot_no_fork",
                "squaresim_snapshot_linear_rollout",
                "squaresim_snapshot_no_merge",
                "squaresim_snapshot_no_feedback",
                "squaresim_snapshot_no_nonlinear",
            ]:
                row = next((r for r in rows if r.get("model") == ablation), None)
                if row:
                    snapshot_comparisons.append((row, ablation))
            snapshot_predictions = Path(snapshot["_manifest"]["predictions_path"])
            for other, label in snapshot_comparisons:
                try:
                    records = paired_bootstrap_record(
                        dataset=str(dataset),
                        target=str(target),
                        split_id=str(split_id),
                        model_a="squaresim_snapshot_rollout",
                        model_b=str(other["model"]),
                        predictions_a=snapshot_predictions,
                        predictions_b=Path(other["_manifest"]["predictions_path"]),
                        samples=samples,
                        seed=seed,
                    )
                    for record in records:
                        record["comparison"] = label
                    group_records.extend(records)
                except Exception as exc:
                    group_records.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "split_id": split_id,
                            "model_a": "squaresim_snapshot_rollout",
                            "model_b": other.get("model"),
                            "comparison": label,
                            "metric": "all",
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )
        if not group_records:
            continue
        output_dir = settings.project_root / "reports" / "comparisons" / experiment_id / str(dataset) / str(target)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "bootstrap_comparisons.json", {"records": group_records})
        lines = ["# Bootstrap Comparisons", "", f"Dataset: {dataset}", f"Target: {target}", ""]
        for record in group_records:
            lines.append(
                f"- {record.get('comparison')}: {record.get('metric')} "
                f"{record.get('model_a')} vs {record.get('model_b')} -> {record.get('status')}"
            )
        write_text(output_dir / "bootstrap_comparisons.md", "\n".join(lines) + "\n")
        try:
            import pandas as pd

            pd.DataFrame(group_records).to_parquet(output_dir / "bootstrap_comparisons.parquet", index=False)
        except Exception:
            pass
        all_records.extend(group_records)
        outputs.append(str(output_dir))
    return {"experiment_id": experiment_id, "records": all_records, "outputs": outputs}
