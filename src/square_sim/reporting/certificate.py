from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.utils.files import read_json, write_json, write_text

CLASSICAL_MODELS = {
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost_optional",
    "lightgbm_optional",
    "mlp",
}
GATE_MODELS = {"gate_inspired_vqc_surrogate", "gate_inspired_fourier_series", "fourier_mlp"}


def _score(row: dict) -> float:
    value = row.get("roc_auc")
    return float(value) if value is not None else float(row.get("accuracy", 0.0))


def classify_certificate(
    target: str,
    metrics_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any] | None = None,
    leakage_warnings: list[str] | None = None,
    min_delta: float = 0.01,
) -> tuple[str, str]:
    leakage_warnings = leakage_warnings or []
    squares = [r for r in metrics_rows if str(r.get("model", "")).startswith("squaresim")]
    classical = [r for r in metrics_rows if r.get("model") in CLASSICAL_MODELS]
    gate = [r for r in metrics_rows if r.get("model") in GATE_MODELS]
    full = next((r for r in squares if r.get("model") == "squaresim_full"), None)
    if not full:
        return "Inconclusive", "No full SQUARESim run was available."
    best_classical = max(classical, key=_score, default=None)
    best_gate = max(gate, key=_score, default=None)
    no_feedback = next((r for r in squares if r.get("model") == "squaresim_no_feedback"), None)
    no_nonlinear = next((r for r in squares if r.get("model") == "squaresim_no_nonlinear"), None)

    if target == "target_real" and best_classical and _score(best_classical) >= _score(full):
        return "Refused", "Classical baselines dominate target_real; this negative result is useful."
    if leakage_warnings:
        return "Inconclusive", "Leakage warnings are present."
    if not best_classical or not best_gate or not no_feedback or not no_nonlinear:
        if best_classical and _score(full) > _score(best_classical) + min_delta:
            return "Candidate", "SQUARESim leads a partial comparison, but baselines or ablations are incomplete."
        return "Inconclusive", "Comparison matrix is incomplete."

    ci_ok = True
    if bootstrap:
        records = bootstrap.get("records", []) if isinstance(bootstrap, dict) else []
        classical_records = [
            r
            for r in records
            if r.get("comparison") == "best_classical" and r.get("metric") == "roc_auc"
        ]
        if classical_records:
            ci_ok = float(classical_records[0].get("ci_lower_95", -1.0)) > 0
        elif bootstrap.get("square_vs_classical"):
            ci = bootstrap["square_vs_classical"].get("ci95", [None, None])
            ci_ok = ci[0] is not None and ci[0] > 0

    beats_classical = _score(full) > _score(best_classical) + min_delta
    beats_gate = _score(full) >= _score(best_gate) - 1e-9
    beats_ablations = _score(full) > _score(no_feedback) and _score(full) > _score(no_nonlinear)
    if target in {"target", "in_pocket"} and beats_classical and beats_gate and beats_ablations and ci_ok:
        return "Simulation-supported advantage", "All conservative simulation criteria were met."
    if beats_classical:
        return "Candidate", "SQUARESim leads classical baselines, but at least one stronger criterion is unmet."
    return "Inconclusive", "Results are mixed, unstable, or underpowered."


def ontology_component_support(metrics_rows: list[dict[str, Any]]) -> dict[str, str]:
    by_model = {str(row.get("model")): row for row in metrics_rows}

    def score(model: str) -> float | None:
        return _score(by_model[model]) if model in by_model else None

    full = score("squaresim_full")
    snapshot = score("squaresim_snapshot_rollout")
    support = {
        "phase_encoding": "inconclusive",
        "emitters": "inconclusive",
        "nonlinear_dynamics": "inconclusive",
        "memory": "inconclusive",
        "feedback": "inconclusive",
        "snapshot_forking": "inconclusive",
        "nonlinear_rollout": "inconclusive",
        "merge_reintegration": "inconclusive",
    }
    if full is not None:
        comparisons = {
            "phase_encoding": score("squaresim_no_phase"),
            "nonlinear_dynamics": score("squaresim_no_nonlinear"),
            "memory": score("squaresim_no_memory"),
            "feedback": score("squaresim_no_feedback"),
            "emitters": score("squaresim_static_emitters"),
        }
        for key, ablated in comparisons.items():
            if ablated is not None:
                support[key] = "supported" if full > ablated else "not_supported"
    if snapshot is not None:
        snapshot_comparisons = {
            "snapshot_forking": score("squaresim_snapshot_no_fork"),
            "nonlinear_rollout": score("squaresim_snapshot_linear_rollout"),
            "merge_reintegration": score("squaresim_snapshot_no_merge"),
        }
        for key, ablated in snapshot_comparisons.items():
            if ablated is not None:
                support[key] = "supported" if snapshot > ablated else "not_supported"
    return support


def generate_certificate_report(
    output_path: Path,
    dataset: str,
    target: str,
    metrics_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any] | None = None,
    leakage_warnings: list[str] | None = None,
    feature_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, interpretation = classify_certificate(target, metrics_rows, bootstrap, leakage_warnings)
    best_classical = max((r for r in metrics_rows if r.get("model") in CLASSICAL_MODELS), key=_score, default={})
    best_gate = max((r for r in metrics_rows if r.get("model") in GATE_MODELS), key=_score, default={})
    best_square = max((r for r in metrics_rows if str(r.get("model", "")).startswith("squaresim")), key=_score, default={})
    payload = {
        "dataset": dataset,
        "target": target,
        "status": status,
        "best_classical": best_classical.get("model"),
        "best_gate": best_gate.get("model"),
        "best_square": best_square.get("model"),
        "efficiency_table": metrics_rows,
        "bootstrap": bootstrap or {},
        "leakage_warnings": leakage_warnings or [],
        "feature_policy": feature_policy or {},
        "ontology_component_support": ontology_component_support(metrics_rows),
        "interpretation": interpretation,
        "next_experiment": f"Run full ablation/bootstrap matrix for {dataset}/{target}.",
    }
    try:
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent / "templates"
        template = Environment(loader=FileSystemLoader(template_dir), autoescape=False).get_template(
            "certificate_report.md.j2"
        )
        text = template.render(**payload)
    except ImportError:
        rows = "\n".join(f"- {r.get('model')}: roc_auc={r.get('roc_auc')}" for r in metrics_rows)
        text = f"# SQUARE Advantage Certificate-Style Report\n\nDataset: {dataset}\nTarget: {target}\nStatus: {status}\n\n{interpretation}\n\n{rows}\n"
    write_text(output_path, text)
    write_json(output_path.with_suffix(".json"), payload)
    return payload


def metrics_from_run_dirs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            row = read_json(metrics_path)
            row.setdefault("run_id", run_dir.name)
            rows.append(row)
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
            manifests.append(manifest)
    return manifests


def generate_certificates_for_experiment(settings: Settings, experiment_id: str) -> dict[str, Any]:
    manifests = _run_manifests(settings, experiment_id)
    groups = sorted({(m.get("dataset"), m.get("target"), m.get("split_id")) for m in manifests})
    outputs: list[dict[str, Any]] = []
    for dataset, target, split_id in groups:
        group = [m for m in manifests if (m.get("dataset"), m.get("target"), m.get("split_id")) == (dataset, target, split_id)]
        metrics_rows = []
        leakage_warnings: list[str] = []
        feature_policy: dict[str, Any] = {}
        for manifest in group:
            metrics_path = manifest.get("metrics_path")
            if metrics_path and Path(metrics_path).exists():
                metrics_rows.append(read_json(Path(metrics_path)))
            leakage_warnings.extend(str(w) for w in manifest.get("leakage_warnings", []))
            feature_manifest_path = manifest.get("feature_manifest_path")
            if feature_manifest_path and Path(feature_manifest_path).exists() and not feature_policy:
                feature_policy = read_json(Path(feature_manifest_path)).get("feature_policy", {})
        leakage_warnings = list(dict.fromkeys(leakage_warnings))
        bootstrap_path = (
            settings.project_root
            / "reports"
            / "comparisons"
            / experiment_id
            / str(dataset)
            / str(target)
            / "bootstrap_comparisons.json"
        )
        bootstrap = read_json(bootstrap_path) if bootstrap_path.exists() else {}
        output_dir = settings.project_root / "reports" / "certificates" / experiment_id / str(dataset) / str(target)
        output = output_dir / "certificate.md"
        payload = generate_certificate_report(
            output,
            str(dataset),
            str(target),
            metrics_rows,
            bootstrap=bootstrap,
            leakage_warnings=leakage_warnings,
            feature_policy=feature_policy,
        )
        payload["dataset_version_id"] = group[0].get("dataset_version_id")
        payload["split_id"] = split_id
        outputs.append(payload | {"path": str(output)})
    index_dir = settings.project_root / "reports" / "certificates" / experiment_id
    write_json(index_dir / "certificate_index.json", {"experiment_id": experiment_id, "certificates": outputs})
    lines = ["# Certificate Index", "", f"Experiment ID: `{experiment_id}`", ""]
    for payload in outputs:
        lines.append(f"- {payload['dataset']} / {payload['target']}: {payload['status']}")
    write_text(index_dir / "certificate_index.md", "\n".join(lines) + "\n")
    try:
        import pandas as pd

        pd.DataFrame(outputs).to_parquet(index_dir / "certificate_index.parquet", index=False)
    except Exception:
        pass
    return {"experiment_id": experiment_id, "certificates": outputs}
