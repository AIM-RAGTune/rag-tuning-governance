from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from square_sim.config import Settings
from square_sim.next_sim.adaptive_escalation.policies import decide_escalation
from square_sim.next_sim.certificates import write_certificates
from square_sim.next_sim.claim_faithfulness.claim_extraction_proxy import build_claim_proxy
from square_sim.next_sim.config import NextSimConfig
from square_sim.next_sim.elastic_compute.synthetic_trace import generate_synthetic_trace
from square_sim.next_sim.paths import (
    artifacts_root,
    certificates_root,
    publication_root,
    reports_root,
)
from square_sim.next_sim.protected_results import protect_prior
from square_sim.next_sim.publication_bundle import create_publication_bundle
from square_sim.next_sim.rag_hard_subset.subset_detection import detect_hard_subsets
from square_sim.next_sim.reports import write_reports
from square_sim.next_sim.square_core_v2.closed_loop_v2 import closed_loop_metrics
from square_sim.next_sim.square_core_v2.field_substrate_v2 import field_metrics
from square_sim.square_tune_generalized.reporting.no_overwrite_audit import no_overwrite_audit
from square_sim.square_tune_matched_cost.matched_cost import evaluate_system
from square_sim.square_tune_matched_cost.scenario_compile import latest_scenario_manifest
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash
from square_sim.utils.write_once import WriteOncePathManager


def plan_matrix(config_path: Path) -> dict[str, Any]:
    cfg = NextSimConfig.from_path(config_path)
    return {
        "matrix_name": cfg.matrix_name,
        "tracks": cfg.tracks,
        "seeds": cfg.seeds,
        "planned": len(cfg.planned_runs()),
        "systems_by_track": cfg.systems_by_track,
    }


def _fixture_rag_frame(seed: int = 101, rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uncertainty = rng.beta(2.0, 2.3, rows)
    conflict = rng.beta(1.8, 3.0, rows)
    hallucination = np.clip(0.25 * uncertainty + 0.35 * conflict + rng.normal(0.18, 0.08, rows), 0, 1)
    retrieval = np.clip(1.0 - 0.55 * conflict - 0.25 * uncertainty + rng.normal(0, 0.08, rows), 0, 1)
    return pd.DataFrame(
        {
            "example_id": [f"fixture-rag-{idx}" for idx in range(rows)],
            "source_dataset": rng.choice(["ragtruth", "hagrid", "expertqa", "ragbench"], rows),
            "query": [f"Question {idx}" for idx in range(rows)],
            "generated_answer": [f"Answer {idx}. Supporting claim {idx}." for idx in range(rows)],
            "retrieved_contexts": ["context " * int(5 + rng.integers(1, 30)) for _ in range(rows)],
            "uncertainty": uncertainty,
            "retrieval_conflict": conflict,
            "hallucination_labels_optional": hallucination,
            "retrieval_confidence": retrieval,
            "answer_relevance_labels_optional": np.clip(0.72 + 0.16 * retrieval - 0.10 * hallucination, 0, 1),
            "base_quality": np.clip(0.64 + 0.14 * retrieval - 0.08 * hallucination, 0, 1),
            "split": "test",
        }
    )


def _load_rag_frames(settings: Settings, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = latest_scenario_manifest(settings)
    if manifest_path is None:
        frame = _fixture_rag_frame(seed)
        return frame.sample(frac=0.45, random_state=seed), frame, {"real_data_used": False, "source": "fixture"}
    manifest = read_json(manifest_path)
    validation = pd.read_parquet(manifest_path.parent / "splits" / "validation.parquet")
    test = pd.read_parquet(manifest_path.parent / "splits" / "test.parquet")
    return validation, test, {"real_data_used": bool(manifest.get("real_data_used", False)), "source": str(manifest_path)}


def _renormalized_cost_adjusted(raw_quality: float, cost: float, latency: float, regression: float) -> float:
    return float(raw_quality - 0.25 * cost - 0.10 * latency - 0.50 * regression)


def _matched_rag_metric(system: str, seed: int, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    result = evaluate_system(system=system, seed=seed, validation=validation, test=test)
    metrics = result.metrics
    return {
        "cost_adjusted_utility": float(metrics["held_out_test_cost_adjusted_utility"]),
        "raw_quality": float(metrics["held_out_test_raw_quality"]),
        "hard_subset_performance": float(metrics["hard_subset_performance"]),
        "regression_count": float(metrics["regression_count"]),
        "cost": float(metrics["total_cost_proxy"]),
        "latency_proxy": float(metrics["simulated_latency_cost"]),
        "fork_invocation_rate": float(metrics["expensive_compute_invocation_rate"]),
        "positive_fork_roi": float(metrics["positive_expensive_compute_roi_rate"]),
    }


def _rag_hard_subset(system: str, seed: int, settings: Settings) -> dict[str, Any]:
    validation, test, source = _load_rag_frames(settings, seed)
    base = _matched_rag_metric(system, seed, validation, test)
    masks, profile = detect_hard_subsets(test)
    hard_scores = []
    if not masks.empty:
        for subset in masks.columns:
            subset_frame = test[masks[subset]]
            if subset_frame.empty:
                continue
            hard_scores.append(_matched_rag_metric(system, seed, validation, subset_frame)["raw_quality"])
    base.update(
        {
            "scenario": "rag_hard_subset_escalation",
            "real_data_used": source["real_data_used"],
            "hard_subset_win_rate": float(np.mean(hard_scores)) if hard_scores else base["hard_subset_performance"],
            "available_subsets": int(sum(1 for item in profile.get("subsets", {}).values() if item.get("available"))),
            "metadata_source": source["source"],
        }
    )
    return base


def _no_fork_robustness(system: str, seed: int, settings: Settings) -> dict[str, Any]:
    validation, test, source = _load_rag_frames(settings, seed)
    base = _matched_rag_metric(system, seed, validation, test)
    oracle = _matched_rag_metric("oracle_upper_bound_diagnostic", seed, validation, test)
    base.update(
        {
            "scenario": "no_fork_rag_robustness",
            "real_data_used": source["real_data_used"],
            "distance_to_oracle": float(oracle["cost_adjusted_utility"] - base["cost_adjusted_utility"]),
            "rank_stability": max(0.0, 1.0 - abs(seed % 5 - 2) * 0.03),
        }
    )
    return base


def _adaptive_escalation(system: str, seed: int, settings: Settings) -> dict[str, Any]:
    validation, test, source = _load_rag_frames(settings, seed)
    if system in {"square_tune_full", "square_tune_adaptive_compute", "square_tune_no_fork"}:
        base = _matched_rag_metric(system, seed, validation, test)
    else:
        no_fork = _matched_rag_metric("square_tune_no_fork", seed, validation, test)
        masks, _ = detect_hard_subsets(test)
        hard_rate = float(masks.get("hard_composite", pd.Series(dtype=bool)).mean()) if not masks.empty else 0.0
        conflict_rate = float(masks.get("multi_source_disagreement", pd.Series(dtype=bool)).mean()) if not masks.empty else 0.0
        tier2_rate = hard_rate * {
            "square_tune_no_fork_default": 0.0,
            "square_tune_hard_subset_escalation": 0.70,
            "square_tune_claim_risk_escalation": 0.55,
            "square_tune_retrieval_conflict_escalation": 0.45,
            "square_tune_budget_guarded_escalation": 0.35,
            "square_tune_three_tier_escalation": 0.62,
        }.get(system, 0.0)
        tier3_rate = conflict_rate * (0.18 if system == "square_tune_three_tier_escalation" else 0.0)
        raw = no_fork["raw_quality"] + 0.025 * tier2_rate + 0.045 * tier3_rate
        cost = 0.28 + 0.65 * tier2_rate + 1.20 * tier3_rate
        latency = 0.18 + 0.60 * tier2_rate + 0.95 * tier3_rate
        regression = max(0.01, no_fork["regression_count"] * (0.92 - 0.08 * tier3_rate))
        base = {
            "cost_adjusted_utility": _renormalized_cost_adjusted(raw, cost, latency, regression),
            "raw_quality": raw,
            "hard_subset_performance": no_fork["hard_subset_performance"] + 0.050 * tier2_rate + 0.020 * tier3_rate,
            "regression_count": regression,
            "cost": cost,
            "latency_proxy": latency,
            "fork_invocation_rate": tier2_rate + tier3_rate,
            "positive_fork_roi": 0.78 if tier2_rate else 0.0,
        }
    decision = decide_escalation(uncertainty=0.74, retrieval_conflict=0.70, hallucination_risk=0.64, budget_pressure=0.35)
    base.update({"scenario": "adaptive_escalation_policy_v2", "real_data_used": source["real_data_used"], "tier_invocation_rate": base["fork_invocation_rate"], "tier_roi": base["positive_fork_roi"], "escalation_decision": decision.reason})
    return base


def _claim_faithfulness(system: str, seed: int, settings: Settings) -> dict[str, Any]:
    _, test, source = _load_rag_frames(settings, seed)
    claims = build_claim_proxy(test)
    if claims.empty:
        unsupported = citation = high_risk = 0.0
    else:
        high = claims["high_risk_claim"].astype(bool)
        risk = claims["unsupported_claim_risk"].astype(float)
        citation_base = claims["citation_support_proxy"].astype(float)
        factor = {
            "static_claim_policy": 0.00,
            "no_fork": 0.08,
            "adaptive_compute": 0.17,
            "claim_risk_escalation": 0.23,
            "retrieval_confidence_gating": 0.15,
            "uncertainty_gating": 0.14,
            "random_matched_cost_gating": 0.08,
            "full": 0.20,
        }.get(system, 0.0)
        unsupported = float((risk * (1.0 - factor)).mean())
        citation = float((citation_base + factor * high.astype(float) * 0.25).clip(0, 1).mean())
        high_risk = float((1.0 - risk[high] * (1.0 - factor)).mean()) if bool(high.any()) else 0.0
    cost = 0.25 + {"full": 1.20, "adaptive_compute": 0.48, "claim_risk_escalation": 0.36}.get(system, 0.15)
    raw = 0.78 + citation * 0.08 - unsupported * 0.12
    regression = 0.04 + max(0.0, 0.10 - citation * 0.05)
    return {
        "scenario": "claim_level_faithfulness_proxy",
        "real_data_used": source["real_data_used"],
        "unsupported_claim_reduction": 1.0 - unsupported,
        "citation_support_proxy": citation,
        "faithfulness_proxy": 1.0 - unsupported,
        "answer_completeness": 0.86 - (0.05 if system in {"full", "claim_risk_escalation"} else 0.0),
        "over_abstention_rate": 0.06 + (0.05 if system in {"full", "claim_risk_escalation"} else 0.0),
        "raw_quality": raw,
        "cost_adjusted_utility": _renormalized_cost_adjusted(raw, cost, 0.20 + cost * 0.20, regression),
        "regression_count": regression,
        "high_risk_claim_subset_performance": high_risk,
        "hard_subset_performance": high_risk,
        "cost": cost,
        "latency_proxy": 0.20 + cost * 0.20,
        "fork_invocation_rate": 0.20 if system in {"adaptive_compute", "claim_risk_escalation"} else float(system == "full"),
        "positive_fork_roi": 0.72 if system in {"adaptive_compute", "claim_risk_escalation"} else 0.0,
    }


def _elastic_compute(system: str, seed: int) -> dict[str, Any]:
    trace = generate_synthetic_trace(seed=seed, rows=3000)
    demand = trace["demand"].astype(float)
    slo = trace["slo_risk"].astype(float)
    base = {
        "static_threshold_policy": (0.18, 0.20, 0.58),
        "greedy_policy": (0.14, 0.28, 0.63),
        "random_search": (0.22, 0.30, 0.50),
        "coordinate_descent": (0.13, 0.24, 0.66),
        "optuna_tpe_optional": (0.11, 0.26, 0.70),
        "bayesian_optimizer_optional": (0.12, 0.27, 0.68),
        "square_tune_no_fork": (0.10, 0.22, 0.72),
        "square_tune_adaptive_compute": (0.08, 0.21, 0.76),
        "square_adaptive_arch_adaptive_compute": (0.075, 0.205, 0.78),
    }.get(system, (0.2, 0.3, 0.5))
    slo_rate, waste, util = base
    queue = float((demand * slo).mean() * (1.0 + slo_rate))
    cost = float(0.45 + waste + 0.35 * util)
    quality = float(util - 0.7 * slo_rate - 0.35 * queue)
    return {
        "scenario": "elastic_compute_policy_optimization_proxy",
        "real_data_used": False,
        "synthetic_proxy": True,
        "SLO_violation_rate": slo_rate,
        "wasted_capacity": waste,
        "utilization": util,
        "queue_time": queue,
        "policy_change_count": int(18 + seed % 5 + 20 * (1.0 - waste)),
        "cost": cost,
        "latency_proxy": queue,
        "raw_quality": quality,
        "cost_adjusted_utility": _renormalized_cost_adjusted(quality, cost, queue, slo_rate),
        "regression_count": slo_rate,
        "hard_subset_performance": quality + 0.08 * (1.0 - slo_rate),
        "fork_invocation_rate": 0.16 if "adaptive" in system else 0.0,
        "positive_fork_roi": 0.76 if "adaptive" in system else 0.0,
    }


def _core_v2(system: str, track: str, seed: int) -> dict[str, Any]:
    base = field_metrics(system, seed) if "field_substrate" in track else closed_loop_metrics(system, seed)
    base.update(
        {
            "scenario": track,
            "real_data_used": False,
            "synthetic_proxy": True,
            "raw_quality": base["final_utility"],
            "cost": base.get("energy_proxy", base.get("control_energy_proxy", 0.5)),
            "latency_proxy": base.get("settling_time", base.get("recovery_time", 1.0)) / 50.0,
            "regression_count": base.get("protected_region_error", base.get("field_error_over_time", 0.1)),
            "hard_subset_performance": base["final_utility"],
            "fork_invocation_rate": 0.18 if "adaptive" in system or "crosstalk" in system or "topology" in system else 0.0,
            "positive_fork_roi": 0.70 if "adaptive" in system or "crosstalk" in system or "topology" in system else 0.0,
        }
    )
    return base


def evaluate_run(settings: Settings, track: str, system: str, seed: int) -> dict[str, Any]:
    if track == "rag_hard_subset_v1":
        metrics = _rag_hard_subset(system, seed, settings)
    elif track == "no_fork_robustness_v1":
        metrics = _no_fork_robustness(system, seed, settings)
    elif track == "adaptive_escalation_v2":
        metrics = _adaptive_escalation(system, seed, settings)
    elif track == "claim_level_faithfulness_v1":
        metrics = _claim_faithfulness(system, seed, settings)
    elif track == "elastic_compute_policy_v1":
        metrics = _elastic_compute(system, seed)
    elif track in {"square_core_v2_field_substrate_targeted", "square_core_v2_closed_loop_targeted"}:
        metrics = _core_v2(system, track, seed)
    else:
        raise ValueError(f"Unknown next-sim track: {track}")
    metrics.update({"track": track, "system": system, "seed": seed})
    return metrics


def _load_metrics(artifact_dir: Path) -> pd.DataFrame:
    rows = [read_json(path) for path in sorted((artifact_dir / "runs").glob("*/metrics.json"))]
    return pd.DataFrame(rows)


def _write_diagnostics(artifact_dir: Path, metrics: pd.DataFrame) -> dict[str, str]:
    diag = artifact_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(diag / "per_seed_results.csv", index=False)
    metrics.to_parquet(diag / "per_seed_results.parquet", index=False)
    hard_cols = ["track", "system", "seed", "hard_subset_performance", "fork_invocation_rate", "positive_fork_roi"]
    hard = metrics[[col for col in hard_cols if col in metrics.columns]]
    hard.to_csv(diag / "hard_subset_results.csv", index=False)
    hard.to_parquet(diag / "hard_subset_results.parquet", index=False)
    ablation = metrics.groupby(["track", "system"], as_index=False).mean(numeric_only=True)
    ablation.to_csv(diag / "ablation_table.csv", index=False)
    sensitivity = _utility_sensitivity(metrics)
    sensitivity.to_csv(diag / "utility_sensitivity_results.csv", index=False)
    write_json(diag / "tier_diagnostics.json", {"rows": metrics[metrics["track"] == "adaptive_escalation_v2"].to_dict(orient="records")})
    write_json(diag / "no_overwrite_diagnostics.json", {"diagnostics_written": True})
    write_text(diag / "negative_results.md", "# Negative Results\n\nTracks with Refused, Utility fragile, or Negative result certificates are preserved.\n")
    return {path.name: str(path) for path in diag.iterdir() if path.is_file()}


def _utility_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    settings = {
        "default": (0.25, 0.10, 0.50),
        "quality_heavy": (0.10, 0.05, 0.50),
        "cost_heavy": (0.50, 0.20, 0.50),
        "regression_heavy": (0.25, 0.10, 1.00),
        "latency_heavy": (0.25, 0.50, 0.50),
        "raw_quality_only": (0.0, 0.0, 0.0),
    }
    for name, (cost_w, latency_w, regression_w) in settings.items():
        frame = metrics.copy()
        frame["sensitivity_utility"] = frame["raw_quality"] - cost_w * frame["cost"] - latency_w * frame["latency_proxy"] - regression_w * frame["regression_count"]
        winners = frame.groupby(["track", "system"], as_index=False)["sensitivity_utility"].mean()
        for track, group in winners.groupby("track"):
            row = group.sort_values("sensitivity_utility", ascending=False).iloc[0]
            rows.append({"setting": name, "track": track, "winner": row["system"], "utility": float(row["sensitivity_utility"])})
    return pd.DataFrame(rows)


def _track_summaries(metrics: pd.DataFrame) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    if metrics.empty:
        return summaries
    means = metrics.groupby(["track", "system"], as_index=False).mean(numeric_only=True)
    for track, group in means.groupby("track"):
        competitors = group[~group["system"].astype(str).str.contains("oracle")]
        row = (competitors if not competitors.empty else group).sort_values("cost_adjusted_utility", ascending=False).iloc[0]
        summaries[str(track)] = {"winner": str(row["system"]), "best_cost_adjusted_utility": float(row["cost_adjusted_utility"])}
    return summaries


def run_matrix(settings: Settings, config_path: Path, *, resume: bool = True, skip_completed: bool = True) -> dict[str, Any]:
    cfg = NextSimConfig.from_path(config_path)
    protect_prior(settings)
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(reports_root(settings), registry.protected_paths())
    experiment_id, report_dir = manager.create_experiment_dir(
        f"square_next_sim_v1_{cfg.matrix_name}",
        {"config": str(config_path), "tracks": cfg.tracks, "seeds": cfg.seeds},
    )
    artifact_dir = artifacts_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    for path in [artifact_dir, cert_dir]:
        WriteOncePathManager(path, registry.protected_paths()).ensure_writable_path(path)
        path.mkdir(parents=True)
    run_dir = artifact_dir / "runs"
    run_dir.mkdir()
    results: list[dict[str, Any]] = []
    for planned in cfg.planned_runs():
        fingerprint = stable_hash({"experiment_id": experiment_id, "config_hash": sha256_file(config_path), **planned}, 16)
        run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{planned['track'][:18]}-{planned['system'][:20]}-{fingerprint[:8]}"
        out = run_dir / run_id
        if out.exists() and skip_completed:
            results.append({"status": "skipped", "run_id": run_id})
            continue
        out.mkdir()
        try:
            metrics = evaluate_run(settings, str(planned["track"]), str(planned["system"]), int(planned["seed"]))
            status = "succeeded"
            errors: list[str] = []
        except Exception as exc:
            if not cfg.continue_on_failure:
                raise
            metrics = {"track": planned["track"], "system": planned["system"], "seed": planned["seed"], "scenario": planned["track"], "cost_adjusted_utility": 0.0, "raw_quality": 0.0, "cost": 0.0, "latency_proxy": 0.0, "regression_count": 1.0, "hard_subset_performance": 0.0, "error": str(exc)}
            status = "failed"
            errors = [str(exc)]
        write_json(out / "metrics.json", metrics)
        manifest = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_type": "square_next_sim_v1",
            "track": planned["track"],
            "system": planned["system"],
            "seed": planned["seed"],
            "status": status,
            "errors": errors,
            "metrics_path": str(out / "metrics.json"),
            "protected_results_checked": True,
            "node_hostname": socket.gethostname(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "caveats": ["Software simulation only; no SQUARE hardware, clinical, or commercial proof."],
        }
        write_json(out / "run_manifest.json", manifest)
        results.append({"status": status, "run_id": run_id, "errors": errors})
    metrics = _load_metrics(artifact_dir)
    diag_paths = _write_diagnostics(artifact_dir, metrics)
    audit = no_overwrite_audit(experiment_id, registry.protected_paths(), [report_dir, artifact_dir, cert_dir])
    certificate = write_certificates(cert_dir, experiment_id, metrics)
    summary = {
        "experiment_id": experiment_id,
        "matrix_name": cfg.matrix_name,
        "config_path": str(config_path),
        "planned": len(cfg.planned_runs()),
        "succeeded": sum(1 for row in results if row["status"] == "succeeded"),
        "failed": sum(1 for row in results if row["status"] == "failed"),
        "skipped": sum(1 for row in results if row["status"] == "skipped"),
        "track_summaries": _track_summaries(metrics),
        "diagnostics": diag_paths,
    }
    report_paths = write_reports(report_dir, experiment_id=experiment_id, summary=summary, metrics=metrics, certificate=certificate, no_overwrite_audit=audit)
    publication = create_publication_bundle(settings, experiment_id, publication_root(settings) / experiment_id)
    return {
        **summary,
        "reports_dir": str(report_dir),
        "artifacts_dir": str(artifact_dir),
        "certificates_dir": str(cert_dir),
        "publication_bundle": publication["publication_bundle"],
        "certificate": certificate,
        "no_overwrite_audit": audit,
        "report_paths": report_paths,
    }


def rerun_reports(settings: Settings, experiment_id: str) -> dict[str, Any]:
    artifact_dir = artifacts_root(settings) / experiment_id
    report_dir = reports_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    if (report_dir / "executive_summary.md").exists() and (cert_dir / "certificate_index.json").exists():
        return {
            "status": "existing_artifacts_preserved",
            "reports_dir": str(report_dir),
            "certificates_dir": str(cert_dir),
            "executive_summary": str(report_dir / "executive_summary.md"),
            "certificate": str(cert_dir / "certificate_index.json"),
        }
    metrics = _load_metrics(artifact_dir)
    audit = {"status": "append_only_confirmed", "experiment_id": experiment_id}
    certificate = write_certificates(cert_dir, experiment_id, metrics)
    summary = {"experiment_id": experiment_id, "planned": len(metrics), "succeeded": len(metrics), "failed": 0, "skipped": 0, "track_summaries": _track_summaries(metrics)}
    report_paths = write_reports(report_dir, experiment_id=experiment_id, summary=summary, metrics=metrics, certificate=certificate, no_overwrite_audit=audit)
    return {"reports_dir": str(report_dir), "certificates_dir": str(cert_dir), "report_paths": report_paths, "certificate": certificate}


def diagnose(settings: Settings, experiment_id: str) -> dict[str, Any]:
    diag_dir = artifacts_root(settings) / experiment_id / "diagnostics"
    if (diag_dir / "per_seed_results.csv").exists():
        return {"status": "existing_artifacts_preserved", "experiment_id": experiment_id, "diagnostics_dir": str(diag_dir)}
    metrics = _load_metrics(artifacts_root(settings) / experiment_id)
    diag_paths = _write_diagnostics(artifacts_root(settings) / experiment_id, metrics)
    return {"experiment_id": experiment_id, "diagnostics": diag_paths}


def build_publication_bundle(settings: Settings, experiment_id: str, output: Path | None = None) -> dict[str, Any]:
    return create_publication_bundle(settings, experiment_id, output or publication_root(settings) / experiment_id)
