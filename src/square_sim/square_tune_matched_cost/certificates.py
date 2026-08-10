from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text

STATUS_VALUES = [
    "Signal supported",
    "Candidate signal",
    "Inconclusive",
    "Refused",
    "Control failed",
    "Budget confounded",
    "Data unavailable",
    "Dependency unavailable",
    "Utility fragile",
    "Negative result",
]

KILL_BASELINES = [
    "random_gating_matched_cost",
    "uncertainty_threshold_gating_matched_cost",
    "retrieval_confidence_gating_matched_cost",
    "square_tune_no_fork",
]


def evaluate_certificate(metrics: pd.DataFrame, sensitivity: pd.DataFrame, *, no_overwrite_status: str) -> dict[str, Any]:
    if metrics.empty:
        return {"status": "Data unavailable", "kill_criteria": {"data_available": False}, "evidence": {}}
    if not bool(metrics.get("real_data_used", pd.Series([False])).all()):
        return {"status": "Data unavailable", "kill_criteria": {"real_data_used": False}, "evidence": {}}
    if no_overwrite_status != "append_only_confirmed":
        return {"status": "Control failed", "kill_criteria": {"no_overwrite_status": no_overwrite_status}, "evidence": {}}
    if bool(metrics.get("budget_confounded_flag", pd.Series([False])).any()):
        return {"status": "Budget confounded", "kill_criteria": {"budget_parity": False}, "evidence": {}}
    means = metrics.groupby("system")["held_out_test_cost_adjusted_utility"].mean()
    raw = metrics.groupby("system")["held_out_test_raw_quality"].mean()
    adaptive = float(means.get("square_tune_adaptive_compute", -999.0))
    baseline_results = {name: float(adaptive - means.get(name, 999.0)) for name in KILL_BASELINES}
    seed_wins = {}
    for name in KILL_BASELINES:
        joined = metrics[metrics["system"].isin(["square_tune_adaptive_compute", name])].pivot_table(
            index="seed",
            columns="system",
            values="held_out_test_cost_adjusted_utility",
            aggfunc="mean",
        )
        if {"square_tune_adaptive_compute", name}.issubset(joined.columns):
            seed_wins[name] = int((joined["square_tune_adaptive_compute"] > joined[name]).sum())
    top_rows = sensitivity[sensitivity["rank"] == 1] if not sensitivity.empty else pd.DataFrame()
    adaptive_best_count = int(top_rows["system"].eq("square_tune_adaptive_compute").sum()) if not top_rows.empty else 0
    adaptive_top3_count = int(sensitivity[(sensitivity["system"] == "square_tune_adaptive_compute") & (sensitivity["rank"] <= 3)].shape[0]) if not sensitivity.empty else 0
    no_fork_raw = float(raw.get("square_tune_no_fork", 0.0))
    raw_preserved = float(raw.get("square_tune_adaptive_compute", 0.0)) >= 0.95 * no_fork_raw
    kill_criteria = {
        "beats_random_gating": baseline_results.get("random_gating_matched_cost", -1) > 0,
        "beats_uncertainty_gating": baseline_results.get("uncertainty_threshold_gating_matched_cost", -1) > 0,
        "beats_retrieval_confidence_gating": baseline_results.get("retrieval_confidence_gating_matched_cost", -1) > 0,
        "beats_no_fork": baseline_results.get("square_tune_no_fork", -1) > 0,
        "raw_quality_preserved_vs_no_fork": raw_preserved,
        "beats_random_at_least_4_of_5": seed_wins.get("random_gating_matched_cost", 0) >= 4,
        "beats_uncertainty_at_least_3_of_5": seed_wins.get("uncertainty_threshold_gating_matched_cost", 0) >= 3,
        "adaptive_best_utility_settings": adaptive_best_count,
        "adaptive_top3_utility_settings": adaptive_top3_count,
    }
    if all(
        bool(kill_criteria[k])
        for k in [
            "beats_random_gating",
            "beats_uncertainty_gating",
            "beats_retrieval_confidence_gating",
            "beats_no_fork",
            "raw_quality_preserved_vs_no_fork",
            "beats_random_at_least_4_of_5",
            "beats_uncertainty_at_least_3_of_5",
        ]
    ) and adaptive_best_count >= 3:
        status = "Signal supported"
    elif baseline_results.get("random_gating_matched_cost", 0) <= 0 or baseline_results.get("uncertainty_threshold_gating_matched_cost", 0) <= 0:
        status = "Negative result"
    elif adaptive_top3_count < 3:
        status = "Utility fragile"
    elif sum(bool(v) for v in kill_criteria.values() if isinstance(v, bool)) >= 5:
        status = "Candidate signal"
    else:
        status = "Inconclusive"
    return {
        "certificate_type": "SQUARETune Matched-Cost RAG Certificate",
        "status": status,
        "kill_criteria": kill_criteria,
        "evidence": {
            "adaptive_compute_mean_cost_adjusted": adaptive,
            "baseline_deltas": baseline_results,
            "seed_wins": seed_wins,
            "adaptive_best_utility_settings": adaptive_best_count,
            "adaptive_top3_utility_settings": adaptive_top3_count,
        },
        "caveats": [
            "Software RAG policy kill-test only.",
            "No SQUARE hardware, quantum architecture, clinical, or commercial claim follows from this certificate.",
            "Oracle upper bound is diagnostic only and is not a valid competitor.",
        ],
    }


def write_certificate(certificate_dir: Path, experiment_id: str, cert: dict[str, Any]) -> dict[str, Any]:
    certificate_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment_id": experiment_id, **cert, "allowed_statuses": STATUS_VALUES}
    write_json(certificate_dir / "certificate.json", payload)
    write_text(
        certificate_dir / "certificate.md",
        f"# SQUARETune Matched-Cost RAG Certificate\n\nStatus: `{payload['status']}`\n\n"
        "This is a matched-cost RAG software kill-test. It does not prove SQUARE hardware or commercial viability.\n",
    )
    write_json(certificate_dir / "certificate_index.json", {"experiment_id": experiment_id, "certificates": [payload]})
    write_text(certificate_dir / "certificate_index.md", f"# Certificate Index\n\n- `real_rag_policy_matched_cost`: `{payload['status']}`\n")
    return payload

