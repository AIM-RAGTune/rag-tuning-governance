from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.utils.files import write_json, write_text


def _winner(frame: pd.DataFrame, *, include_diagnostics: bool = False) -> str:
    if frame.empty:
        return "none"
    competitors = frame if include_diagnostics else frame[~frame["system"].astype(str).str.contains("oracle")]
    if competitors.empty:
        competitors = frame
    row = competitors.sort_values("cost_adjusted_utility", ascending=False).iloc[0]
    return str(row["system"])


def evaluate_track_certificate(track: str, metrics: pd.DataFrame) -> dict[str, Any]:
    frame = metrics[metrics["track"] == track].copy()
    if frame.empty:
        return {"track": track, "status": "Data unavailable", "evidence": []}
    means = frame.groupby("system", as_index=False).mean(numeric_only=True)
    winner = _winner(means)
    evidence: list[str] = [f"cost_adjusted_winner={winner}"]
    status = "Inconclusive"
    if track == "rag_hard_subset_v1":
        adaptive = means[means["system"].isin(["square_tune_adaptive_compute", "square_tune_hard_subset_escalation"])]
        no_fork = means[means["system"] == "square_tune_no_fork"]
        if not adaptive.empty and not no_fork.empty and adaptive["hard_subset_performance"].max() > float(no_fork["hard_subset_performance"].iloc[0]):
            status = "Candidate signal"
            evidence.append("adaptive compute improved at least one predefined hard subset")
        else:
            status = "Negative result"
            evidence.append("no-fork matched or beat adaptive compute across hard-subset metrics")
    elif track == "no_fork_robustness_v1":
        competitors = means[~means["system"].astype(str).str.contains("oracle")]
        no_fork_rank = competitors["cost_adjusted_utility"].rank(ascending=False, method="min")[
            competitors["system"] == "square_tune_no_fork"
        ]
        status = "Signal supported" if not no_fork_rank.empty and int(no_fork_rank.iloc[0]) <= 3 else "Candidate signal"
        evidence.append("no-fork evaluated as commercial RAG default candidate; oracle is a ceiling diagnostic only")
    elif track == "adaptive_escalation_v2":
        candidates = means[means["system"].str.contains("escalation")]
        no_fork = means[means["system"] == "square_tune_no_fork"]
        full = means[means["system"] == "square_tune_full"]
        if not candidates.empty and not no_fork.empty and not full.empty and candidates["cost_adjusted_utility"].max() > max(float(no_fork["cost_adjusted_utility"].iloc[0]), float(full["cost_adjusted_utility"].iloc[0])):
            status = "Signal supported"
        elif not candidates.empty and candidates["cost_adjusted_utility"].max() >= means["cost_adjusted_utility"].quantile(0.75):
            status = "Candidate signal"
        else:
            status = "Refused"
    elif track == "claim_level_faithfulness_v1":
        status = "Candidate signal" if winner in {"claim_risk_escalation", "adaptive_compute"} else "Inconclusive"
        evidence.append("claim-level faithfulness is a proxy unless human claim labels are present")
    elif track == "elastic_compute_policy_v1":
        status = "Candidate signal" if winner in {"square_tune_adaptive_compute", "square_adaptive_arch_adaptive_compute"} else "Proxy only"
        evidence.append("elastic compute trace is synthetic proxy unless external cluster trace is imported")
    elif track in {"square_core_v2_field_substrate_targeted", "square_core_v2_closed_loop_targeted"}:
        status = "Candidate signal" if "square" in winner else "Inconclusive"
        evidence.append("targeted SQUARE Core v2 refinement; not hardware validation")
    return {"track": track, "status": status, "winner": winner, "evidence": evidence}


def write_certificates(cert_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    tracks = sorted(metrics["track"].unique().tolist()) if not metrics.empty else []
    track_certs = [evaluate_track_certificate(track, metrics) for track in tracks]
    statuses = [cert["status"] for cert in track_certs]
    if "Signal supported" in statuses or "Candidate signal" in statuses:
        global_status = "Candidate signal"
    elif "Negative result" in statuses:
        global_status = "Negative result"
    else:
        global_status = "Inconclusive"
    payload = {
        "certificate_type": "SQUARE Next Simulation Certificate",
        "experiment_id": experiment_id,
        "status": global_status,
        "track_certificates": track_certs,
        "caveats": [
            "Software simulation only.",
            "No physical SQUARE hardware, quantum advantage, clinical, or commercial claim follows from this certificate.",
        ],
    }
    write_json(cert_dir / "certificate_index.json", payload)
    lines = ["# SQUARE Next Simulation Certificate", "", f"- Experiment: `{experiment_id}`", f"- Status: {global_status}", ""]
    for cert in track_certs:
        lines.append(f"## {cert['track']}")
        lines.append(f"- Status: {cert['status']}")
        lines.append(f"- Winner: {cert.get('winner', 'none')}")
        lines.extend(f"- {item}" for item in cert.get("evidence", []))
        lines.append("")
    write_text(cert_dir / "certificate_index.md", "\n".join(lines))
    return payload
