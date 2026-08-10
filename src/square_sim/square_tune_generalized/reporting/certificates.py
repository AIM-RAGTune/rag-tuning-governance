from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.square_tune_generalized.simulation.policies import is_square_variant
from square_sim.utils.files import write_json, write_text

STATUSES = {
    "Signal supported",
    "Candidate signal",
    "Inconclusive",
    "Refused",
    "Control failed",
    "Budget confounded",
    "Data unavailable",
    "License restricted / internal only",
    "Publication restricted",
    "Requires manual credentialed data",
}


def certificate_for_group(
    track: str,
    scenario: str,
    group: pd.DataFrame,
    *,
    stress_profile: str | None = None,
) -> dict[str, Any]:
    if group.empty:
        status = "Data unavailable"
        evidence = {}
    elif not bool(group.get("budget_parity_ok", pd.Series([True])).all()):
        status = "Budget confounded"
        evidence = {}
    else:
        by_system = group.groupby("system")["cost_adjusted_utility"].mean().sort_values(ascending=False)
        best_system = str(by_system.index[0])
        adaptive = float(by_system.get("square_tune_adaptive_compute", -999))
        best_non_square = float(group[~group["system"].map(is_square_variant)]["cost_adjusted_utility"].max())
        best_ablation = float(
            group[group["system"].str.contains("no_|static_topology", regex=True)]["cost_adjusted_utility"].max()
        )
        evidence = {
            "best_system": best_system,
            "square_tune_adaptive_compute": adaptive,
            "best_non_square": best_non_square,
            "best_ablation": best_ablation,
        }
        if "random" in scenario or scenario == "prediction_only_baseline" and best_system == "classical_only_baseline":
            status = "Refused"
        elif best_system == "square_tune_adaptive_compute" and adaptive > best_non_square + 0.01 and adaptive > best_ablation + 0.005:
            status = "Signal supported"
        elif adaptive > best_non_square + 0.01:
            status = "Candidate signal"
        elif best_non_square >= adaptive:
            status = "Refused"
        else:
            status = "Inconclusive"
        if bool(group.get("publication_restricted", pd.Series([False])).any()):
            status = "Publication restricted"
    if track == "patient_flow":
        caveat = "Healthcare operations proxy only; not diagnosis, treatment planning, or autonomous clinical recommendation."
    else:
        caveat = "Software benchmark evidence only; not SQUARE hardware validation."
    return {
        "track": track,
        "scenario": scenario,
        "stress_profile": stress_profile or "nominal",
        "status": status,
        "evidence": evidence,
        "caveats": [caveat],
    }


def write_generalized_certificates(certificate_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    certificate_dir.mkdir(parents=True, exist_ok=True)
    certificates: list[dict[str, Any]] = []
    if not metrics.empty:
        group_cols = ["track", "scenario"]
        if "stress_profile" in metrics:
            group_cols.append("stress_profile")
        for keys, group in metrics.groupby(group_cols):
            if len(group_cols) == 3:
                track, scenario, stress_profile = keys
            else:
                track, scenario = keys
                stress_profile = "nominal"
            cert = certificate_for_group(str(track), str(scenario), group, stress_profile=str(stress_profile))
            out = certificate_dir / str(track) / str(scenario) / str(stress_profile)
            out.mkdir(parents=True, exist_ok=True)
            write_json(out / "certificate.json", cert)
            write_text(
                out / "certificate.md",
                f"# Generalized Optimization Certificate: {track}/{scenario}/{stress_profile}\n\nStatus: `{cert['status']}`\n\n{cert['caveats'][0]}\n",
            )
            certificates.append(cert)
    index = {
        "experiment_id": experiment_id,
        "certificate_type": "SQUARETune Generalized Optimization Certificate",
        "statuses": sorted(STATUSES),
        "certificates": certificates,
    }
    write_json(certificate_dir / "certificate_index.json", index)
    write_text(
        certificate_dir / "certificate_index.md",
        "# SQUARETune Generalized Certificate Index\n\n"
        + "\n".join(f"- `{c['track']}/{c['scenario']}/{c['stress_profile']}`: `{c['status']}`" for c in certificates)
        + "\n",
    )
    return index
