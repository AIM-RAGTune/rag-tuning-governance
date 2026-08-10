from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.adaptive_arch.reporting import CAUTION
from square_sim.utils.files import write_json, write_text


COMPONENTS = [
    "local_reconfiguration",
    "snapshotting",
    "conditional_forking",
    "nonlinear_rollout",
    "merge_reintegration",
    "architecture_memory",
    "dynamic_topology",
    "adaptive_compute_allocation",
    "regression_protection",
    "cost_awareness",
]


def _mean(group: pd.DataFrame, system: str, metric: str = "cost_adjusted_utility") -> float | None:
    rows = group[group["system"] == system]
    return None if rows.empty else float(rows[metric].mean())


def certificate_for_task(task: str, group: pd.DataFrame) -> dict[str, Any]:
    component_support = {component: "inconclusive" for component in COMPONENTS}
    full = _mean(group, "square_adaptive_arch_full")
    static = max([v for v in [_mean(group, "static_policy"), _mean(group, "linear_static_baseline"), _mean(group, "greedy_immediate")] if v is not None] or [None])
    if task == "random_unlearnable_control":
        status, reason = "Refused", "Random control has no learnable architecture signal."
    elif task == "linear_static_control":
        if static is not None and (full is None or static >= full - 0.01):
            status, reason = "Refused", "Linear/static control was won or tied by a static baseline, as expected; no adaptive support is awarded."
        else:
            status, reason = "Control failed", "Linear/static control did not pass the static-baseline sanity expectation."
    elif full is None:
        status, reason = "Inconclusive", "Missing square_adaptive_arch_full runs."
    else:
        best_baseline = max([v for v in [static, _mean(group, "coordinate_descent"), _mean(group, "evolutionary_search")] if v is not None] or [-1e9])
        no_fork = _mean(group, "square_adaptive_arch_no_fork")
        no_merge = _mean(group, "square_adaptive_arch_no_merge")
        if full > best_baseline + 0.01:
            status, reason = "Candidate architecture signal", "Adaptive architecture beat static/classical baselines under budget-normalized scoring."
        else:
            status, reason = "Inconclusive", "Adaptive architecture did not clearly beat static/classical baselines."
        if no_fork is not None and full > no_fork + 0.005:
            component_support["conditional_forking"] = "supported"
        if no_merge is not None and task == "merge_required_architecture" and full > no_merge + 0.005:
            component_support["merge_reintegration"] = "supported"
        comparisons = {
            "local_reconfiguration": ("local_regime_shift", "square_adaptive_arch_no_local_reconfiguration"),
            "architecture_memory": ("memory_prevents_repeated_failure", "square_adaptive_arch_no_memory"),
            "dynamic_topology": ("dynamic_topology_routing", "square_adaptive_arch_static_topology"),
            "nonlinear_rollout": ("nonlinear_extrapolation_required", "square_adaptive_arch_linear_rollout"),
            "regression_protection": ("protect_known_good_while_adapting", "square_adaptive_arch_no_regression_protection"),
            "adaptive_compute_allocation": ("compute_allocation_trap", "square_adaptive_arch_always_fork"),
        }
        for component, (required_task, ablation) in comparisons.items():
            ablation_score = _mean(group, ablation)
            if task == required_task and ablation_score is not None and full > ablation_score + 0.005:
                component_support[component] = "supported"
        if task in {"future_rollout_required", "compute_allocation_trap"} and no_fork is not None and full > no_fork + 0.005:
            component_support["snapshotting"] = "supported"
            component_support["cost_awareness"] = "supported"
    return {
        "task": task,
        "certificate_type": "Adaptive Architecture Certificate",
        "status": status,
        "reason": reason,
        "component_support": component_support,
        "caveats": [CAUTION],
    }


def write_certificates(project_root: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    out = project_root / "certificates" / "square_adaptive_arch" / "v1" / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    certs = []
    for task, group in metrics.groupby("task") if not metrics.empty else []:
        cert = certificate_for_task(str(task), group)
        task_dir = out / str(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "certificate.json", cert)
        write_text(task_dir / "certificate.md", f"# {task}\n\nStatus: **{cert['status']}**\n\n{cert['reason']}\n")
        certs.append(cert)
    index = {"experiment_id": experiment_id, "certificate_type": "Adaptive Architecture Certificate Index", "certificates": certs}
    write_json(out / "certificate_index.json", index)
    write_text(out / "certificate_index.md", "# Adaptive Architecture Certificate Index\n\n" + "\n".join(f"- `{c['task']}`: {c['status']}" for c in certs) + "\n")
    return index
