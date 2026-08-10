from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.utils.files import write_text


def write_run_explanation(path: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> Path:
    lines = [
        f"# SQUARETune Run Explanation: {manifest['run_id']}",
        "",
        "## Objective",
        "",
        "This run tests an LLM adaptation optimizer in a controlled SQUARETune response-surface simulator.",
        "",
        "## Scientific Framing",
        "",
        "SQUARETune is an ontology-level comparison. It is not physical hardware validation.",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{manifest['dataset_key']}`",
        f"- Dataset version: `{manifest['dataset_version_id']}`",
        f"- Generator manifest hash: `{manifest['generator_manifest_hash']}`",
        f"- Protocol hash: `{manifest['protocol_hash']}`",
        "",
        "## Optimizer",
        "",
        f"- Optimizer: `{manifest['model_or_optimizer_name']}`",
        f"- Ablation flags: `{manifest['ablation_flags']}`",
        f"- Budget: `{manifest['budget_config']}`",
        "",
        "## Metrics",
        "",
    ]
    for key in [
        "final_utility",
        "utility_improvement",
        "cost_adjusted_improvement",
        "regression_count",
        "experiments_to_threshold",
        "simulated_gpu_hours",
    ]:
        lines.append(f"- {key}: `{metrics.get(key)}`")
    lines.extend(
        [
            "",
            "## Trajectory And Branch Behavior",
            "",
            f"- Trajectory: `{manifest['trajectory_path']}`",
            f"- Branch diagnostics: `{manifest['branch_diagnostics_path']}`",
            f"- Final policy: `{manifest['final_policy_path']}`",
            "",
            "## Caveats",
            "",
            "- Synthetic mechanism diagnostics do not establish external benchmark performance.",
            "- Results should be interpreted as simulation-supported signals only when controls and ablations agree.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")
    return path

