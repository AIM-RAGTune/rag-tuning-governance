from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.utils.files import write_text


def component_flags(model_name: str) -> dict[str, str]:
    is_snapshot = model_name.startswith("squaresim_snapshot")
    return {
        "phase_encoding": "inactive" if model_name in {"squaresim_no_phase", "squaresim_snapshot_no_phase"} else "active",
        "emitters": "static" if model_name == "squaresim_static_emitters" else "active",
        "scalar_field_grid": "active" if model_name.startswith("squaresim") else "not applicable",
        "nonlinear_dynamics": "inactive" if model_name in {"squaresim_no_nonlinear", "squaresim_linear_field_only", "squaresim_snapshot_no_nonlinear", "squaresim_snapshot_linear_rollout"} else "active",
        "memory_zones": "inactive" if model_name in {"squaresim_no_memory", "squaresim_linear_field_only", "squaresim_snapshot_no_memory"} else "active",
        "overlap_zones": "inactive" if model_name == "squaresim_no_overlap_zones" else "active",
        "feedback": "inactive" if model_name in {"squaresim_no_feedback", "squaresim_linear_field_only", "squaresim_snapshot_no_feedback"} else "active",
        "snapshot_rollout": "active" if is_snapshot else "inactive",
        "snapshot_forking": "inactive" if model_name == "squaresim_snapshot_no_fork" else ("active" if is_snapshot else "not applicable"),
        "merge_reintegration": "inactive" if model_name == "squaresim_snapshot_no_merge" else ("active" if is_snapshot else "not applicable"),
    }


def render_template_fallback(payload: dict[str, Any]) -> str:
    metrics = "\n".join(f"- {k}: {v}" for k, v in payload["metrics"].items())
    resources = "\n".join(f"- {k}: {v}" for k, v in payload["resources"].items())
    components = "\n".join(f"- {k}: {v}" for k, v in payload["components"].items())
    return f"""# SQUARESim Run Explanation: {payload['run_id']}

## Objective
Evaluate a SQUARE-inspired computational ontology against relevant baselines.

## Dataset
- Dataset: {payload['dataset']}
- Dataset version: {payload['dataset_version']}
- Source: {payload['source']}
- Target: {payload['target']}

## Model
- Model tested: {payload['model']}
{components}

## Results
{metrics}

## Resource Use
{resources}

## Status
Preliminary run status: {payload['status']}

## Caveats
- This is not physical SQUARE hardware validation.
- This is not proof of quantum advantage.
- Confidence bounds may be insufficient until bootstrap comparisons and ablations are complete.

## Recommended Next Run
Run the full comparison matrix and certificate generator.
"""


def generate_run_explanation(output_path: Path, payload: dict[str, Any]) -> str:
    payload = dict(payload)
    payload.setdefault("components", component_flags(payload["model"]))
    try:
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent / "templates"
        template = Environment(loader=FileSystemLoader(template_dir), autoescape=False).get_template(
            "run_explanation.md.j2"
        )
        text = template.render(**payload)
    except ImportError:
        text = render_template_fallback(payload)
    write_text(output_path, text)
    return text
