from __future__ import annotations

from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.tune.external.protection import ProtectedResultsRegistry

PRIOR_EXPERIMENT_IDS = [
    "square_tune_calibration_v2_matrix_20260731-135458-7829d0a8bd",
    "square_tune_external_v1_square_tune_external_v1_minimal_expanded_20260802-003119-62efdb463e",
    "square_tune_external_v1_square_tune_external_v1_ragtruth_full_matrix_20260802-012118-136d1eefab",
    "square_tune_external_v1_square_tune_external_v2_adaptive_compute_ragtruth_20260802-021733-b3cf204dc6",
    "square_adaptive_arch_v1_external_proxy_20260802-030901-0cdcd779b7",
    "square_core_validation_v1_full_matrix_20260802-152150-d2d7bb0cc3",
    "square_tune_matched_cost_rag_v1_full_matrix_20260802-232631-5449febfb3",
]


def next_sim_prior_paths(settings: Settings) -> list[Path]:
    root = settings.project_root
    return [
        root / "reports" / "square_tune" / "calibration" / PRIOR_EXPERIMENT_IDS[0],
        root / "certificates" / "square_tune" / "calibration" / PRIOR_EXPERIMENT_IDS[0],
        root / "reports" / "square_tune" / "external_transfer" / PRIOR_EXPERIMENT_IDS[1],
        root / "reports" / "square_tune" / "external_transfer" / PRIOR_EXPERIMENT_IDS[2],
        root / "reports" / "square_tune" / "external_transfer" / PRIOR_EXPERIMENT_IDS[3],
        root / "reports" / "square_adaptive_arch" / "v1" / PRIOR_EXPERIMENT_IDS[4],
        root / "reports" / "square_core" / "v1" / PRIOR_EXPERIMENT_IDS[5],
        root / "reports" / "square_tune" / "matched_cost_rag" / "v1" / PRIOR_EXPERIMENT_IDS[6],
        root / "certificates" / "square_tune" / "matched_cost_rag" / "v1" / PRIOR_EXPERIMENT_IDS[6],
        root / "publication" / "square_tune_matched_cost_rag" / "v1" / PRIOR_EXPERIMENT_IDS[6],
    ]


def protect_prior(settings: Settings) -> dict[str, Any]:
    registry = ProtectedResultsRegistry(settings)
    payload = registry.protect_defaults(notes="SQUARE Next Simulation Package v1 prior protection")
    existing = {str(Path(row["path"])) for row in payload.get("protected_paths", [])}
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    for path in next_sim_prior_paths(settings):
        if not path.exists() or str(path) in existing:
            continue
        payload.setdefault("protected_paths", []).append(
            {
                "path": str(path),
                "reason": "protected prior result for SQUARE Next Simulation Package v1",
                "created_at_utc": now,
                "added_by_command": "square-sim next-sim protect-prior",
                "notes": "append-only evidence artifact",
            }
        )
    payload["updated_at_utc"] = now
    registry.save(payload)
    return payload

