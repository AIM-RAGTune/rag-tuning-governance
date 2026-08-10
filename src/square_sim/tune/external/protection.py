from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.utils.files import read_json, write_json
from square_sim.utils.write_once import WriteOnceError


def protection_manifest_path(settings: Settings) -> Path:
    return settings.project_root / "protection" / "protected_results.json"


def default_protected_paths(settings: Settings) -> list[Path]:
    root = settings.project_root
    candidates = [
        root / "reports" / "square_tune" / "calibration",
        root / "certificates" / "square_tune" / "calibration",
        root / "datasets" / "synthetic" / "square_tune_calibration_v2",
        root / "reports" / "certificates",
        root / "reports" / "square_tune" / "external_transfer" / "square_tune_external_v1_smoke_20260801-231731-2582f9972a",
        root / "certificates" / "square_tune" / "external_transfer" / "square_tune_external_v1_smoke_20260801-231731-2582f9972a",
        root
        / "reports"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v1_minimal_expanded_20260802-003119-62efdb463e",
        root
        / "certificates"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v1_minimal_expanded_20260802-003119-62efdb463e",
        root
        / "reports"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v1_ragtruth_full_matrix_20260802-012118-136d1eefab",
        root
        / "certificates"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v1_ragtruth_full_matrix_20260802-012118-136d1eefab",
        root
        / "reports"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v2_adaptive_compute_ragtruth_20260802-021733-b3cf204dc6",
        root
        / "certificates"
        / "square_tune"
        / "external_transfer"
        / "square_tune_external_v1_square_tune_external_v2_adaptive_compute_ragtruth_20260802-021733-b3cf204dc6",
        root
        / "reports"
        / "square_adaptive_arch"
        / "v1"
        / "square_adaptive_arch_v1_external_proxy_20260802-030901-0cdcd779b7",
        root
        / "certificates"
        / "square_adaptive_arch"
        / "v1"
        / "square_adaptive_arch_v1_external_proxy_20260802-030901-0cdcd779b7",
        root
        / "reports"
        / "square_core"
        / "v1"
        / "square_core_validation_v1_full_matrix_20260802-152150-d2d7bb0cc3",
        root
        / "certificates"
        / "square_core"
        / "v1"
        / "square_core_validation_v1_full_matrix_20260802-152150-d2d7bb0cc3",
        root / "datasets" / "external" / "square_tune_v1",
    ]
    return [path for path in candidates if path.exists()]


@dataclass
class ProtectedResultsRegistry:
    settings: Settings

    @property
    def path(self) -> Path:
        return protection_manifest_path(self.settings)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"protected_paths": [], "created_at_utc": None, "updated_at_utc": None}
        return read_json(self.path)

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, payload)

    def protect_defaults(self, *, notes: str = "Calibration v2 prior results") -> dict[str, Any]:
        payload = self.load()
        existing = {str(Path(row["path"]).expanduser()) for row in payload.get("protected_paths", [])}
        now = datetime.now(timezone.utc).isoformat()
        if not payload.get("created_at_utc"):
            payload["created_at_utc"] = now
        for path in default_protected_paths(self.settings):
            key = str(path)
            if key in existing:
                continue
            payload.setdefault("protected_paths", []).append(
                {
                    "path": key,
                    "reason": "protected prior SQUARETune/SQUARESim result",
                    "created_at_utc": now,
                    "added_by_command": "square-sim tune external protect-prior",
                    "notes": notes,
                }
            )
        payload["updated_at_utc"] = now
        self.save(payload)
        return payload

    def protected_paths(self) -> list[Path]:
        return [Path(str(row["path"])) for row in self.load().get("protected_paths", [])]

    def assert_not_protected(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        for protected in self.protected_paths():
            protected_resolved = protected.expanduser().resolve()
            if resolved == protected_resolved or protected_resolved in resolved.parents:
                raise WriteOnceError(f"Refusing to write inside protected result path: {path}")
