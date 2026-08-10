from __future__ import annotations

from typing import Any


def license_record(
    *,
    dataset_key: str,
    source_type: str,
    license_status: str,
    publication_safe: bool,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "source_type": source_type,
        "license_status": license_status,
        "publication_safe": publication_safe,
        "allowed_use_caveats": caveats or [],
    }


def generalized_license_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "datasets": records,
        "publication_safe_dataset_count": sum(bool(row.get("publication_safe")) for row in records),
        "restricted_dataset_count": sum(not bool(row.get("publication_safe")) for row in records),
        "healthcare_caveat": (
            "Healthcare operations datasets are operations proxies only; no clinical diagnosis, "
            "treatment planning, or autonomous clinical decision-making is supported."
        ),
    }
