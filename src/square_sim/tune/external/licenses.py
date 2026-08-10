from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from square_sim.utils.files import write_json

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def license_status(item: dict[str, Any], *, allow_unknown_license: bool = False) -> str:
    value = item.get("license") or item.get("license_override")
    if value:
        return "captured"
    if allow_unknown_license:
        return "unknown"
    return "missing"


def license_metadata(item: dict[str, Any], *, allow_unknown_license: bool = False) -> dict[str, Any]:
    status = license_status(item, allow_unknown_license=allow_unknown_license)
    return {
        "license": item.get("license") or item.get("license_override"),
        "license_status": status,
        "allowed_licenses": item.get("allowed_licenses", []),
        "allowed_use_note": (
            "internal-benchmark-only until license reviewed"
            if status != "captured"
            else "use according to captured source license"
        ),
        "source_type": item.get("source_type"),
        "source_id": item.get("dataset_id") or item.get("repository") or item.get("manual_import_path"),
    }


def write_license_metadata(path: Path, item: dict[str, Any], *, allow_unknown_license: bool = False) -> dict[str, Any]:
    payload = license_metadata(item, allow_unknown_license=allow_unknown_license)
    write_json(path, payload)
    return payload


def scan_pii_phi_texts(texts: list[str]) -> dict[str, Any]:
    joined = "\n".join(texts[:1000])
    warnings: list[str] = []
    counts = {
        "email_like": len(EMAIL_RE.findall(joined)),
        "phone_like": len(PHONE_RE.findall(joined)),
        "ssn_like": len(SSN_RE.findall(joined)),
    }
    for key, count in counts.items():
        if count:
            warnings.append(f"PII/PHI heuristic warning: {key} pattern count={count}")
    return {
        "counts": counts,
        "email_like_count": counts["email_like"],
        "phone_like_count": counts["phone_like"],
        "ssn_like_count": counts["ssn_like"],
        "warnings": warnings,
        "limitations": "Names and domain-specific patient identifiers are not reliably detected.",
    }
