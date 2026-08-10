from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any

TARGET_COLUMNS = {"target", "target_real", "in_pocket"}
LEAKAGE_NAME_TERMS = ("target", "label", "outcome", "churn", "failure", "pocket", "class", "segment")


@dataclass(frozen=True)
class FeaturePolicy:
    exclude_roles: list[str] = field(default_factory=lambda: ["target", "target_real", "pocket_flag"])
    exclude_columns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    include_columns: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)
    leakage_warning_policy: str = "warn_and_downgrade_certificate"
    acknowledged_leakage_warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, payload: dict[str, Any] | None) -> FeaturePolicy:
        payload = payload or {}
        return cls(
            exclude_roles=list(payload.get("exclude_roles", ["target", "target_real", "pocket_flag"])),
            exclude_columns=list(payload.get("exclude_columns", [])),
            exclude_patterns=list(payload.get("exclude_patterns", [])),
            include_columns=list(payload.get("include_columns", [])),
            include_patterns=list(payload.get("include_patterns", [])),
            leakage_warning_policy=str(
                payload.get("leakage_warning_policy", "warn_and_downgrade_certificate")
            ),
            acknowledged_leakage_warnings=list(payload.get("acknowledged_leakage_warnings", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclude_roles": self.exclude_roles,
            "exclude_columns": self.exclude_columns,
            "exclude_patterns": self.exclude_patterns,
            "include_columns": self.include_columns,
            "include_patterns": self.include_patterns,
            "leakage_warning_policy": self.leakage_warning_policy,
            "acknowledged_leakage_warnings": self.acknowledged_leakage_warnings,
        }


@dataclass(frozen=True)
class FeatureSelection:
    selected_features: list[str]
    excluded_features: dict[str, str]
    leakage_warnings: list[str]
    unacknowledged_leakage_warnings: list[str]
    policy: FeaturePolicy

    def to_manifest(self, *, dataset_version_id: str, split_id: str, target: str) -> dict[str, Any]:
        return {
            "dataset_version_id": dataset_version_id,
            "split_id": split_id,
            "target": target,
            "selected_features": self.selected_features,
            "excluded_features": self.excluded_features,
            "feature_count": len(self.selected_features),
            "leakage_warnings": self.leakage_warnings,
            "unacknowledged_leakage_warnings": self.unacknowledged_leakage_warnings,
            "feature_policy": self.policy.to_dict(),
        }


def _column_roles(schema: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for col in schema.get("columns", []) if isinstance(schema.get("columns"), list) else []:
        if isinstance(col, dict) and col.get("name"):
            roles[str(col["name"])] = str(col.get("role", "unknown"))
    return roles


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) or re.search(pattern, name) for pattern in patterns)


def _is_acknowledged(warning: str, acknowledged: list[str]) -> bool:
    lowered = warning.lower()
    return any(item.lower() in lowered for item in acknowledged)


def infer_name_warnings(columns: list[str], target: str) -> list[str]:
    warnings = []
    for column in columns:
        lowered = column.lower()
        if column in TARGET_COLUMNS or column == target:
            continue
        if any(term in lowered for term in LEAKAGE_NAME_TERMS):
            warnings.append(f"Potential leakage-like column name: {column}")
    return warnings


def select_features(
    columns: list[str],
    schema: dict[str, Any],
    target: str,
    policy: FeaturePolicy,
    leakage_warnings: list[str] | None = None,
) -> FeatureSelection:
    roles = _column_roles(schema)
    excluded: dict[str, str] = {}
    selected: list[str] = []
    explicit_includes = set(policy.include_columns)
    inferred_warnings = infer_name_warnings(columns, target)
    all_warnings = list(dict.fromkeys((leakage_warnings or []) + inferred_warnings))

    for column in columns:
        role = roles.get(column, "unknown")
        if column == target:
            excluded[column] = "active target"
        elif column in TARGET_COLUMNS:
            excluded[column] = "known SPECTRA target/pocket column"
        elif role in policy.exclude_roles:
            excluded[column] = f"schema role excluded: {role}"
        elif column in policy.exclude_columns:
            excluded[column] = "explicit column exclusion"
        elif policy.exclude_patterns and _matches_any(column, policy.exclude_patterns):
            excluded[column] = "explicit pattern exclusion"
        elif policy.include_columns or policy.include_patterns:
            if column in explicit_includes or _matches_any(column, policy.include_patterns):
                selected.append(column)
            else:
                excluded[column] = "not included by explicit include policy"
        else:
            selected.append(column)

    unacknowledged = [
        warning
        for warning in all_warnings
        if not _is_acknowledged(warning, policy.acknowledged_leakage_warnings)
    ]
    if not selected:
        raise ValueError(
            "Feature policy produced zero features. Relax include/exclude settings or inspect schema roles."
        )
    return FeatureSelection(
        selected_features=selected,
        excluded_features=excluded,
        leakage_warnings=all_warnings,
        unacknowledged_leakage_warnings=unacknowledged,
        policy=policy,
    )
