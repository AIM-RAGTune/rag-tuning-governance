from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    kaggle_slug: str
    expected_targets: list[str]
    preferred_first_target: str


TARGET_ROLES = {
    "target": "target",
    "target_real": "target",
    "in_pocket": "pocket flag",
}

