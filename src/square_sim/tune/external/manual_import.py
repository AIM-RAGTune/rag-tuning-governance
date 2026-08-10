from __future__ import annotations

from pathlib import Path

from square_sim.tune.external.acquire import import_manual_dataset


def manual_import_dataset(
    dataset_key: str,
    source_path: Path,
    *,
    output_root: Path,
    license_note: Path | None = None,
    scenario_families: list[str] | None = None,
    allow_unknown_license: bool = False,
    max_rows: int | None = None,
) -> dict:
    return import_manual_dataset(
        dataset_key,
        source_path,
        output_root=output_root,
        license_note=license_note,
        scenario_families=scenario_families,
        allow_unknown_license=allow_unknown_license,
        max_rows=max_rows,
    )
