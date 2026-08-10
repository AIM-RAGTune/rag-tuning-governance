from __future__ import annotations

from pathlib import Path

from square_sim.data.schema import DatasetConfig
from square_sim.utils.files import latest_child, read_json

EMBEDDED_DATASETS = {
    "energy": {
        "display_name": "SPECTRA Energy",
        "kaggle_slug": "javierfalcondale/spectra-energy-quantum-ready-steel-plant-data",
        "expected_targets": ["target", "target_real", "in_pocket"],
        "preferred_first_target": "target",
    },
    "oilgas": {
        "display_name": "SPECTRA Oil & Gas",
        "kaggle_slug": "javierfalcondale/spectra-oil-and-gas-quantum-ready-gas-turbine-data",
        "expected_targets": ["target", "target_real", "in_pocket"],
        "preferred_first_target": "target",
    },
    "maintenance": {
        "display_name": "SPECTRA Maintenance",
        "kaggle_slug": "javierfalcondale/spectra-maintenance-quantum-ready-failure-data",
        "expected_targets": ["target", "target_real", "in_pocket"],
        "preferred_first_target": "target",
    },
    "telecom": {
        "display_name": "SPECTRA Telecom",
        "kaggle_slug": "javierfalcondale/spectra-telecom-quantum-ready-churn-data",
        "expected_targets": ["target", "target_real", "in_pocket"],
        "preferred_first_target": "target",
    },
}


def load_dataset_configs(config_path: Path | None = None) -> dict[str, DatasetConfig]:
    data = {"datasets": EMBEDDED_DATASETS}
    path = config_path or Path("configs/datasets.yaml")
    if path.exists():
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except ImportError:
            pass
    return {
        name: DatasetConfig(
            name=name,
            display_name=cfg["display_name"],
            kaggle_slug=cfg["kaggle_slug"],
            expected_targets=list(cfg["expected_targets"]),
            preferred_first_target=cfg["preferred_first_target"],
        )
        for name, cfg in data["datasets"].items()
    }


def latest_processed_version(project_root: Path, dataset_name: str) -> str:
    latest = latest_child(project_root / "datasets" / "processed" / dataset_name)
    if latest is None:
        raise FileNotFoundError(
            f"No processed version for dataset '{dataset_name}'. Run `square-sim data normalize "
            f"--dataset {dataset_name}` first."
        )
    return latest.name


def show_catalog(project_root: Path, dataset_name: str) -> dict:
    root = project_root / "datasets" / "processed" / dataset_name
    versions = []
    if root.exists():
        for version_dir in sorted(root.iterdir()):
            manifest = version_dir / "schema.json"
            versions.append(
                {
                    "version": version_dir.name,
                    "path": str(version_dir),
                    "schema": read_json(manifest) if manifest.exists() else None,
                }
            )
    return {"dataset": dataset_name, "versions": versions}

