from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.square_tune_generalized.simulation.runner import certificates_root, reports_root
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import write_json, write_text
from square_sim.utils.write_once import WriteOncePathManager


def create_publication_bundle(settings: Settings, experiment_id: str, output: Path) -> dict[str, Any]:
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(output.parent, registry.protected_paths())
    manager.ensure_writable_path(output)
    output.mkdir(parents=True)
    report_dir = reports_root(settings) / experiment_id
    certificate_dir = certificates_root(settings) / experiment_id
    for filename in [
        "generalized_benchmark_summary.md",
        "generalized_benchmark_summary.json",
        "aggregate_metrics.parquet",
        "aggregate_metrics.csv",
        "publication_readiness_report.json",
        "no_overwrite_audit.md",
        "no_overwrite_audit.json",
    ]:
        src = report_dir / filename
        if src.exists():
            shutil.copy2(src, output / filename)
    if certificate_dir.exists():
        shutil.copytree(certificate_dir, output / "certificates", dirs_exist_ok=False)
    for subdir in ["benchmark_cards", "scenario_cards", "configs", "ablation_tables", "plots"]:
        (output / subdir).mkdir()
    manifest = {
        "bundle_id": output.name,
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "restricted_raw_data_excluded": True,
        "credentialed_healthcare_data_excluded": True,
        "contents": sorted(path.name for path in output.iterdir()),
    }
    write_json(output / "reproducibility_manifest.json", manifest)
    write_text(
        output / "README_publication.md",
        "# SQUARETune Generalized Benchmark Publication Bundle\n\n"
        "This bundle contains aggregate metrics, certificates, configs, and reproducibility metadata. "
        "It excludes restricted raw datasets, MIMIC rows, credentials, model weights, and large tensors.\n",
    )
    write_text(output / "methods.md", "Adaptive compute is evaluated against static, greedy, evolutionary, optional Bayesian, SQUARETune full, and ablation systems under budget parity.\n")
    write_text(output / "datasets_and_licenses.md", "Dataset manifests and license summaries are referenced; restricted raw data is excluded.\n")
    write_text(output / "limitations.md", "Software benchmark evidence only. No hardware, clinical, or commercial proof is claimed.\n")
    return manifest
