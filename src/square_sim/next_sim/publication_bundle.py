from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.next_sim.paths import artifacts_root, certificates_root, reports_root
from square_sim.tune.external.protection import ProtectedResultsRegistry
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.write_once import WriteOncePathManager


def create_publication_bundle(settings: Settings, experiment_id: str, output: Path) -> dict[str, Any]:
    if (output / "reproducibility_manifest.json").exists():
        return {
            "status": "existing_artifacts_preserved",
            "publication_bundle": str(output),
            "manifest": str(output / "reproducibility_manifest.json"),
            "raw_data_included": False,
        }
    registry = ProtectedResultsRegistry(settings)
    manager = WriteOncePathManager(output.parent, registry.protected_paths())
    manager.ensure_writable_path(output)
    output.mkdir(parents=True)
    report_dir = reports_root(settings) / experiment_id
    artifact_dir = artifacts_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    for source, target in [
        (report_dir / "executive_summary.md", output / "README_publication.md"),
        (report_dir / "aggregate_metrics.csv", output / "aggregate_metrics.csv"),
        (artifact_dir / "diagnostics" / "per_seed_results.csv", output / "per_seed_results.csv"),
        (artifact_dir / "diagnostics" / "hard_subset_results.csv", output / "hard_subset_results.csv"),
        (cert_dir / "certificate_index.md", output / "certificates" / "certificate_index.md"),
        (cert_dir / "certificate_index.json", output / "certificates" / "certificate_index.json"),
        (report_dir / "no_overwrite_audit.md", output / "no_overwrite_audit.md"),
    ]:
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_text(output / "methods.md", "# Methods\n\nMatched-cost and proxy simulations with append-only result preservation.\n")
    write_text(output / "limitations.md", "# Limitations\n\nNo restricted raw data, credentials, model weights, huge tensors, hardware validation, or commercial proof are included.\n")
    write_text(output / "negative_results.md", "# Negative Results\n\nNegative and utility-fragile outcomes are preserved as evidence.\n")
    manifest = {
        "experiment_id": experiment_id,
        "report_dir": str(report_dir),
        "artifact_dir": str(artifact_dir),
        "certificate_dir": str(cert_dir),
        "raw_data_included": False,
        "restricted_data_included": False,
        "files": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
    }
    if (report_dir / "executive_summary.json").exists():
        manifest["summary"] = read_json(report_dir / "executive_summary.json")
    write_json(output / "reproducibility_manifest.json", manifest)
    return {"publication_bundle": str(output), "manifest": str(output / "reproducibility_manifest.json"), "raw_data_included": False}
