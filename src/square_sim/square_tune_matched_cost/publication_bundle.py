from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from square_sim.config import Settings
from square_sim.square_tune_matched_cost.paths import certificates_root, reports_root
from square_sim.utils.files import read_json, write_json, write_text


def create_publication_bundle(settings: Settings, experiment_id: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite publication bundle: {output}")
    report_dir = reports_root(settings) / experiment_id
    cert_dir = certificates_root(settings) / experiment_id
    output.mkdir(parents=True)
    copies = [
        "executive_summary.md",
        "methods.md",
        "aggregate_metrics.csv",
        "aggregate_metrics.parquet",
        "per_seed_results.csv",
        "paired_deltas.csv",
        "bootstrap_intervals.csv",
        "utility_sensitivity_summary.csv",
        "negative_result_report.md",
        "no_overwrite_audit.md",
    ]
    for name in copies:
        src = report_dir / name
        if src.exists():
            shutil.copy2(src, output / name)
    if (cert_dir / "certificate.json").exists():
        shutil.copy2(cert_dir / "certificate.json", output / "certificate.json")
    if (cert_dir / "certificate.md").exists():
        shutil.copy2(cert_dir / "certificate.md", output / "certificate.md")
    write_text(output / "README_publication.md", "# SQUARETune Matched-Cost RAG Kill-Test Bundle\n\nNo raw datasets, credentials, or model weights are included.\n")
    write_text(output / "dataset_and_license_summary.md", "# Dataset And License Summary\n\nSee dataset_report.json in the report root. Raw data is excluded.\n")
    write_text(output / "scenario_card.md", "# Scenario Card\n\nreal_rag_policy_matched_cost.\n")
    write_text(output / "limitations.md", "# Limitations\n\nThis is a real-data-derived policy simulation, not live RAG deployment.\n")
    manifest = {
        "experiment_id": experiment_id,
        "bundle_path": str(output),
        "raw_data_excluded": True,
        "credentials_excluded": True,
        "model_weights_excluded": True,
        "certificate_status": read_json(cert_dir / "certificate.json").get("status") if (cert_dir / "certificate.json").exists() else None,
    }
    write_json(output / "reproducibility_manifest.json", manifest)
    return manifest
