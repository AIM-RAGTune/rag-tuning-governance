from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.square_tune_generalized.config import TRACK_DATASET_KEYS, GeneralizedConfig
from square_sim.square_tune_generalized.scenarios.scenario_cards import scenario_card_text
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash


def _latest_dataset_manifest(dataset_root: Path, track: str) -> Path:
    manifests = sorted((dataset_root / track).glob("*/dataset_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No dataset manifest found for generalized track {track} under {dataset_root}")
    return max(manifests, key=lambda path: str(read_json(path).get("created_at_utc", "")))


def compile_generalized_scenarios(
    config_path: Path,
    *,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    cfg = GeneralizedConfig.from_path(config_path)
    compiled: list[dict[str, Any]] = []
    for track in cfg.tracks:
        manifest_path = _latest_dataset_manifest(dataset_root, track)
        manifest = read_json(manifest_path)
        data = pd.read_parquet(Path(str(manifest["data_path"])))
        for scenario in cfg.scenarios[track]:
            scenario_id = (
                f"{scenario}-{stable_hash({'scenario': scenario, 'manifest': manifest['dataset_version_id'], 'time': datetime.now(timezone.utc).isoformat()}, 12)}"
            )
            out = output_root / track / scenario / scenario_id
            if out.exists():
                raise FileExistsError(f"Refusing to overwrite scenario version: {out}")
            out.mkdir(parents=True)
            frame = data.head(cfg.scenario_max_rows).copy()
            frame["scenario"] = scenario
            frame["scenario_track"] = track
            frame["row_checksum"] = [
                stable_hash({"row_id": row_id, "scenario": scenario}, 16) for row_id in frame["row_id"].astype(str)
            ]
            scenario_path = out / "scenario.parquet"
            frame.to_parquet(scenario_path, index=False)
            train = frame.sample(frac=0.70, random_state=101)
            rest = frame.drop(train.index)
            val = rest.sample(frac=0.50, random_state=202) if len(rest) else rest
            test = rest.drop(val.index)
            split_dir = out / "splits"
            split_dir.mkdir()
            train.to_parquet(split_dir / "train.parquet", index=False)
            val.to_parquet(split_dir / "val.parquet", index=False)
            test.to_parquet(split_dir / "test.parquet", index=False)
            source_counts = frame["source_dataset"].value_counts().to_dict()
            source_datasets = sorted(source_counts)
            license_summary = {
                "source_datasets": source_datasets,
                "license_status": manifest.get("license_status", "unknown"),
                "publication_safe": bool(manifest.get("publication_safe")),
            }
            scenario_manifest = {
                "scenario_id": scenario_id,
                "scenario": scenario,
                "track": track,
                "dataset_key": TRACK_DATASET_KEYS[track],
                "source_dataset_versions": [manifest["dataset_version_id"]],
                "source_datasets": source_datasets,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "row_count": len(frame),
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "scenario_path": str(scenario_path),
                "split_manifest_path": str(split_dir / "split_manifest.json"),
                "license_summary": license_summary,
                "checksum": sha256_file(scenario_path),
                "publication_caveats": [
                    "Generalized benchmark scenario; no hardware validation.",
                    "Patient-flow scenarios are operations proxies only." if track == "patient_flow" else "No clinical or commercial proof is implied.",
                ],
            }
            write_json(out / "scenario_manifest.json", scenario_manifest)
            write_json(out / "source_distribution.json", source_counts)
            write_json(out / "license_summary.json", license_summary)
            write_json(out / "scenario_profile.json", {"columns": list(frame.columns), "row_count": len(frame)})
            write_json(
                split_dir / "split_manifest.json",
                {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test), "split_seed": 101},
            )
            write_text(out / "scenario_card.md", scenario_card_text(track, scenario, source_datasets))
            write_text(out / "checksums.sha256", f"{scenario_manifest['checksum']}  scenario.parquet\n")
            compiled.append(scenario_manifest)
    report = {
        "config_path": str(config_path),
        "scenario_count": len(compiled),
        "scenarios": compiled,
    }
    write_json(output_root / "scenario_compilation_report.json", report)
    write_text(
        output_root / "scenario_compilation_report.md",
        "# Generalized Scenario Compilation\n\n"
        + "\n".join(f"- `{row['track']}/{row['scenario']}` rows={row['row_count']}" for row in compiled)
        + "\n",
    )
    return report
