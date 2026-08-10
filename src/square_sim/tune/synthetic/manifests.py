from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from square_sim.utils.hashing import sha256_file, stable_hash


def git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def generator_manifest(
    *,
    dataset_key: str,
    mechanism_name: str,
    seed: int,
    rows: int,
    feature_count: int,
    observed_feature_columns: list[str],
    label_columns: list[str],
    noise_level: float,
    intended_winner: str,
    intended_failing_ablations: list[str],
    data_path: Path,
) -> dict[str, Any]:
    return {
        "generator_name": dataset_key,
        "generator_version": "llm_tuning_v1",
        "git_commit": git_commit(),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "heldout_seed_group": "final_eval" if seed >= 100 else "development",
        "row_count": rows,
        "feature_count": feature_count,
        "observed_feature_columns": observed_feature_columns,
        "latent_columns": [],
        "label_columns": label_columns,
        "mechanism_name": mechanism_name,
        "label_formula_summary": "Deterministic seeded response-surface diagnostic; see mechanism_card.md.",
        "full_label_formula_machine_readable_if_practical": None,
        "noise_level": noise_level,
        "train_val_test_split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "checksum": sha256_file(data_path),
        "dataset_version_id": f"{dataset_key}-{stable_hash({'path': str(data_path), 'sha': sha256_file(data_path)}, 12)}",
        "intended_winner": intended_winner,
        "intended_failing_ablations": intended_failing_ablations,
        "scientific_caveats": [
            "Synthetic mechanism diagnostic only.",
            "Not physical hardware validation.",
            "Not an external LLM benchmark result.",
        ],
    }
