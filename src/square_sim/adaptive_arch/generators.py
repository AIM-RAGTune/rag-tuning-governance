from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from square_sim.adaptive_arch.config import ARCH_TASKS
from square_sim.adaptive_arch.tasks import TASK_CARDS
from square_sim.tune.synthetic.schemas import FEATURE_COLUMNS
from square_sim.utils.files import write_json, write_text
from square_sim.utils.hashing import stable_hash, write_checksums


def _rng(task: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(int(stable_hash({"task": task, "seed": seed}, 12), 16) % (2**32))


def _split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 91)
    order = rng.permutation(len(df))
    split = np.empty(len(df), dtype=object)
    split[order[: int(0.7 * len(df))]] = "train"
    split[order[int(0.7 * len(df)) : int(0.85 * len(df))]] = "val"
    split[order[int(0.85 * len(df)) :]] = "test"
    df["split"] = split
    return df


def generate_task(
    task: str,
    output: Path,
    *,
    rows: int,
    seed: int,
    noise_level: float = 0.04,
    difficulty: str = "mixed",
) -> dict[str, Any]:
    if task not in ARCH_TASKS:
        raise ValueError(f"Unknown adaptive architecture task: {task}")
    rng = _rng(task, seed)
    df = pd.DataFrame({col: rng.beta(2.0, 2.0, rows).clip(0, 1) for col in FEATURE_COLUMNS})
    df.insert(0, "row_id", [f"{task}-{seed}-{idx:08d}" for idx in range(rows)])
    df["task"] = task
    df["dataset_key"] = f"square_adaptive_arch_{task}"
    df["seed"] = seed
    df["failure_cluster"] = rng.choice(["region_a", "region_b", "region_c", "region_d"], rows)
    df["difficulty"] = rng.choice(["easy", "medium", "hard"] if difficulty == "mixed" else [difficulty], rows)
    df["domain"] = rng.choice(["field_control", "routing", "memory", "rollout"], rows)
    df["input_text"] = [f"Adaptive architecture case {idx} for {task}." for idx in range(rows)]
    df["reference_answer"] = "architecture target"
    df["expected_behavior"] = TASK_CARDS[task]["tests"]
    regime = rng.integers(0, 4, rows)
    hard = df["difficulty"].eq("hard").astype(float)
    local = df["feature_failure_severity"]
    conflict = df["feature_instruction_conflict"]
    risk = df["feature_regression_risk"]
    cost = 0.25 + df["feature_domain_complexity"] + 0.4 * df["feature_tool_need"]
    df["cost_weight"] = np.clip(cost, 0, None)
    df["candidate_response_optional"] = ""
    df["prompt_variant_id"] = rng.integers(0, 5, rows)
    df["retrieval_policy_id"] = rng.integers(0, 6, rows)
    df["adapter_policy_id"] = rng.integers(0, 8, rows)
    df["tool_policy_id"] = rng.integers(0, 5, rows)
    df["safety_label"] = (df["feature_safety_sensitivity"] > 0.65).astype(int)
    signal = 0.35 * df["feature_data_quality"] + 0.25 * local + 0.20 * df["feature_example_novelty"]
    if task == "linear_static_control":
        utility = 0.35 + 0.45 * df["feature_data_quality"] - 0.20 * risk + rng.normal(0, noise_level, rows)
    elif task == "random_unlearnable_control":
        utility = rng.random(rows)
    elif task == "compute_allocation_trap":
        utility = signal + 0.35 * hard * (local > 0.65) - 0.28 * cost + rng.normal(0, noise_level, rows)
    elif task == "merge_required_architecture":
        utility = 0.2 + np.minimum.reduce([df["feature_rag_sensitivity"], df["feature_adapter_sensitivity"], df["feature_prompt_sensitivity"]]) + rng.normal(0, noise_level, rows)
    elif task == "memory_prevents_repeated_failure":
        utility = signal + 0.30 * df["feature_curriculum_sensitivity"] - 0.45 * (risk > 0.72) + rng.normal(0, noise_level, rows)
    elif task == "dynamic_topology_routing":
        utility = signal + 0.35 * df["feature_tool_need"] * (regime % 2) - 0.2 * conflict + rng.normal(0, noise_level, rows)
    elif task == "nonlinear_extrapolation_required":
        utility = 0.2 + 0.9 * df["feature_rag_sensitivity"] * (1 - df["feature_rag_sensitivity"]) - 0.25 * risk + rng.normal(0, noise_level, rows)
    elif task == "protect_known_good_while_adapting":
        utility = signal - 0.55 * df["feature_safety_sensitivity"] * risk + rng.normal(0, noise_level, rows)
    else:
        utility = signal + 0.25 * (regime == 2) - 0.18 * conflict + rng.normal(0, noise_level, rows)
    df["target_utility"] = np.clip(utility, 0, 1)
    df["target_branch_success"] = ((local + conflict + hard) > 1.45).astype(int)
    df["target_merge_success"] = ((df["feature_rag_sensitivity"] + df["feature_adapter_sensitivity"] + df["feature_prompt_sensitivity"]) > 1.6).astype(int)
    df["target_regression"] = ((risk + df["feature_safety_sensitivity"]) > 1.35).astype(int)
    df["target_improvement"] = df["target_utility"] - 0.35
    df["hard_subset"] = hard.astype(int)
    df["protected_region"] = (df["feature_safety_sensitivity"] > 0.68).astype(int)
    df["regression_group"] = np.where(df["protected_region"].eq(1), "protected", "ordinary")
    df["latent_regime"] = regime
    df["latent_topology_blocked"] = (rng.random(rows) < 0.12 + 0.25 * conflict).astype(int)
    df["ground_truth_score_vector_json"] = [
        json.dumps({"utility": float(u), "regression_risk": float(r), "cost": float(c)}, sort_keys=True)
        for u, r, c in zip(df["target_utility"], risk, cost, strict=True)
    ]
    df = _split(df, seed)
    version = f"{task}-{seed}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{stable_hash({'task': task, 'seed': seed, 'rows': rows}, 12)}"
    root = output / task / version
    if root.exists():
        root = output / task / f"{version}-{stable_hash({'collision': str(root)}, 6)}"
    root.mkdir(parents=True, exist_ok=False)
    df.to_parquet(root / "data.parquet", index=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_parquet(root / f"{split}.parquet", index=False)
    manifest = {
        "generator_name": "square_adaptive_arch_v1",
        "generator_version": "1.0",
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "row_count": rows,
        "feature_count": len(FEATURE_COLUMNS),
        "latent_columns": ["latent_regime", "latent_topology_blocked"],
        "observed_columns": FEATURE_COLUMNS,
        "label_columns": ["target_utility", "target_branch_success", "target_merge_success", "target_regression"],
        "mechanism_name": task,
        "expected_winner": TASK_CARDS[task]["expected_winner"],
        "expected_failing_ablations": TASK_CARDS[task]["failing_ablations"],
        "noise_level": noise_level,
        "difficulty": difficulty,
        "caveats": ["Synthetic mechanism benchmark; not physical hardware validation."],
    }
    write_json(root / "generator_manifest.json", manifest)
    write_json(root / "schema.json", {"columns": list(df.columns), "row_count": len(df)})
    write_json(root / "profile.json", {"row_count": len(df), "task": task, "seed": seed})
    write_json(root / "expected_outcomes.json", TASK_CARDS[task])
    write_text(
        root / "mechanism_card.md",
        "\n".join(
            [
                f"# {task}",
                "",
                f"Tests: {TASK_CARDS[task]['tests']}",
                "",
                f"Expected winner: `{TASK_CARDS[task]['expected_winner']}`",
                "",
                "This is a synthetic adaptive-architecture mechanism diagnostic.",
            ]
        )
        + "\n",
    )
    checksums = write_checksums([root / "data.parquet", root / "generator_manifest.json"], root / "checksums.sha256")
    manifest["checksum"] = checksums.get(str(root / "data.parquet"))
    write_json(root / "generator_manifest.json", manifest)
    return {"task": task, "seed": seed, "path": str(root), "row_count": rows}


def generate_suite(
    output: Path,
    *,
    suite: str = "square_adaptive_arch_v1",
    rows: int = 50_000,
    seeds: list[int] | None = None,
    tasks: list[str] | None = None,
    noise_level: float = 0.04,
    difficulty: str = "mixed",
) -> dict[str, Any]:
    if suite != "square_adaptive_arch_v1":
        raise ValueError(f"Unknown adaptive architecture suite: {suite}")
    seeds = seeds or [101, 202, 303, 404, 505]
    tasks = tasks or ARCH_TASKS
    generated = [
        generate_task(task, output, rows=rows, seed=seed, noise_level=noise_level, difficulty=difficulty)
        for task in tasks
        for seed in seeds
    ]
    write_json(output / "generation_report.json", {"suite": suite, "generated": generated})
    write_text(output / "generation_report.md", f"# {suite} Generation\n\nGenerated `{len(generated)}` task versions.\n")
    return {"suite": suite, "generated_count": len(generated), "output": str(output), "generated": generated}


def validate_suite(root: Path) -> dict[str, Any]:
    manifests = sorted(root.glob("*/*/generator_manifest.json"))
    failures = []
    for manifest in manifests:
        base = manifest.parent
        for name in ["data.parquet", "train.parquet", "val.parquet", "test.parquet", "mechanism_card.md", "expected_outcomes.json"]:
            if not (base / name).exists():
                failures.append(f"{base}: missing {name}")
    return {"root": str(root), "dataset_versions": len(manifests), "failed": len(failures), "failures": failures}
