from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from square_sim.tune.config import TUNE_DATASETS
from square_sim.tune.synthetic.manifests import generator_manifest
from square_sim.tune.synthetic.mechanism_cards import MECHANISMS, mechanism_card
from square_sim.tune.synthetic.schemas import FEATURE_COLUMNS, LATENT_COLUMNS, REQUIRED_COLUMNS
from square_sim.tune.synthetic.validators import validate_suite
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, write_checksums

FAILURE_CLUSTERS = [
    "hallucination_cluster",
    "retrieval_miss_cluster",
    "tone_mismatch_cluster",
    "tool_routing_cluster",
    "safety_boundary_cluster",
    "regression_cluster",
    "domain_confusion_cluster",
]
DOMAINS = ["finance", "clinical_ops", "manufacturing", "support", "legal_ops"]
DIFFICULTIES = ["easy", "medium", "hard"]


def _rng_for(dataset_key: str, seed: int) -> np.random.Generator:
    import hashlib

    digest = hashlib.sha256(f"{dataset_key}:{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16) % (2**32))


def _base_frame(dataset_key: str, rows: int, seed: int, difficulty: str) -> pd.DataFrame:
    rng = _rng_for(dataset_key, seed)
    clusters = rng.choice(FAILURE_CLUSTERS, rows)
    domains = rng.choice(DOMAINS, rows)
    diff = rng.choice(DIFFICULTIES if difficulty == "mixed" else [difficulty], rows)
    feature_data = {
        col: np.clip(rng.beta(2.0, 2.0, rows), 0.0, 1.0) for col in FEATURE_COLUMNS
    }
    df = pd.DataFrame(feature_data)
    df.insert(0, "row_id", [f"{dataset_key}-{seed}-{i:08d}" for i in range(rows)])
    df["dataset_key"] = dataset_key
    df["mechanism_name"] = MECHANISMS[dataset_key].mechanism_name
    df["seed"] = seed
    df["task_type"] = "llm_adaptation_sim"
    df["failure_cluster"] = clusters
    df["difficulty"] = diff
    df["domain"] = domains
    df["input_text"] = [
        f"Synthetic enterprise adaptation case {i} for {cluster} in {domain}."
        for i, (cluster, domain) in enumerate(zip(clusters, domains, strict=True))
    ]
    df["reference_answer"] = "synthetic reference behavior"
    df["candidate_response_optional"] = ""
    df["expected_behavior"] = "improve target eval without protected regression"
    df["prompt_variant_id"] = rng.integers(0, 5, rows)
    df["retrieval_policy_id"] = rng.integers(0, 6, rows)
    df["adapter_policy_id"] = rng.integers(0, 8, rows)
    df["tool_policy_id"] = rng.integers(0, 5, rows)
    df["safety_label"] = (df["feature_safety_sensitivity"] > 0.65).astype(int)
    df["regression_group"] = np.where(df["feature_regression_risk"] > 0.65, "protected", "ordinary")
    df["cost_weight"] = 0.2 + df["feature_domain_complexity"] + 0.5 * df["feature_tool_need"]
    df["latent_cluster_optimum"] = rng.normal(0.5, 0.12, rows).clip(0, 1)
    df["latent_poison_flag"] = (rng.random(rows) < (0.02 + 0.10 * df["feature_regression_risk"])).astype(int)
    df["latent_order_effect"] = rng.normal(0, 0.25, rows)
    df["latent_merge_group"] = rng.integers(0, 4, rows)
    return df


def _score_vector(row: pd.Series) -> dict[str, float]:
    return {
        "domain_accuracy": float(0.45 + 0.30 * row.feature_data_quality - 0.10 * row.feature_domain_complexity),
        "retrieval_faithfulness": float(0.45 + 0.25 * (1 - row.feature_retrieval_ambiguity)),
        "instruction_following": float(0.50 + 0.20 * row.feature_style_specificity),
        "style_match": float(0.45 + 0.25 * row.feature_style_specificity),
        "safety": float(0.70 - 0.25 * row.feature_safety_sensitivity),
        "latency": float(0.20 + 0.50 * row.cost_weight),
        "cost": float(row.cost_weight),
        "calibration": float(0.45 + 0.15 * (1 - row.feature_instruction_conflict)),
        "regression_score": float(0.85 - 0.50 * row.feature_regression_risk),
    }


def _apply_mechanism(df: pd.DataFrame, dataset_key: str, noise_level: float) -> pd.DataFrame:
    rng = _rng_for(dataset_key + ":targets", int(df["seed"].iloc[0]))
    q = df["feature_data_quality"]
    sev = df["feature_failure_severity"]
    reg = df["feature_regression_risk"]
    rag = df["feature_rag_sensitivity"]
    adapter = df["feature_adapter_sensitivity"]
    prompt = df["feature_prompt_sensitivity"]
    tool = df["feature_tool_need"]
    novelty = df["feature_example_novelty"]
    noise = rng.normal(0.0, noise_level, len(df))

    if dataset_key == "synthetic_llm_linear_control":
        utility = 0.30 + 0.50 * q + 0.20 * novelty - 0.25 * reg + 0.10 * (1 - df["feature_domain_complexity"]) + noise
        branch = utility > np.quantile(utility, 0.55)
        merge = branch
    elif dataset_key == "synthetic_llm_random_label":
        utility = rng.random(len(df))
        branch = rng.random(len(df)) > 0.5
        merge = rng.random(len(df)) > 0.5
    elif dataset_key == "synthetic_llm_failure_cluster_routing":
        cluster_bonus = df["failure_cluster"].map({c: i / len(FAILURE_CLUSTERS) for i, c in enumerate(FAILURE_CLUSTERS)})
        utility = 0.20 + 0.30 * sev + 0.30 * q + 0.25 * cluster_bonus - 0.15 * reg + noise
        branch = (sev + cluster_bonus) > 0.8
        merge = branch & (reg < 0.65)
    elif dataset_key == "synthetic_llm_nonmonotonic_data_mix":
        mix = df["latent_cluster_optimum"]
        utility = 0.15 + 0.80 * mix * (1 - mix) + 0.20 * q - 0.35 * reg**2 + noise
        branch = utility > 0.48
        merge = branch & (reg < 0.7)
    elif dataset_key == "synthetic_llm_adapter_tradeoff":
        utility = 0.20 + 0.35 * adapter + 0.25 * q - 0.25 * reg * adapter + noise
        branch = (adapter > 0.45) & (reg < 0.75)
        merge = branch & (df["cost_weight"] < 1.2)
    elif dataset_key == "synthetic_llm_rag_policy_conflict":
        utility = 0.20 + 0.45 * rag * (1 - df["feature_retrieval_ambiguity"]) - 0.20 * df["cost_weight"] + noise
        branch = utility > 0.35
        merge = branch & (df["feature_retrieval_ambiguity"] < 0.75)
    elif dataset_key == "synthetic_llm_prompt_regression":
        pressure = prompt * (1 - df["feature_instruction_conflict"])
        utility = 0.15 + 0.45 * pressure - 0.30 * df["feature_safety_sensitivity"] * prompt + noise
        branch = pressure > 0.35
        merge = branch & (df["feature_safety_sensitivity"] < 0.65)
    elif dataset_key == "synthetic_llm_tool_routing":
        utility = 0.15 + 0.55 * tool * (1 - df["feature_instruction_conflict"]) - 0.10 * reg + noise
        branch = utility > 0.38
        merge = branch & (df["feature_tool_need"] > 0.35)
    elif dataset_key == "synthetic_llm_data_poison_regression":
        local_gain = 0.45 * q + 0.25 * novelty
        poison = df["latent_poison_flag"]
        utility = 0.20 + local_gain - 0.55 * poison - 0.25 * reg + noise
        branch = local_gain > 0.45
        merge = branch & (poison == 0) & (reg < 0.7)
    elif dataset_key == "synthetic_llm_merge_required":
        group = df["latent_merge_group"]
        complementary = np.minimum.reduce([q + 0.15, rag + 0.15, adapter + 0.15, prompt + 0.15])
        utility = 0.12 + 0.55 * complementary - 0.20 * reg + noise
        branch = utility > 0.45
        merge = branch & (group.isin([1, 2]) | ((rag + adapter + prompt) > 1.85))
    elif dataset_key == "synthetic_llm_curriculum_order":
        order = df["latent_order_effect"]
        utility = 0.20 + 0.30 * q + 0.28 * df["feature_curriculum_sensitivity"] + 0.20 * order - 0.15 * reg + noise
        branch = utility > 0.45
        merge = branch & (df["feature_curriculum_sensitivity"] > 0.35)
    elif dataset_key == "synthetic_llm_hard_external_transfer_proxy":
        shift = np.where(df["domain"].isin(["legal_ops", "clinical_ops"]), 0.20, -0.08)
        utility = 0.22 + 0.35 * q - 0.30 * reg - shift * df["feature_domain_complexity"] + noise
        branch = utility > 0.38
        merge = branch & (reg < 0.6)
    elif dataset_key == "synthetic_llm_repeated_regression_memory":
        memory_signal = 0.45 * df["feature_curriculum_sensitivity"] + 0.35 * (1 - reg)
        utility = 0.20 + memory_signal + 0.10 * q - 0.35 * df["latent_poison_flag"] + noise
        branch = utility > 0.45
        merge = branch & (df["latent_poison_flag"] == 0)
    elif dataset_key == "synthetic_llm_regression_veto":
        raw_gain = 0.55 * q + 0.25 * adapter
        protected_penalty = 0.60 * df["feature_safety_sensitivity"] + 0.30 * reg
        utility = 0.25 + raw_gain - protected_penalty + noise
        branch = raw_gain > 0.55
        merge = branch & (protected_penalty < 0.55)
    elif dataset_key == "synthetic_llm_cost_tradeoff":
        expensive_gain = 0.60 * adapter + 0.35 * rag
        utility = 0.22 + expensive_gain - 0.45 * df["cost_weight"] + 0.15 * q + noise
        branch = expensive_gain > 0.55
        merge = branch & (df["cost_weight"] < 1.05)
    else:
        raise ValueError(f"Unknown SQUARETune synthetic dataset: {dataset_key}")

    df["target_utility"] = np.clip(utility, 0.0, 1.0)
    df["target_improvement"] = np.clip(df["target_utility"] - 0.35, -1.0, 1.0)
    df["target_regression"] = (reg + 0.4 * df["latent_poison_flag"] > 0.72).astype(int)
    df["target_branch_success"] = branch.astype(int)
    df["target_merge_success"] = merge.astype(int)
    df["ground_truth_score_vector_json"] = [json.dumps(_score_vector(row), sort_keys=True) for _, row in df.iterrows()]
    return df


def _assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(len(df))
    split = np.empty(len(df), dtype=object)
    train_end = int(0.70 * len(df))
    val_end = int(0.85 * len(df))
    split[order[:train_end]] = "train"
    split[order[train_end:val_end]] = "val"
    split[order[val_end:]] = "test"
    df["split"] = split
    return df


def _schema(df: pd.DataFrame, dataset_key: str) -> dict[str, Any]:
    return {
        "dataset_key": dataset_key,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "role": "feature" if col in FEATURE_COLUMNS else "target" if col.startswith("target_") else "metadata",
            }
            for col in df.columns
        ],
        "warnings": [],
    }


def _profile(df: pd.DataFrame, dataset_key: str) -> dict[str, Any]:
    targets = ["target_utility", "target_improvement", "target_regression", "target_branch_success", "target_merge_success"]
    return {
        "dataset_key": dataset_key,
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_summary": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in targets
        },
        "failure_cluster_counts": df["failure_cluster"].value_counts().to_dict(),
        "domain_counts": df["domain"].value_counts().to_dict(),
    }


def _expected_outcomes(dataset_key: str) -> dict[str, Any]:
    spec = MECHANISMS[dataset_key]
    return {
        "dataset_key": dataset_key,
        "control_type": spec.control_type,
        "intended_winner": spec.expected_winner,
        "intended_failing_ablations": spec.expected_losing_ablations,
        "certificate_rules": {
            "random_label": "refuse any advantage-like claim",
            "linear_control": "classical sanity should pass",
            "mechanism": "full must beat relevant ablations across seeds",
        },
    }


def generate_dataset(
    dataset_key: str,
    output_root: Path,
    *,
    rows: int,
    seed: int,
    noise_level: float = 0.05,
    difficulty: str = "mixed",
    write_latent: bool = False,
) -> dict[str, Any]:
    if dataset_key not in MECHANISMS:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    dataset_dir = output_root / dataset_key / f"seed_{seed}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df = _base_frame(dataset_key, rows, seed, difficulty)
    df = _apply_mechanism(df, dataset_key, noise_level)
    df = _assign_splits(df, seed)
    if not write_latent:
        df = df.drop(columns=[col for col in LATENT_COLUMNS if col in df.columns])
    ordered = [col for col in REQUIRED_COLUMNS + FEATURE_COLUMNS if col in df.columns]
    df = df[ordered + [col for col in df.columns if col not in ordered]]

    data_path = dataset_dir / "data.parquet"
    df.to_parquet(data_path, index=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_parquet(dataset_dir / f"{split}.parquet", index=False)
    write_json(dataset_dir / "schema.json", _schema(df, dataset_key))
    write_json(dataset_dir / "profile.json", _profile(df, dataset_key))
    spec = MECHANISMS[dataset_key]
    write_text(dataset_dir / "mechanism_card.md", mechanism_card(spec))
    write_json(dataset_dir / "expected_outcomes.json", _expected_outcomes(dataset_key))
    manifest = generator_manifest(
        dataset_key=dataset_key,
        mechanism_name=spec.mechanism_name,
        seed=seed,
        rows=len(df),
        feature_count=len(FEATURE_COLUMNS),
        observed_feature_columns=FEATURE_COLUMNS,
        label_columns=[
            "target_improvement",
            "target_regression",
            "target_utility",
            "target_branch_success",
            "target_merge_success",
        ],
        noise_level=noise_level,
        intended_winner=spec.expected_winner,
        intended_failing_ablations=spec.expected_losing_ablations,
        data_path=data_path,
    )
    write_json(dataset_dir / "generator_manifest.json", manifest)
    checksum_paths = [
        dataset_dir / name
        for name in [
            "data.parquet",
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "schema.json",
            "profile.json",
            "generator_manifest.json",
            "expected_outcomes.json",
            "mechanism_card.md",
        ]
    ]
    checksums = write_checksums(checksum_paths, dataset_dir / "checksums.sha256")
    return {
        "dataset_key": dataset_key,
        "seed": seed,
        "path": str(dataset_dir),
        "rows": len(df),
        "data_checksum": sha256_file(data_path),
        "checksums": checksums,
    }


def generate_suite(
    output_root: Path,
    *,
    suite: str = "llm_tuning_v1",
    rows: int = 50_000,
    seeds: list[int] | None = None,
    noise_level: float = 0.05,
    difficulty: str = "mixed",
    write_latent: bool = False,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    if suite != "llm_tuning_v1":
        raise ValueError(f"Unsupported suite: {suite}")
    seeds = seeds or [101, 202, 303, 404, 505]
    dataset_names = datasets or TUNE_DATASETS
    results = [
        generate_dataset(
            dataset_key,
            output_root,
            rows=rows,
            seed=seed,
            noise_level=noise_level,
            difficulty=difficulty,
            write_latent=write_latent,
        )
        for dataset_key in dataset_names
        for seed in seeds
    ]
    report_dir = output_root.parent.parent.parent / "reports" / "square_tune" / "datasets" / suite
    if "datasets" not in output_root.parts:
        report_dir = output_root / "reports" / suite
    report = "# SQUARETune Synthetic Generation Report\n\n"
    report += f"Suite: `{suite}`\n\nRows per dataset/seed: `{rows}`\n\nSeeds: `{seeds}`\n\n"
    report += "| Dataset | Seed | Rows | Path |\n|---|---:|---:|---|\n"
    for row in results:
        report += f"| {row['dataset_key']} | {row['seed']} | {row['rows']} | `{row['path']}` |\n"
    write_text(report_dir / "generation_report.md", report)
    write_json(report_dir / "generation_report.json", {"suite": suite, "rows": rows, "seeds": seeds, "results": results})
    return {"suite": suite, "output_root": str(output_root), "generated": len(results), "report_path": str(report_dir / "generation_report.md"), "results": results}


def list_generated_datasets(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/seed_*/generator_manifest.json")):
        payload = read_json(manifest_path)
        rows.append(
            {
                "dataset_key": payload.get("generator_name"),
                "seed": payload.get("seed"),
                "dataset_version_id": payload.get("dataset_version_id"),
                "path": str(manifest_path.parent),
                "intended_winner": payload.get("intended_winner"),
            }
        )
    return rows


def describe_mechanism(dataset_key: str) -> dict[str, Any]:
    spec = MECHANISMS[dataset_key]
    return {
        "dataset_key": spec.key,
        "mechanism_name": spec.mechanism_name,
        "purpose": spec.purpose,
        "ground_truth": spec.ground_truth,
        "expected_winner": spec.expected_winner,
        "expected_losing_ablations": spec.expected_losing_ablations,
        "control_type": spec.control_type,
    }


__all__ = [
    "describe_mechanism",
    "generate_dataset",
    "generate_suite",
    "list_generated_datasets",
    "validate_suite",
]
