from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from square_sim.config import Settings
from square_sim.tune.external.paths import external_root
from square_sim.tune.external.schemas import SCENARIO_FEATURE_COLUMNS
from square_sim.utils.files import read_json, write_json, write_text
from square_sim.utils.hashing import sha256_file, stable_hash, write_checksums

FAILURE_BY_FAMILY = {
    "rag_policy_optimization": [
        "missing_retrieval",
        "unsupported_claim",
        "answer_incomplete",
        "over_retrieval_latency",
        "citation_mismatch",
    ],
    "hallucination_faithfulness_reduction": [
        "unsupported_claim",
        "citation_mismatch",
        "over_refusal",
        "answer_incomplete",
    ],
    "prompt_regression_optimization": [
        "format_violation",
        "instruction_omission",
        "over_constrained_response",
        "reasoning_degradation",
    ],
    "data_curation_preference_optimization": [
        "low_quality_examples",
        "too_easy_examples",
        "duplicated_examples",
        "harmful_examples",
        "missing_domain_examples",
    ],
    "adapter_planning_simulation": [
        "adapter_overfit",
        "rank_cost",
        "missing_domain_examples",
        "regression_cluster",
    ],
    "tool_routing_policy_optimization": [
        "unnecessary_tool_call",
        "missing_tool_call",
        "wrong_tool",
        "wrong_arguments",
        "tool_policy_violation",
    ],
}

SOURCE_MAPPING = {
    "rag_policy_optimization": {
        "primary": {"ragbench", "hagrid", "expertqa"},
        "secondary": {"ragtruth"},
        "exclude": {"ifeval", "helpsteer2", "dolly15k", "bfcl", "ultrafeedback", "oasst1"},
    },
    "hallucination_faithfulness_reduction": {
        "primary": {"ragtruth", "ragbench", "hagrid"},
        "secondary": {"expertqa"},
        "exclude": {"ifeval", "dolly15k", "bfcl", "helpsteer2", "ultrafeedback", "oasst1"},
    },
    "prompt_regression_optimization": {
        "primary": {"ifeval", "promptbench", "bbh_subset"},
        "secondary": {"dolly15k"},
        "exclude": {"ragbench", "hagrid", "expertqa", "bfcl", "helpsteer2", "ultrafeedback", "oasst1"},
    },
    "data_curation_preference_optimization": {
        "primary": {"helpsteer2", "ultrafeedback", "dolly15k", "oasst1"},
        "secondary": set(),
        "exclude": {"bfcl", "ragbench", "hagrid", "expertqa", "ifeval"},
    },
    "adapter_planning_simulation": {
        "primary": {"helpsteer2", "ultrafeedback", "dolly15k", "oasst1"},
        "secondary": {"ifeval"},
        "exclude": {"bfcl", "ragbench", "hagrid", "expertqa"},
    },
    "tool_routing_policy_optimization": {
        "primary": {"bfcl", "api_bank", "tau_bench"},
        "secondary": set(),
        "exclude": {"ragbench", "hagrid", "expertqa", "ifeval", "helpsteer2", "dolly15k", "ultrafeedback", "oasst1"},
    },
}


def _load_external_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw["scenario_root"] = (
        str(Path(os.path.expandvars(str(raw.get("scenario_root", "")))).expanduser())
        if raw.get("scenario_root")
        else None
    )
    return raw


def _feature_frame(df: pd.DataFrame, family: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    text_len = (
        out["input_text"].fillna("").str.len()
        + out["context_text"].fillna("").str.len()
        + out["candidate_response"].fillna("").str.len()
    ).clip(lower=1)
    norm_len = (text_len / max(float(text_len.quantile(0.95)), 1.0)).clip(0, 1)
    label = pd.to_numeric(out["label"], errors="coerce").fillna(0.5).clip(0, 1)
    family_bias = {
        "rag_policy_optimization": (0.75, 0.35, 0.75, 0.20),
        "hallucination_faithfulness_reduction": (0.85, 0.45, 0.70, 0.30),
        "prompt_regression_optimization": (0.35, 0.75, 0.25, 0.25),
        "data_curation_preference_optimization": (0.25, 0.35, 0.35, 0.80),
        "adapter_planning_simulation": (0.30, 0.40, 0.40, 0.75),
        "tool_routing_policy_optimization": (0.20, 0.45, 0.30, 0.30),
    }.get(family, (0.4, 0.4, 0.4, 0.4))
    rag_bias, prompt_bias, tool_bias, adapter_bias = family_bias
    out["feature_domain_complexity"] = norm_len
    out["feature_failure_severity"] = (1 - label + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_retrieval_ambiguity"] = (rag_bias * norm_len + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_instruction_conflict"] = (prompt_bias * (1 - label) + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_safety_sensitivity"] = (0.25 + 0.45 * (1 - label) + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_style_specificity"] = (0.25 + prompt_bias * label + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_tool_need"] = (tool_bias * norm_len + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_data_quality"] = label.clip(0, 1)
    out["feature_example_novelty"] = rng.beta(2, 2, len(out))
    out["feature_duplication_risk"] = rng.beta(1.5, 4, len(out))
    out["feature_regression_risk"] = (0.2 + 0.55 * (1 - label) + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_adapter_sensitivity"] = (adapter_bias * label + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_prompt_sensitivity"] = (prompt_bias * label + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_rag_sensitivity"] = (rag_bias * label + 0.1 * rng.random(len(out))).clip(0, 1)
    out["feature_curriculum_sensitivity"] = rng.beta(2, 2, len(out))
    clusters = FAILURE_BY_FAMILY.get(family, ["external_cluster"])
    out["failure_cluster"] = [clusters[i % len(clusters)] for i in range(len(out))]
    out["cost_weight"] = (0.2 + norm_len + 0.4 * out["feature_tool_need"]).clip(0, 2)
    out["target_utility"] = label
    out["target_improvement"] = (label - 0.35).clip(-1, 1)
    out["target_regression"] = (out["feature_regression_risk"] > 0.72).astype(int)
    out["target_branch_success"] = (label > 0.55).astype(int)
    out["target_merge_success"] = ((label > 0.50) & (out["feature_regression_risk"] < 0.65)).astype(int)
    return out


def _split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    split = np.empty(len(df), dtype=object)
    train_end = int(0.70 * len(df))
    val_end = int(0.85 * len(df))
    split[order[:train_end]] = "train"
    split[order[train_end:val_end]] = "val"
    split[order[val_end:]] = "test"
    df = df.copy()
    df["split"] = split
    return df


def _source_allowed(dataset_key: str, family: str, *, source_appropriate: bool) -> bool:
    if not source_appropriate:
        return True
    mapping = SOURCE_MAPPING.get(family)
    if not mapping:
        return True
    if dataset_key in mapping["exclude"]:
        return False
    return dataset_key in mapping["primary"] or dataset_key in mapping["secondary"]


def compile_scenarios(
    config_path: Path,
    *,
    output_root: Path | None = None,
    seed: int = 101,
    source_appropriate: bool | None = None,
) -> dict[str, Any]:
    cfg = _load_external_config(config_path)
    scenario_root = output_root or Path(str(cfg.get("scenario_root") or "scenarios")).expanduser()
    source_appropriate = bool(cfg.get("source_appropriate", False)) if source_appropriate is None else source_appropriate
    families = list(cfg.get("scenario_families", []))
    if not families:
        families = ["external_transfer"]
    normalized_root = scenario_root.parent / "normalized"
    if not normalized_root.exists():
        # Common layout: scenarios/ is sibling of normalized/.
        normalized_root = external_root(Settings.from_env()) / "normalized"
    manifests = sorted(normalized_root.glob("*/*/external_dataset_manifest.json"))
    latest_by_dataset: dict[str, Path] = {}
    for manifest_path in manifests:
        dataset_key = manifest_path.parent.parent.name
        current = latest_by_dataset.get(dataset_key)
        if current is None or manifest_path.stat().st_mtime > current.stat().st_mtime:
            latest_by_dataset[dataset_key] = manifest_path
    manifests = sorted(latest_by_dataset.values())
    results: list[dict[str, Any]] = []
    normalized_rows = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        parquet = Path(str(manifest["normalized_path"]))
        if parquet.exists():
            normalized_rows.append((manifest, pd.read_parquet(parquet)))
    if not normalized_rows:
        return {"status": "skipped", "reason": "No normalized external datasets found.", "results": []}
    for family in families:
        frames = []
        source_versions = []
        for manifest, frame in normalized_rows:
            dataset_key = str(manifest.get("dataset_key"))
            source_families = set(frame["scenario_family"].dropna().unique())
            family_matches = (
                _source_allowed(dataset_key, family, source_appropriate=True)
                if source_appropriate
                else family in source_families
            )
            if family_matches and _source_allowed(dataset_key, family, source_appropriate=source_appropriate):
                frames.append(frame.copy())
                source_versions.append(manifest["dataset_version_id"])
        if not frames:
            continue
        scenario = pd.concat(frames, ignore_index=True).drop_duplicates("row_id")
        scenario["scenario_family"] = family
        scenario = _feature_frame(scenario, family, seed)
        scenario = _split(scenario, seed)
        generated_at = datetime.now(timezone.utc).isoformat()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        scenario_id = (
            f"{family}_{timestamp}-"
            f"{stable_hash({'family': family, 'sources': source_versions, 'rows': len(scenario), 'generated_at': generated_at}, 12)}"
        )
        out_dir = scenario_root / family / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)
        scenario_path = out_dir / "scenario.parquet"
        scenario.to_parquet(scenario_path, index=False)
        split_dir = out_dir / "splits"
        split_dir.mkdir(exist_ok=True)
        for split_name in ["train", "val", "test"]:
            scenario[scenario["split"] == split_name].to_parquet(split_dir / f"{split_name}.parquet", index=False)
        license_summary = scenario["license_status"].value_counts().to_dict()
        source_distribution = scenario["source_dataset"].value_counts().to_dict()
        source_distribution_percent = {
            key: value / max(1, len(scenario)) for key, value in source_distribution.items()
        }
        mapping = SOURCE_MAPPING.get(family, {"primary": set(), "secondary": set(), "exclude": set()})
        sources = set(source_distribution)
        warnings = []
        if source_appropriate and not (sources & mapping["primary"]):
            warnings.append(f"missing primary source dataset for {family}")
        if sources & mapping["exclude"]:
            warnings.append(f"excluded source datasets present: {sorted(sources & mapping['exclude'])}")
        manifest = {
            "scenario_id": scenario_id,
            "scenario_family": family,
            "source_datasets": sorted(scenario["source_dataset"].dropna().unique().tolist()),
            "source_dataset_versions": source_versions,
            "generated_at_utc": generated_at,
            "split_seed": seed,
            "row_count": len(scenario),
            "train_count": int((scenario["split"] == "train").sum()),
            "val_count": int((scenario["split"] == "val").sum()),
            "test_count": int((scenario["split"] == "test").sum()),
            "label_definitions": ["external proxy label normalized to 0..1"],
            "objective_definitions": ["cost-adjusted external-transfer simulation utility"],
            "available_metrics": [
                "final_utility",
                "cost_adjusted_improvement",
                "regression_count",
                "rag_faithfulness_proxy",
                "tool_policy_accuracy_proxy",
            ],
            "budget_assumptions": cfg.get("square_tune", {}),
            "license_summary": license_summary,
            "source_appropriate": source_appropriate,
            "source_distribution": source_distribution,
            "source_distribution_percent": source_distribution_percent,
            "source_mapping": {
                "primary": sorted(mapping["primary"]),
                "secondary": sorted(mapping["secondary"]),
                "exclude": sorted(mapping["exclude"]),
            },
            "warnings": warnings,
            "checksum": sha256_file(scenario_path),
        }
        write_json(out_dir / "scenario_manifest.json", manifest)
        write_json(out_dir / "scenario_license_summary.json", license_summary)
        write_json(out_dir / "source_distribution.json", source_distribution)
        write_json(out_dir / "scenario_profile.json", {"row_count": len(scenario), "feature_columns": SCENARIO_FEATURE_COLUMNS})
        write_json(
            split_dir / "split_manifest.json",
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "train": manifest["train_count"],
                "val": manifest["val_count"],
                "test": manifest["test_count"],
            },
        )
        write_text(
            out_dir / "scenario_card.md",
            "\n".join(
                [
                    f"# Scenario Card: {family}",
                    "",
                    "This is an external-transfer simulation over open/external examples.",
                    "It does not prove SQUARE hardware or commercial ROI.",
                    f"Source datasets: `{manifest['source_datasets']}`",
                    f"Source distribution: `{source_distribution}`",
                    f"Source-appropriate mapping: `{source_appropriate}`",
                    f"Rows: `{len(scenario)}`",
                    f"License summary: `{license_summary}`",
                    f"Warnings: `{warnings}`",
                ]
            )
            + "\n",
        )
        write_checksums([scenario_path, out_dir / "scenario_manifest.json"], out_dir / "checksums.sha256")
        results.append(manifest)
    project_root = scenario_root.parent.parent.parent.parent
    report_dir = project_root / "reports" / "square_tune" / "external_transfer" / "scenario_compilation"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "completed", "scenario_root": str(scenario_root), "results": results}
    write_json(report_dir / "scenario_compilation_report.json", payload)
    lines = ["# SQUARETune External Scenario Compilation Report", "", "| Scenario | Rows | Sources |", "|---|---:|---|"]
    for row in results:
        lines.append(f"| {row['scenario_family']} | {row['row_count']} | `{row['source_datasets']}` |")
    write_text(report_dir / "scenario_compilation_report.md", "\n".join(lines) + "\n")
    return payload
