from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

from ragtune.external_evaluator_adapter_demo import run_external_evaluator_adapter_demo
from ragtune.generative_validation_common import write_json, write_md
from ragtune.open_source_arxiv_readiness_synthesis import run_open_source_arxiv_readiness_synthesis
from ragtune.promotion_decision import build_promotion_decision, write_promotion_decision
from ragtune.public_mini_reproduction import run_public_mini_reproduction
from ragtune.rc1_maturity import verify_run
from ragtune.selector_ablation_matrix import run_selector_ablation_matrix


ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except Exception:
        config: dict[str, object] = {}
        current_list_key: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- ") and current_list_key:
                value = stripped[2:].strip()
                config.setdefault(current_list_key, [])
                assert isinstance(config[current_list_key], list)
                config[current_list_key].append(value)
                continue
            current_list_key = None
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                value = value.strip()
                if value == "":
                    config[key] = []
                    current_list_key = key
                elif value.lower() in {"true", "false"}:
                    config[key] = value.lower() == "true"
                else:
                    config[key] = value.strip("'\"")
        return config
    return yaml.safe_load(text) or {}


def _run_script(args: list[str]) -> int:
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=False).returncode


def cmd_validate_bundle(_args: argparse.Namespace) -> int:
    return _run_script(["scripts/validate_publication_bundle.py"])


def cmd_run_public_mini(args: argparse.Namespace) -> int:
    run_public_mini_reproduction(ROOT, output_root=ROOT / args.output_root)
    return 0


def cmd_run_selector_ablation(args: argparse.Namespace) -> int:
    run_selector_ablation_matrix(ROOT, output_root=ROOT / args.output_root)
    return 0


def cmd_run_external_evaluator_demo(args: argparse.Namespace) -> int:
    run_external_evaluator_adapter_demo(ROOT, output_root=ROOT / args.output_root)
    return 0


def cmd_run_claim_check(_args: argparse.Namespace) -> int:
    return cmd_validate_bundle(_args)


def cmd_synthesize_readiness(args: argparse.Namespace) -> int:
    run_open_source_arxiv_readiness_synthesis(ROOT, output_root=ROOT / args.output_root)
    return 0


def cmd_inspect_environment(args: argparse.Namespace) -> int:
    payload = {
        "containerized": os.environ.get("RAGTUNE_CONTAINER") == "1",
        "cloud_provider_hint": os.environ.get("RAGTUNE_CLOUD_PROVIDER", "unknown"),
        "python_version": platform.python_version(),
        "os_family": platform.system(),
        "storage_mode": os.environ.get("RAGTUNE_STORAGE_MODE", "local"),
        "output_root_configured": bool(os.environ.get("RAGTUNE_OUTPUT_ROOT") or args.output_root),
        "hostnames_exported": False,
        "private_paths_exported": False,
        "secrets_exported": False,
    }
    if args.output_root:
        write_json(Path(args.output_root) / "environment_inspection.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_export_decision(args: argparse.Namespace) -> int:
    result_class = args.result_class
    decision = build_promotion_decision(
        run_id=args.run_id,
        suite=args.suite,
        result_class=result_class,
        selected_policy=args.selected_policy,
        baseline_policy=args.baseline_policy,
        artifact_uris=args.artifact_uri,
        validator_status=args.validator_status,
    )
    write_promotion_decision(Path(args.decision_out), decision)
    return 0


def cmd_verify_run(args: argparse.Namespace) -> int:
    result = verify_run(ROOT, run_dir=ROOT / args.run_dir, output_root=ROOT / args.output_root)
    return 0 if result["result_class"] == "VERIFY_RUN_PASSED" else 1


def cmd_run_governance_job(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        decision = build_promotion_decision(
            run_id="missing_config",
            suite="ragtune_governance_job",
            result_class="BLOCK_MISSING_INPUTS",
            decision_reason="job config missing",
            validator_status="not_run",
        )
        write_promotion_decision(Path(args.decision_out), decision)
        return 2
    config = _load_config(config_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    steps = config.get("steps") or ["public_mini_reproduction"]
    artifacts: list[str] = []
    mini_result: dict[str, object] | None = None
    if "public_mini_reproduction" in steps:
        mini_result = run_public_mini_reproduction(
            ROOT,
            output_root=output_root / "public_mini_reproduction",
            write_repository_results=False,
        )
        artifacts.append("public_mini_reproduction/mini_reproduction_manifest.json")
    if "selector_ablation" in steps:
        run_selector_ablation_matrix(ROOT, output_root=output_root / "selector_ablation_matrix")
        artifacts.append("selector_ablation_matrix/selector_ablation_manifest.json")
    validator_status = "not_run"
    validator_rc = 0
    if config.get("run_publication_validator", True):
        validator_rc = cmd_validate_bundle(args)
        validator_status = "passed" if validator_rc == 0 else "failed"
    result_class = str((mini_result or {}).get("result_class", "INCONCLUSIVE"))
    decision = build_promotion_decision(
        run_id=str(config.get("run_id", "public_mini_governance_job")),
        suite=str(config.get("suite", "ragtune_cloud_agnostic_governance_job_v1")),
        result_class=result_class,
        selected_policy=str((mini_result or {}).get("governed_winner", "")),
        baseline_policy=str((mini_result or {}).get("quality_only_winner", "")),
        decision_reason="finite governance job completed with sanitized public mini reproduction",
        artifact_uris=artifacts,
        validator_status=validator_status,
        deltas={
            "quality_delta": float((mini_result or {}).get("governed_quality_delta_vs_quality_only", 0.0)),
            "cost_delta": float((mini_result or {}).get("governed_cost_delta_vs_quality_only", 0.0)),
        },
    )
    decision_out = Path(args.decision_out)
    write_promotion_decision(decision_out, decision)
    write_json(output_root / "run_manifest.json", {"job_config": "<config>", "decision_out": str(decision_out.name), "raw_text_exported": False})
    write_md(output_root / "validation_report.md", f"Governance job decision: `{decision['decision']}`. Validator: `{validator_status}`.")
    write_json(output_root / "validation_report.json", {"validator_status": validator_status, "decision": decision["decision"]})
    return 0 if validator_rc == 0 else validator_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragtune", description="RAGTune governance and promotion-control engine")
    sub = parser.add_subparsers(dest="command", required=False)
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "validate-bundle": cmd_validate_bundle,
        "run-public-mini": cmd_run_public_mini,
        "run-selector-ablation": cmd_run_selector_ablation,
        "run-external-evaluator-demo": cmd_run_external_evaluator_demo,
        "run-claim-check": cmd_run_claim_check,
        "synthesize-readiness": cmd_synthesize_readiness,
        "run-governance-job": cmd_run_governance_job,
        "export-decision": cmd_export_decision,
        "inspect-environment": cmd_inspect_environment,
        "verify-run": cmd_verify_run,
    }
    for name, func in commands.items():
        child = sub.add_parser(name)
        child.set_defaults(func=func)
        child.add_argument("--config", default="configs/jobs/public_mini_governance_job.yaml")
        child.add_argument("--output-root", default="artifacts/public_mini_governance_job")
        child.add_argument("--decision-out", default="artifacts/public_mini_governance_job/promotion_decision.json")
        child.add_argument("--force", action="store_true")
    sub.choices["run-public-mini"].set_defaults(output_root="artifacts/public_mini_reproduction")
    sub.choices["run-selector-ablation"].set_defaults(output_root="artifacts/selector_ablation_matrix")
    sub.choices["run-external-evaluator-demo"].set_defaults(output_root="artifacts/external_evaluator_adapters")
    sub.choices["synthesize-readiness"].set_defaults(output_root="results/open_source_arxiv_readiness")
    verify = sub.choices["verify-run"]
    verify.add_argument("--run-dir", default="artifacts/public_mini_reproduction")
    verify.set_defaults(output_root="artifacts/verify_run_demo")
    export = sub.choices["export-decision"]
    export.add_argument("--run-id", default="manual_export")
    export.add_argument("--suite", default="manual")
    export.add_argument("--result-class", default="INCONCLUSIVE")
    export.add_argument("--selected-policy", default="")
    export.add_argument("--baseline-policy", default="")
    export.add_argument("--artifact-uri", action="append", default=[])
    export.add_argument("--validator-status", default="not_run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
