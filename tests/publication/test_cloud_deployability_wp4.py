from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _deploy_files() -> list[Path]:
    return sorted(p for p in (ROOT / "deploy").rglob("*") if p.is_file())


def test_deploy_templates_do_not_use_latest_or_example_registries() -> None:
    offenders: list[str] = []
    for path in _deploy_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "ubuntu-latest" in text:
            continue
        for token in ("ghcr.io/example", "ragtune-governance:latest", "rag-tuning-governance:latest", "dkr.ecr", "us-docker.pkg.dev"):
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert offenders == []


def test_digest_helper_fails_closed_when_digest_pending() -> None:
    result = subprocess.run(
        ["bash", "deploy/load-image-reference.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Image digest is pending" in result.stderr


def test_cloud_scripts_preflight_with_digest_helper() -> None:
    scripts = [
        "deploy/aws/deploy-ecs-fargate.sh",
        "deploy/aws/run-ecs-task.sh",
        "deploy/aws/submit-batch-job.sh",
        "deploy/azure/deploy-aca-job.sh",
        "deploy/azure/run-aca-job.sh",
        "deploy/gcp/deploy-cloud-run-job.sh",
        "deploy/gcp/run-cloud-run-job.sh",
    ]
    for script in scripts:
        text = (ROOT / script).read_text(encoding="utf-8")
        assert "load-image-reference.sh" in text


def test_kubernetes_manifests_are_non_root_and_read_only() -> None:
    for name in ("ragtune-job.yaml", "ragtune-cronjob.yaml"):
        text = (ROOT / "deploy" / "kubernetes" / name).read_text(encoding="utf-8")
        assert "runAsNonRoot: true" in text
        assert "runAsUser: 10001" in text
        assert "readOnlyRootFilesystem: true" in text
        assert "allowPrivilegeEscalation: false" in text
        assert "mountPath: /inputs" in text
        assert "mountPath: /outputs" in text


def test_k8s_kind_validator_reports_fallback_without_tools(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    result = subprocess.run(
        ["/bin/bash", "scripts/validate_k8s_kind.sh", str(tmp_path / "report")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "FALLBACK_KUBECTL_UNAVAILABLE" in (tmp_path / "report" / "k8s_kind_validation_report.json").read_text(encoding="utf-8")
