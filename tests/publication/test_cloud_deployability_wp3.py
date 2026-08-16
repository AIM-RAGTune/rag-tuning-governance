from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_publish_container_workflow_has_supply_chain_controls() -> None:
    text = (ROOT / ".github" / "workflows" / "publish-container.yml").read_text(encoding="utf-8")
    assert "platforms: linux/amd64,linux/arm64" in text
    assert "provenance: mode=max" in text
    assert "sbom: true" in text
    assert "cosign sign --yes" in text
    assert "ghcr.io/aim-ragtune/rag-tuning-governance" in text
    assert ":latest" not in text


def test_image_digest_file_is_pending_or_real_digest() -> None:
    text = (ROOT / "deploy" / "IMAGE_DIGEST").read_text(encoding="utf-8")
    fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    assert fields["IMAGE"] == "ghcr.io/aim-ragtune/rag-tuning-governance"
    assert fields["REAL_CLOUD_DEPLOYMENT_PERFORMED"] == "false"
    digest = fields["DIGEST"]
    assert digest == "PENDING_FIRST_WORKFLOW_RUN" or re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert re.fullmatch(r"[0-9a-f]{64}", fields["BUILD_INPUT_SHA256"])


def test_build_input_hash_is_stable_and_nonempty() -> None:
    first = subprocess.check_output([sys.executable, "scripts/compute_container_build_input_hash.py"], cwd=ROOT, text=True).strip()
    second = subprocess.check_output([sys.executable, "scripts/compute_container_build_input_hash.py"], cwd=ROOT, text=True).strip()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_record_image_digest_rejects_non_digest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/record_image_digest.py",
            "--digest",
            "not-a-digest",
            "--build-input-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "IMAGE_DIGEST"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "IMAGE_DIGEST").exists()

