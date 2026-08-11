from __future__ import annotations

import json
from pathlib import Path

from ragtune.container_smoke_tests import SMOKE_RESULT_CLASSES


ROOT = Path(__file__).resolve().parents[2]


def test_container_smoke_test_manifest_exists() -> None:
    assert (ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").exists()


def test_container_smoke_test_result_class_allowed() -> None:
    payload = json.loads((ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").read_text(encoding="utf-8"))
    assert payload["result_class"] in SMOKE_RESULT_CLASSES


def test_container_runtime_skip_has_clear_reason() -> None:
    payload = json.loads((ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").read_text(encoding="utf-8"))
    if "SKIPPED" in payload["result_class"]:
        assert payload["skip_reason"]
        static = json.loads((ROOT / "artifacts/docker_hardening/docker_static_validation.json").read_text(encoding="utf-8"))
        assert static["result_class"] == "DOCKER_STATIC_VALIDATION_PASSED"


def test_container_smoke_test_does_not_claim_cloud_validation() -> None:
    payload = json.loads((ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").read_text(encoding="utf-8"))
    assert payload["live_cloud_validation_claimed"] is False
    assert payload["official_platform_benchmarking_claimed"] is False


def test_container_smoke_test_does_not_claim_production_readiness() -> None:
    payload = json.loads((ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").read_text(encoding="utf-8"))
    assert payload["production_readiness_claimed"] is False


def test_promotion_decision_container_exists_if_runtime_validated() -> None:
    payload = json.loads((ROOT / "artifacts/docker_hardening/container_smoke_test_manifest.json").read_text(encoding="utf-8"))
    if "VALIDATED" in payload["result_class"]:
        assert (ROOT / "artifacts/docker_hardening/promotion_decision_container.json").exists()
