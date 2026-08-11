from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_exists() -> None:
    assert (ROOT / "Dockerfile").exists()


def test_dockerfile_sets_container_env() -> None:
    assert "RAGTUNE_CONTAINER=1" in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_has_entrypoint() -> None:
    assert 'ENTRYPOINT ["ragtune"]' in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_supports_outputs_mount() -> None:
    assert "/outputs" in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_does_not_copy_local_data() -> None:
    assert ".local_data" not in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_docker_compose_public_mini_exists() -> None:
    assert (ROOT / "docker/compose.public-mini.yml").exists()
    assert (ROOT / "docker-compose.yml").exists()


def test_docker_helper_scripts_exist() -> None:
    for rel in ["docker/run_public_mini.sh", "docker/run_governance_job.sh", "docker/healthcheck.sh"]:
        assert (ROOT / rel).exists()


def test_docker_helper_scripts_are_executable() -> None:
    for rel in ["docker/run_public_mini.sh", "docker/run_governance_job.sh", "docker/healthcheck.sh"]:
        assert (ROOT / rel).stat().st_mode & 0o111


def test_makefile_docker_targets_exist() -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ["docker-build", "docker-validate", "docker-run-public-mini", "docker-compose-public-mini"]:
        assert target in text
