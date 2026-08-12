from pathlib import Path
import json
import subprocess
import importlib.util


ROOT = Path(__file__).resolve().parents[2]


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_publication_bundle", ROOT / "scripts" / "validate_publication_bundle.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_publication_files_exist() -> None:
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "docs/claim_boundaries.md",
        "data/DATA_AVAILABILITY.md",
        "results/run_index.csv",
        "results/evidence_summary.json",
        "results/claim_status/claim_status_table.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert missing == []


def test_claim_boundaries_reject_rag_compass_superiority() -> None:
    text = (ROOT / "docs/claim_boundaries.md").read_text(encoding="utf-8")
    assert "RAG Compass superiority" in text
    assert "Unsupported" in text


def test_evidence_summary_has_unsupported_claims() -> None:
    summary = json.loads((ROOT / "results/evidence_summary.json").read_text(encoding="utf-8"))
    assert "RAG Compass superiority" in summary["unsupported_claims"]
    assert "human-evaluation validation" in summary["unsupported_claims"]


def test_publication_validator_passes() -> None:
    result = subprocess.run(
        ["python", "scripts/validate_publication_bundle.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_publication_validator_remote_modes(monkeypatch) -> None:
    validator = load_validator_module()
    remote = "origin\thttps://github.com/AIM-RAGTune/rag-tuning-governance.git (fetch)"
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setenv("RAGTUNE_PUBLICATION_REMOTE_MODE", "local_unpublished")
    assert validator.public_repository_remote_allowed(remote) is False
    monkeypatch.setenv("RAGTUNE_PUBLICATION_REMOTE_MODE", "deployed_public_repo")
    assert validator.public_repository_remote_allowed(remote) is True
    monkeypatch.setenv("RAGTUNE_PUBLICATION_REMOTE_MODE", "unexpected_mode")
    assert validator.public_repository_remote_allowed(remote) is False


def test_publication_validator_github_actions_public_mode(monkeypatch) -> None:
    validator = load_validator_module()
    remote_without_dot_git = "origin\thttps://github.com/AIM-RAGTune/rag-tuning-governance (fetch)"
    monkeypatch.delenv("RAGTUNE_PUBLICATION_REMOTE_MODE", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "AIM-RAGTune/rag-tuning-governance")
    assert validator.publication_remote_mode() == "github_actions_public_repo"
    assert validator.public_repository_remote_allowed(remote_without_dot_git) is True


def test_publication_validator_rejects_unexpected_github_actions_repo(monkeypatch) -> None:
    validator = load_validator_module()
    remote = "origin\thttps://github.com/other/repo.git (fetch)"
    monkeypatch.delenv("RAGTUNE_PUBLICATION_REMOTE_MODE", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "other/repo")
    assert validator.publication_remote_mode() == "github_actions_unapproved_repo"
    assert validator.public_repository_remote_allowed(remote) is False
