from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[2]


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
