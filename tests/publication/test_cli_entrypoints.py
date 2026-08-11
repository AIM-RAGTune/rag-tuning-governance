from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, "-m", "ragtune.cli", "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "run-governance-job" in result.stdout


def test_cli_export_decision_writes_machine_readable_json(tmp_path: Path) -> None:
    out = tmp_path / "promotion_decision.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragtune.cli",
            "export-decision",
            "--decision-out",
            str(out),
            "--result-class",
            "PUBLIC_MINI_REPRODUCTION_FAIL_CLOSED",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["decision"] == "BLOCK"


def test_cli_inspect_environment_sanitized(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ragtune.cli", "inspect-environment", "--output-root", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["secrets_exported"] is False
    assert payload["private_paths_exported"] is False
