from pathlib import Path

from square_sim.reporting.certificate import classify_certificate, generate_certificate_report
from square_sim.reporting.explain_run import generate_run_explanation


def test_explanation_generation(tmp_path: Path):
    output = tmp_path / "explanation.md"
    text = generate_run_explanation(
        output,
        {
            "run_id": "run1",
            "dataset": "energy",
            "dataset_version": "v1",
            "source": "synthetic",
            "target": "target",
            "model": "squaresim_full",
            "metrics": {"roc_auc": 0.8},
            "resources": {"train_seconds": 1.0},
            "status": "Candidate",
        },
    )
    assert output.exists()
    assert "not physical" in text


def test_certificate_status_logic():
    rows = [
        {"model": "logistic_regression", "roc_auc": 0.80},
        {"model": "fourier_mlp", "roc_auc": 0.81, "train_seconds": 2},
        {"model": "squaresim_full", "roc_auc": 0.85, "train_seconds": 3},
        {"model": "squaresim_no_feedback", "roc_auc": 0.82},
        {"model": "squaresim_no_nonlinear", "roc_auc": 0.83},
    ]
    status, _ = classify_certificate(
        "target",
        rows,
        {"square_vs_classical": {"ci95": [0.01, 0.08]}},
    )
    assert status == "Simulation-supported advantage"


def test_certificate_report_generation(tmp_path: Path):
    output = tmp_path / "cert.md"
    payload = generate_certificate_report(output, "energy", "target_real", [{"model": "logistic_regression", "roc_auc": 0.9}, {"model": "squaresim_full", "roc_auc": 0.8}])
    assert payload["status"] == "Refused"
    assert output.exists()

