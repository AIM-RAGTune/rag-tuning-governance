from __future__ import annotations

from pathlib import Path

from square_sim.reporting.certificate import _score
from square_sim.utils.files import write_text


def compare_metrics_rows(output_path: Path, rows: list[dict]) -> str:
    sorted_rows = sorted(rows, key=_score, reverse=True)
    lines = ["# SQUARESim Run Comparison", ""]
    for row in sorted_rows:
        lines.append(
            f"- {row.get('model')}: ROC-AUC={row.get('roc_auc')}, PR-AUC={row.get('pr_auc')}, "
            f"train_seconds={row.get('train_seconds')}"
        )
    text = "\n".join(lines) + "\n"
    write_text(output_path, text)
    return text

