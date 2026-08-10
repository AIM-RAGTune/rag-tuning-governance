from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from square_sim.square_core.common.certificates import certificate_for_group
from square_sim.utils.files import write_json, write_text


def write_core_certificates(cert_dir: Path, experiment_id: str, metrics: pd.DataFrame) -> dict[str, Any]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    certs = []
    for (track, task), group in metrics.groupby(["track", "task"]) if not metrics.empty else []:
        cert = certificate_for_group(str(track), str(task), group)
        task_dir = cert_dir / str(track) / str(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "certificate.json", cert)
        write_text(task_dir / "certificate.md", f"# {track}/{task}\n\nStatus: **{cert['status']}**\n\n{cert['caveats'][0]}\n")
        certs.append(cert)
    index = {
        "experiment_id": experiment_id,
        "certificate_type": "SQUARE Core Validation Certificate Index",
        "certificates": certs,
    }
    write_json(cert_dir / "certificate_index.json", index)
    write_text(cert_dir / "certificate_index.md", "# SQUARE Core Certificate Index\n\n" + "\n".join(f"- `{c['track']}/{c['task']}`: {c['status']}" for c in certs) + "\n")
    return index
