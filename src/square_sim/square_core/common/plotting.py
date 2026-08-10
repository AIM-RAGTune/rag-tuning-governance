from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def write_placeholder_plot(path: Path, title: str, values: Iterable[float]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = ",".join(f"{float(v):.4f}" for v in values)
    path.write_text(f"{title}\n{vals}\n", encoding="utf-8")
    return str(path)
