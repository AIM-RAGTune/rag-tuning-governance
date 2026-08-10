from __future__ import annotations

import pandas as pd


def repeated_seed_summary(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    values = frame[metric].astype(float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "count": int(values.count())}
