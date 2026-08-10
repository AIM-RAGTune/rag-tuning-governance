from __future__ import annotations

from collections.abc import Callable


def paired_bootstrap_delta(
    y_true,
    score_a,
    score_b,
    metric_fn: Callable,
    samples: int = 1000,
    seed: int = 42,
) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = len(y_true)
    deltas = []
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        try:
            deltas.append(float(metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])))
        except ValueError:
            continue
    if not deltas:
        return {"mean_delta": None, "ci95": [None, None], "sign_estimate": None, "samples": 0, "seed": seed}
    arr = np.asarray(deltas)
    return {
        "mean_delta": float(arr.mean()),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "sign_estimate": float((arr > 0).mean()),
        "samples": len(arr),
        "seed": seed,
    }

