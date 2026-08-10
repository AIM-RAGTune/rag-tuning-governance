from __future__ import annotations


def sigmoid_scores(logits):
    import numpy as np

    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))

