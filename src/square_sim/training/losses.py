from __future__ import annotations


def positive_class_weight(y):
    import numpy as np

    y = np.asarray(y).astype(int)
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    return negatives / positives

