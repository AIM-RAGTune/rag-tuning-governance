from __future__ import annotations

from typing import Any


def make_sklearn_baseline(model_name: str, seed: int = 42) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=seed
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    if model_name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed)),
            ]
        )
    raise ValueError(f"Unknown sklearn baseline '{model_name}'.")


def predict_proba_positive(model: Any, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        import numpy as np

        logits = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-logits))
    raise RuntimeError("Model does not provide predict_proba or decision_function.")

