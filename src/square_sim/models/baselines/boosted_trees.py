from __future__ import annotations


def make_optional_booster(model_name: str, seed: int = 42):
    if model_name == "xgboost_optional":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    if model_name == "lightgbm_optional":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            return None
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed, verbose=-1)
    raise ValueError(f"Unknown optional booster '{model_name}'.")

