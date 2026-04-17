"""XGBoost fraud classifier with Optuna hyperparameter tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.common.logging import get_logger
from src.models.base import FEATURE_COLS, FraudClassifier

logger = get_logger(__name__)


class XGBoostFraudClassifier(FraudClassifier):
    name = "xgboost"

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 50,   # imbalance correction
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model = xgb.XGBClassifier(**self._params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        **kwargs: Any,
    ) -> "XGBoostFraudClassifier":
        X_proc = self.preprocess(X)
        fit_kwargs: Dict[str, Any] = {"verbose": 50}
        if eval_set:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(self.preprocess(X_val), y_val)]
            fit_kwargs["early_stopping_rounds"] = 30
        self._model.fit(X_proc, y, **fit_kwargs)
        logger.info("XGBoost trained", best_iteration=getattr(self._model, "best_iteration", "N/A"))
        return self

    def tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> Dict[str, Any]:
        """Optuna HPO – returns best params."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0, 1),
                "scale_pos_weight": trial.suggest_int("scale_pos_weight", 10, 100),
                "tree_method": "hist",
                "eval_metric": "aucpr",
                "random_state": 42,
            }
            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, self.preprocess(X), y, cv=cv, scoring="average_precision", n_jobs=-1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best = study.best_params
        logger.info("Optuna tuning complete", best_score=study.best_value, best_params=best)
        self._params.update(best)
        self._model = xgb.XGBClassifier(**self._params)
        return best

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.preprocess(X)
        return self._model.predict_proba(X_proc)

    def feature_importance(self) -> Dict[str, float]:
        fi = self._model.feature_importances_
        return dict(zip(FEATURE_COLS, fi.tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        meta_path = path.with_suffix(".params.json")
        meta_path.write_text(json.dumps(self._params, indent=2))

    @classmethod
    def load(cls, path: Path) -> "XGBoostFraudClassifier":
        instance = cls()
        instance._model = xgb.XGBClassifier()
        instance._model.load_model(str(path))
        meta_path = path.with_suffix(".params.json")
        if meta_path.exists():
            instance._params = json.loads(meta_path.read_text())
        return instance
