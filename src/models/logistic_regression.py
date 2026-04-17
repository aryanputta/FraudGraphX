"""Logistic Regression baseline fraud classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import FEATURE_COLS, FraudClassifier


class LogRegFraudClassifier(FraudClassifier):
    name = "logistic_regression"

    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="saga",
                        C=0.1,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "LogRegFraudClassifier":
        X_proc = self.preprocess(X)
        self._pipeline.fit(X_proc, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = self.preprocess(X)
        return self._pipeline.predict_proba(X_proc)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f)

    @classmethod
    def load(cls, path: Path) -> "LogRegFraudClassifier":
        instance = cls()
        with open(path, "rb") as f:
            instance._pipeline = pickle.load(f)
        return instance
