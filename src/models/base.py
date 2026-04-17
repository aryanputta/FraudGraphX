"""Abstract base class for all FraudGraphX classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "velocity_1h",
    "velocity_24h",
    "amount_zscore",
    "merchant_risk_score",
    "device_seen_before",
    "ip_seen_before",
    "geo_distance_km",
    "is_cross_border",
    "is_online",
]

LABEL_COL = "is_fraud"


class FraudClassifier(ABC):
    """Common interface for all fraud detection models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "FraudClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n, 2) probability array, col-1 = P(fraud)."""
        ...

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "FraudClassifier":
        ...

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce types, fill missing values, return feature matrix."""
        out = df.reindex(columns=FEATURE_COLS, fill_value=0.0).copy()
        bool_cols = ["device_seen_before", "ip_seen_before", "is_cross_border", "is_online"]
        for col in bool_cols:
            if col in out.columns:
                out[col] = out[col].astype(float)
        return out.fillna(0.0)
