"""SHAP explainability for XGBoost and LogReg models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from src.common.logging import get_logger
from src.models.base import FEATURE_COLS

logger = get_logger(__name__)


class SHAPExplainer:
    """Compute SHAP values for XGBoost and LogReg models."""

    def __init__(self, model: Any, model_type: str = "xgboost") -> None:
        self._model = model
        self._model_type = model_type
        self._explainer: Optional[Any] = None

    def _build_explainer(self, background: pd.DataFrame) -> None:
        if self._model_type == "xgboost":
            self._explainer = shap.TreeExplainer(self._model._model)
        elif self._model_type == "logistic_regression":
            bg = shap.sample(background, min(100, len(background)))
            self._explainer = shap.KernelExplainer(
                lambda x: self._model._pipeline.predict_proba(pd.DataFrame(x, columns=FEATURE_COLS))[:, 1],
                bg,
            )
        else:
            self._explainer = shap.Explainer(self._model.predict_proba, background)

    def explain(self, X: pd.DataFrame, background: Optional[pd.DataFrame] = None) -> List[Dict[str, float]]:
        """Return per-feature SHAP values for each row in X."""
        X_proc = X.reindex(columns=FEATURE_COLS, fill_value=0.0)

        if self._explainer is None:
            bg = background if background is not None else X_proc
            self._build_explainer(bg)

        try:
            if self._model_type == "xgboost":
                shap_values = self._explainer.shap_values(X_proc)
                # XGBoost returns shape (n, features) for binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                sv = self._explainer(X_proc)
                shap_values = sv.values if hasattr(sv, "values") else sv

            results = []
            for row_shap in shap_values:
                results.append(
                    {col: round(float(v), 5) for col, v in zip(FEATURE_COLS, row_shap)}
                )
            return results
        except Exception as exc:
            logger.error("SHAP explanation failed", error=str(exc))
            return [{col: 0.0 for col in FEATURE_COLS}] * len(X)

    def top_features(self, shap_dict: Dict[str, float], n: int = 5) -> List[Dict[str, Any]]:
        """Return top-n features sorted by |SHAP|."""
        sorted_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [{"feature": k, "shap_value": v, "direction": "↑ fraud" if v > 0 else "↓ fraud"} for k, v in sorted_items[:n]]
