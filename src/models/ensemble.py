"""
Ensemble scorer combining LogReg + XGBoost + GNN with soft voting.
Also implements the adaptive threshold RL agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import FraudType, ModelScore, RiskLevel
from src.models.base import FraudClassifier
from src.models.logistic_regression import LogRegFraudClassifier
from src.models.xgboost_model import XGBoostFraudClassifier

# GNN is optional – requires torch + torch-geometric
try:
    from src.models.gnn_model import GNNFraudClassifier
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class EnsembleResult:
    ensemble_score: float
    risk_level: RiskLevel
    model_scores: List[ModelScore]
    latency_ms: float
    action: str


class AdaptiveThreshold:
    """
    Simple ε-greedy reinforcement learning agent that adjusts the decision
    threshold based on observed false-positive and false-negative feedback.
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        lr: float = 0.01,
        epsilon: float = 0.1,
    ) -> None:
        self.threshold = initial_threshold
        self.lr = lr
        self.epsilon = epsilon

    def update(self, predicted_fraud: bool, actual_fraud: bool) -> None:
        """Q-learning-style update using binary reward signal."""
        if predicted_fraud and not actual_fraud:            # False Positive
            reward = settings.rl_reward_fp_penalty
        elif not predicted_fraud and actual_fraud:          # False Negative
            reward = settings.rl_reward_fn_penalty
        elif predicted_fraud and actual_fraud:              # True Positive
            reward = settings.rl_reward_tp_reward
        else:                                               # True Negative
            reward = 0.0

        # Gradient step: lower threshold when we miss fraud, raise when FP too high
        if not predicted_fraud and actual_fraud:
            self.threshold -= self.lr * abs(reward)
        elif predicted_fraud and not actual_fraud:
            self.threshold += self.lr * abs(reward)

        self.threshold = max(0.05, min(0.95, self.threshold))

    def decide(self, score: float) -> bool:
        """ε-greedy decision: explore with probability epsilon."""
        if np.random.random() < self.epsilon:
            return bool(np.random.random() > 0.5)
        return score >= self.threshold


class EnsembleScorer:
    """
    Weighted ensemble of LogReg + XGBoost + GNN.
    Weights are configurable; GNN gets a higher weight for graph anomalies.
    """

    WEIGHTS = {"logistic_regression": 0.2, "xgboost": 0.5, "gnn": 0.3}

    def __init__(self) -> None:
        self._models: Dict[str, FraudClassifier] = {}
        self._threshold_agent = AdaptiveThreshold(
            initial_threshold=settings.fraud_threshold,
            lr=settings.rl_threshold_learning_rate,
            epsilon=settings.rl_epsilon,
        )

    def load_models(self) -> "EnsembleScorer":
        """Load all three models from disk (best-effort; skip missing)."""
        logreg_path = Path(settings.logreg_model_path)
        xgb_path = Path(settings.xgboost_model_path)
        gnn_path = Path(settings.gnn_model_path)

        if logreg_path.exists():
            self._models["logistic_regression"] = LogRegFraudClassifier.load(logreg_path)
            logger.info("LogReg model loaded")
        if xgb_path.exists():
            self._models["xgboost"] = XGBoostFraudClassifier.load(xgb_path)
            logger.info("XGBoost model loaded")
        if gnn_path.exists() and _GNN_AVAILABLE:
            self._models["gnn"] = GNNFraudClassifier.load(gnn_path)  # type: ignore[name-defined]
            logger.info("GNN model loaded")
        elif gnn_path.exists():
            logger.warning("GNN model found but torch not installed – skipping")

        if not self._models:
            logger.warning("No models loaded – using deterministic rule-based fallback")
        return self

    def score(self, features: pd.DataFrame) -> EnsembleResult:
        """Score one or more transactions and return ensemble result."""
        t0 = time.perf_counter()
        model_scores: List[ModelScore] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for name, model in self._models.items():
            t_m = time.perf_counter()
            proba = model.predict_proba(features)[:, 1]
            m_latency = (time.perf_counter() - t_m) * 1000
            score = float(proba.mean())
            weight = self.WEIGHTS.get(name, 0.33)
            weighted_sum += score * weight
            total_weight += weight
            model_scores.append(ModelScore(model_name=name, score=round(score, 4), latency_ms=round(m_latency, 2)))

        if total_weight == 0:
            # Fallback rule engine
            amount = features.get("amount", pd.Series([0])).iloc[0]
            vel = features.get("velocity_1h", pd.Series([0])).iloc[0]
            ensemble_score = min(1.0, (amount / 10000) * 0.4 + (vel / 10) * 0.6)
        else:
            ensemble_score = weighted_sum / total_weight

        is_fraud = self._threshold_agent.decide(ensemble_score)
        risk_level = self._classify_risk(ensemble_score)
        action = "block" if risk_level == RiskLevel.CRITICAL else ("review" if is_fraud else "pass")
        latency_ms = (time.perf_counter() - t0) * 1000

        return EnsembleResult(
            ensemble_score=round(ensemble_score, 4),
            risk_level=risk_level,
            model_scores=model_scores,
            latency_ms=round(latency_ms, 2),
            action=action,
        )

    def feedback(self, predicted_fraud: bool, actual_fraud: bool) -> None:
        """Incorporate analyst feedback to adapt decision threshold."""
        self._threshold_agent.update(predicted_fraud, actual_fraud)

    @staticmethod
    def _classify_risk(score: float) -> RiskLevel:
        if score >= 0.85:
            return RiskLevel.CRITICAL
        if score >= 0.65:
            return RiskLevel.HIGH
        if score >= 0.40:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
