"""Unit tests for ML fraud classifiers."""

import numpy as np
import pandas as pd
import pytest

from src.models.base import FEATURE_COLS
from src.models.ensemble import AdaptiveThreshold, EnsembleScorer
from src.models.logistic_regression import LogRegFraudClassifier
from src.models.xgboost_model import XGBoostFraudClassifier


def make_dataset(n: int = 200, fraud_rate: float = 0.1) -> tuple:
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_rate)
    n_normal = n - n_fraud

    X_fraud = pd.DataFrame(
        {
            "amount_log": rng.uniform(5, 10, n_fraud),
            "hour_of_day": rng.integers(0, 4, n_fraud),
            "day_of_week": rng.integers(0, 7, n_fraud),
            "velocity_1h": rng.integers(5, 20, n_fraud),
            "velocity_24h": rng.integers(10, 50, n_fraud),
            "amount_zscore": rng.uniform(3, 10, n_fraud),
            "merchant_risk_score": rng.uniform(0.7, 1.0, n_fraud),
            "device_seen_before": rng.integers(0, 2, n_fraud).astype(float),
            "ip_seen_before": rng.integers(0, 2, n_fraud).astype(float),
            "geo_distance_km": rng.uniform(1000, 15000, n_fraud),
            "is_cross_border": np.ones(n_fraud),
            "is_online": np.ones(n_fraud),
        }
    )
    X_normal = pd.DataFrame(
        {
            "amount_log": rng.uniform(1, 5, n_normal),
            "hour_of_day": rng.integers(8, 22, n_normal),
            "day_of_week": rng.integers(0, 7, n_normal),
            "velocity_1h": rng.integers(0, 3, n_normal),
            "velocity_24h": rng.integers(0, 10, n_normal),
            "amount_zscore": rng.uniform(-1, 1, n_normal),
            "merchant_risk_score": rng.uniform(0.0, 0.4, n_normal),
            "device_seen_before": np.ones(n_normal),
            "ip_seen_before": np.ones(n_normal),
            "geo_distance_km": rng.uniform(0, 50, n_normal),
            "is_cross_border": np.zeros(n_normal),
            "is_online": rng.integers(0, 2, n_normal).astype(float),
        }
    )
    X = pd.concat([X_fraud, X_normal], ignore_index=True)
    y = pd.Series([1] * n_fraud + [0] * n_normal)
    return X, y


class TestLogRegClassifier:
    def test_fit_predict_shape(self):
        X, y = make_dataset()
        model = LogRegFraudClassifier()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_fraud_score_higher_for_fraud(self):
        X, y = make_dataset(n=500)
        model = LogRegFraudClassifier()
        model.fit(X, y)
        fraud_X = X[y == 1]
        normal_X = X[y == 0]
        avg_fraud_score = model.predict_proba(fraud_X)[:, 1].mean()
        avg_normal_score = model.predict_proba(normal_X)[:, 1].mean()
        assert avg_fraud_score > avg_normal_score

    def test_save_load(self, tmp_path):
        X, y = make_dataset()
        model = LogRegFraudClassifier()
        model.fit(X, y)
        path = tmp_path / "logreg.pkl"
        model.save(path)
        loaded = LogRegFraudClassifier.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X[:10]),
            loaded.predict_proba(X[:10]),
        )


class TestXGBoostClassifier:
    def test_fit_predict_shape(self):
        X, y = make_dataset()
        model = XGBoostFraudClassifier(params={"n_estimators": 10, "max_depth": 3})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_feature_importance_keys(self):
        X, y = make_dataset()
        model = XGBoostFraudClassifier(params={"n_estimators": 5})
        model.fit(X, y)
        fi = model.feature_importance()
        assert set(fi.keys()) == set(FEATURE_COLS)

    def test_save_load(self, tmp_path):
        X, y = make_dataset()
        model = XGBoostFraudClassifier(params={"n_estimators": 5})
        model.fit(X, y)
        path = tmp_path / "xgb.ubj"
        model.save(path)
        loaded = XGBoostFraudClassifier.load(path)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X[:5]),
            loaded.predict_proba(X[:5]),
        )


class TestAdaptiveThreshold:
    def test_threshold_decreases_on_false_negative(self):
        agent = AdaptiveThreshold(initial_threshold=0.5, lr=0.05)
        initial = agent.threshold
        agent.update(predicted_fraud=False, actual_fraud=True)
        assert agent.threshold < initial

    def test_threshold_increases_on_false_positive(self):
        agent = AdaptiveThreshold(initial_threshold=0.5, lr=0.05)
        initial = agent.threshold
        agent.update(predicted_fraud=True, actual_fraud=False)
        assert agent.threshold > initial

    def test_threshold_bounds(self):
        agent = AdaptiveThreshold(initial_threshold=0.5, lr=0.5)
        for _ in range(100):
            agent.update(predicted_fraud=True, actual_fraud=False)
        assert agent.threshold <= 0.95
        for _ in range(200):
            agent.update(predicted_fraud=False, actual_fraud=True)
        assert agent.threshold >= 0.05


class TestEnsembleScorer:
    def test_score_without_models_returns_result(self):
        scorer = EnsembleScorer()
        X, _ = make_dataset(n=5)
        result = scorer.score(X)
        assert 0.0 <= result.ensemble_score <= 1.0
        assert result.action in ("block", "review", "pass")

    def test_risk_level_classification(self):
        scorer = EnsembleScorer()
        from src.common.schemas import RiskLevel
        assert scorer._classify_risk(0.9) == RiskLevel.CRITICAL
        assert scorer._classify_risk(0.7) == RiskLevel.HIGH
        assert scorer._classify_risk(0.5) == RiskLevel.MEDIUM
        assert scorer._classify_risk(0.2) == RiskLevel.LOW
