"""Unit tests for FastAPI endpoints (no external dependencies)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.common.schemas import FraudType, RiskLevel
from src.models.ensemble import EnsembleResult


@pytest.fixture
def mock_deps():
    deps = MagicMock()
    deps.cassandra = None
    deps.neo4j = None
    deps.graph_builder = None
    deps.graph_path_explainer = None
    deps.shap_explainer = None
    deps.llm_investigator = None

    # Ensemble returns a deterministic result
    deps.ensemble.score.return_value = EnsembleResult(
        ensemble_score=0.75,
        risk_level=RiskLevel.HIGH,
        model_scores=[],
        latency_ms=25.0,
        action="review",
    )
    deps.ensemble.feedback = MagicMock()
    deps.ensemble._threshold_agent.threshold = 0.5
    return deps


@pytest.fixture
def client(mock_deps):
    app = create_app()
    app.state.deps = mock_deps

    # Bypass lifespan
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestScoreEndpoint:
    def test_score_transaction(self, client):
        payload = {
            "transaction": {
                "amount": 5000.0,
                "customer_id": "cust-test",
                "account_id": "acc-test",
                "merchant_id": "merch-test",
                "merchant_category": "electronics",
            },
            "explain": False,
            "graph_path": False,
        }
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "ensemble_score" in data
        assert "risk_level" in data
        assert "action" in data
        assert "latency_ms" in data

    def test_score_includes_transaction_id(self, client):
        payload = {
            "transaction": {
                "transaction_id": "test-txn-id-123",
                "amount": 100.0,
                "customer_id": "cust-1",
                "account_id": "acc-1",
                "merchant_id": "merch-1",
            }
        }
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200
        assert resp.json()["transaction_id"] == "test-txn-id-123"

    def test_batch_size_limit(self, client):
        txns = [{"amount": 100.0, "customer_id": f"c{i}", "account_id": f"a{i}", "merchant_id": "m"} for i in range(101)]
        resp = client.post("/v1/score/batch", json=txns)
        assert resp.status_code == 400


class TestAlertsEndpoint:
    def test_alerts_empty_without_cassandra(self, client):
        resp = client.get("/v1/alerts")
        assert resp.status_code == 200
        assert resp.json() == []


class TestFeedbackEndpoint:
    def test_submit_feedback(self, client, mock_deps):
        payload = {
            "alert_id": "alert-123",
            "predicted_fraud": True,
            "actual_fraud": False,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 204
        mock_deps.ensemble.feedback.assert_called_once_with(True, False)


class TestBenchmarkEndpoint:
    def test_benchmark_returns_list(self, client):
        resp = client.get("/v1/benchmark")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
