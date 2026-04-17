"""Unit tests for FastAPI endpoints (no external dependencies required)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.schemas import FraudType, RiskLevel
from src.models.ensemble import EnsembleResult, EnsembleScorer, AdaptiveThreshold


def _make_mock_deps():
    deps = MagicMock()
    deps.cassandra = None
    deps.neo4j = None
    deps.graph_builder = None
    deps.graph_path_explainer = None
    deps.shap_explainer = None
    deps.llm_investigator = None
    deps.ensemble = MagicMock(spec=EnsembleScorer)
    deps.ensemble.score.return_value = EnsembleResult(
        ensemble_score=0.75,
        risk_level=RiskLevel.HIGH,
        model_scores=[],
        latency_ms=25.0,
        action="review",
    )
    deps.ensemble.feedback = MagicMock()
    deps.ensemble._threshold_agent = AdaptiveThreshold()
    return deps


@pytest.fixture
def client():
    """Build the app with mocked dependencies injected via deps_override."""
    from src.api.main import create_app

    mock_deps = _make_mock_deps()
    # Pass mock_deps directly into the lifespan so no real connections are made
    app = create_app(deps_override=mock_deps)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_security_headers_present(self, client):
        resp = client.get("/healthz")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "x-latency-ms" in resp.headers


class TestScoreEndpoint:
    def _valid_payload(self, **overrides):
        base = {
            "transaction": {
                "amount": 5000.0,
                "customer_id": "cust-test-001",
                "account_id": "acc-test-001",
                "merchant_id": "merch-test-001",
                "merchant_category": "electronics",
                "channel": "web",
            },
            "explain": False,
            "graph_path": False,
        }
        base.update(overrides)
        return base

    def test_score_returns_200(self, client):
        resp = client.post("/v1/score", json=self._valid_payload())
        assert resp.status_code == 200

    def test_score_response_fields(self, client):
        resp = client.post("/v1/score", json=self._valid_payload())
        data = resp.json()
        assert "ensemble_score" in data
        assert "risk_level" in data
        assert "action" in data
        assert "latency_ms" in data
        assert "alert_id" in data

    def test_score_preserves_transaction_id(self, client):
        payload = self._valid_payload()
        payload["transaction"]["transaction_id"] = "txn-explicit-id-999"
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200
        assert resp.json()["transaction_id"] == "txn-explicit-id-999"

    def test_score_rejects_negative_amount(self, client):
        payload = self._valid_payload()
        payload["transaction"]["amount"] = -100.0
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 422

    def test_score_rejects_zero_amount(self, client):
        payload = self._valid_payload()
        payload["transaction"]["amount"] = 0.0
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 422

    def test_score_rejects_injection_in_customer_id(self, client):
        payload = self._valid_payload()
        payload["transaction"]["customer_id"] = "'; DROP TABLE customers; --"
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 422

    def test_score_rejects_invalid_geolocation(self, client):
        payload = self._valid_payload()
        payload["transaction"]["geolocation"] = {"lat": 999.0, "lon": 0.0}
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 422

    def test_batch_limit_enforced(self, client):
        txns = [
            {"amount": 100.0, "customer_id": f"cust-{i}", "account_id": f"acc-{i}", "merchant_id": "m"}
            for i in range(101)
        ]
        resp = client.post("/v1/score/batch", json=txns)
        assert resp.status_code == 400
        assert "100" in resp.json()["detail"]

    def test_batch_within_limit(self, client):
        txns = [
            {"amount": 100.0, "customer_id": f"cust-{i:04d}", "account_id": f"acc-{i:04d}", "merchant_id": "merch-001"}
            for i in range(5)
        ]
        resp = client.post("/v1/score/batch", json=txns)
        assert resp.status_code == 200
        assert len(resp.json()) == 5


class TestAlertsEndpoint:
    def test_alerts_empty_without_cassandra(self, client):
        resp = client.get("/v1/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_alerts_respects_limit_param(self, client):
        resp = client.get("/v1/alerts?limit=10")
        assert resp.status_code == 200

    def test_alerts_rejects_excessive_limit(self, client):
        resp = client.get("/v1/alerts?limit=9999")
        assert resp.status_code == 422


class TestFeedbackEndpoint:
    def test_feedback_accepted(self, client):
        payload = {
            "alert_id": "alert-abc-123",
            "predicted_fraud": True,
            "actual_fraud": False,
        }
        resp = client.post("/v1/feedback", json=payload)
        assert resp.status_code == 204

    def test_feedback_calls_ensemble(self, client):
        from src.api.dependencies import get_dependencies
        payload = {
            "alert_id": "alert-xyz",
            "predicted_fraud": False,
            "actual_fraud": True,
        }
        client.post("/v1/feedback", json=payload)
        # Verify the mock was called
        app_deps = client.app.state.deps
        app_deps.ensemble.feedback.assert_called_with(False, True)


class TestBenchmarkEndpoint:
    def test_benchmark_returns_list(self, client):
        resp = client.get("/v1/benchmark")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_benchmark_run_queues_background_task(self, client):
        # Patch train_all so the background task never touches disk
        with patch("src.api.routers.benchmark.train_all", return_value={}):
            resp = client.post("/v1/benchmark/run")
        assert resp.status_code == 200
        assert "status" in resp.json()


class TestCasesEndpoint:
    def test_create_case(self, client):
        payload = {
            "alert_id": "alert-001",
            "transaction_id": "txn-001",
            "customer_id": "cust-001",
            "risk_level": "high",
            "fraud_type": "fraud_ring",
            "fraud_amount": 5000.0,
        }
        resp = client.post("/v1/cases", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert "case_id" in data
        assert data["customer_id"] == "cust-001"
