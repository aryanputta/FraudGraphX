"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    FraudAlert,
    FraudCase,
    FraudType,
    ModelScore,
    RiskLevel,
    Transaction,
)


class TestTransaction:
    def test_minimal_transaction(self):
        txn = Transaction(
            amount=100.0,
            customer_id="cust-001",
            account_id="acc-001",
            merchant_id="merch-001",
        )
        assert txn.transaction_id  # auto-generated
        assert txn.currency == "USD"

    def test_transaction_id_auto_generated(self):
        t1 = Transaction(amount=1.0, customer_id="c", account_id="a", merchant_id="m")
        t2 = Transaction(amount=1.0, customer_id="c", account_id="a", merchant_id="m")
        assert t1.transaction_id != t2.transaction_id

    def test_serialization_roundtrip(self):
        txn = Transaction(
            amount=500.0,
            customer_id="cust-abc",
            account_id="acc-xyz",
            merchant_id="merch-123",
            geolocation={"lat": 40.7, "lon": -74.0},
        )
        data = txn.model_dump_json()
        restored = Transaction.model_validate_json(data)
        assert restored.transaction_id == txn.transaction_id
        assert restored.amount == txn.amount


class TestFraudAlert:
    def test_alert_defaults(self):
        alert = FraudAlert(
            transaction_id="txn-1",
            customer_id="cust-1",
            ensemble_score=0.87,
            risk_level=RiskLevel.CRITICAL,
        )
        assert alert.alert_id
        assert alert.fraud_type == FraudType.UNKNOWN
        assert alert.action == "review"

    def test_model_scores_optional(self):
        alert = FraudAlert(
            transaction_id="t",
            customer_id="c",
            ensemble_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            model_scores=[
                ModelScore(model_name="xgboost", score=0.6),
                ModelScore(model_name="gnn", score=0.4),
            ],
        )
        assert len(alert.model_scores) == 2


class TestFraudCase:
    def test_case_creation(self):
        from src.common.schemas import CaseStatus
        case = FraudCase(
            alert_id="alert-1",
            transaction_id="txn-1",
            customer_id="cust-1",
        )
        assert case.status == CaseStatus.OPEN
        assert case.case_id
