"""Unit tests for the feature engineering pipeline."""

import math
from datetime import datetime, timezone

import pytest

from src.common.schemas import Transaction
from src.ingestion.feature_engineering import FeatureEngineer, GeoTracker, VelocityTracker


class TestVelocityTracker:
    def test_single_transaction_velocity(self):
        tracker = VelocityTracker()
        ts = datetime(2024, 1, 1, 12, 0, 0)
        v1h, v24h, zscore = tracker.record_and_count("cust1", ts, 100.0)
        assert v1h == 1
        assert v24h == 1
        assert zscore == 0.0

    def test_velocity_accumulates(self):
        tracker = VelocityTracker()
        base = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            from datetime import timedelta
            ts = base + timedelta(minutes=i * 5)
            tracker.record_and_count("cust1", ts, 100.0)
        v1h, v24h, _ = tracker.record_and_count("cust1", base, 100.0)
        assert v1h >= 1
        assert v24h >= 1

    def test_zscore_nonzero_with_variance(self):
        tracker = VelocityTracker()
        base = datetime(2024, 1, 1, 12, 0, 0)
        from datetime import timedelta
        tracker.record_and_count("cust1", base, 100.0)
        tracker.record_and_count("cust1", base + timedelta(minutes=5), 200.0)
        _, _, zscore = tracker.record_and_count("cust1", base + timedelta(minutes=10), 5000.0)
        assert zscore > 0


class TestGeoTracker:
    def test_first_transaction_zero_distance(self):
        geo = GeoTracker()
        dist = geo.distance_km("cust1", 40.7128, -74.0060)
        assert dist == 0.0

    def test_same_location_zero_distance(self):
        geo = GeoTracker()
        geo.distance_km("cust1", 40.7128, -74.0060)
        dist = geo.distance_km("cust1", 40.7128, -74.0060)
        assert dist == pytest.approx(0.0, abs=0.01)

    def test_cross_continent_large_distance(self):
        geo = GeoTracker()
        geo.distance_km("cust1", 40.7128, -74.0060)  # New York
        dist = geo.distance_km("cust1", 51.5074, -0.1278)  # London
        assert dist > 5000


class TestFeatureEngineer:
    def _make_txn(self, **kwargs) -> Transaction:
        defaults = {
            "amount": 250.0,
            "customer_id": "cust-test-001",
            "account_id": "acc-test-001",
            "merchant_id": "merch-test-001",
            "merchant_category": "grocery",
            "device_id": "dev-abc",
            "ip_address": "192.168.1.1",
            "geolocation": {"lat": 40.0, "lon": -74.0},
        }
        defaults.update(kwargs)
        return Transaction(**defaults)

    def test_basic_enrichment(self):
        engineer = FeatureEngineer()
        txn = self._make_txn()
        enriched = engineer.enrich(txn)
        assert enriched.amount_log == pytest.approx(math.log1p(250.0), rel=1e-4)
        assert 0 <= enriched.hour_of_day <= 23
        assert 0 <= enriched.day_of_week <= 6

    def test_device_seen_before(self):
        engineer = FeatureEngineer()
        txn1 = self._make_txn(device_id="dev-unique-123")
        txn2 = self._make_txn(device_id="dev-unique-123")
        e1 = engineer.enrich(txn1)
        e2 = engineer.enrich(txn2)
        assert not e1.device_seen_before  # first time
        assert e2.device_seen_before       # second time

    def test_cross_border_detection(self):
        engineer = FeatureEngineer()
        txn1 = self._make_txn(geolocation={"lat": 40.7128, "lon": -74.0060})
        txn2 = self._make_txn(geolocation={"lat": 51.5074, "lon": -0.1278})
        engineer.enrich(txn1)
        e2 = engineer.enrich(txn2)
        assert e2.is_cross_border

    def test_merchant_risk_score_applied(self):
        engineer = FeatureEngineer()
        engineer.set_merchant_risk("high-risk-merchant", 0.9)
        txn = self._make_txn(merchant_id="high-risk-merchant")
        enriched = engineer.enrich(txn)
        assert enriched.merchant_risk_score == 0.9
