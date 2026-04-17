"""Real-time feature engineering applied to raw transactions."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

from src.common.schemas import EnrichedTransaction, Transaction


class VelocityTracker:
    """Rolling-window transaction velocity counter (in-memory, per-customer)."""

    def __init__(self, window_seconds: int = 3600) -> None:
        self._window = window_seconds
        # customer_id → deque of (timestamp, amount)
        self._history: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(deque)

    def record_and_count(self, customer_id: str, ts: datetime, amount: float) -> Tuple[int, int, float]:
        """Returns (velocity_1h, velocity_24h, amount_zscore)."""
        history = self._history[customer_id]
        history.append((ts, amount))

        # Prune entries older than 24h
        cutoff_24h = ts - timedelta(hours=24)
        cutoff_1h = ts - timedelta(hours=1)
        while history and history[0][0] < cutoff_24h:
            history.popleft()

        amounts_24h = [a for t, a in history]
        velocity_24h = len(amounts_24h)
        velocity_1h = sum(1 for t, _ in history if t >= cutoff_1h)

        if len(amounts_24h) >= 2:
            mean = sum(amounts_24h) / len(amounts_24h)
            variance = sum((a - mean) ** 2 for a in amounts_24h) / len(amounts_24h)
            std = math.sqrt(variance) or 1.0
            zscore = (amount - mean) / std
        else:
            zscore = 0.0

        return velocity_1h, velocity_24h, zscore


class GeoTracker:
    """Track last known geolocation per customer."""

    _EARTH_RADIUS_KM = 6371.0

    def __init__(self) -> None:
        self._last: Dict[str, Tuple[float, float]] = {}

    def distance_km(self, customer_id: str, lat: float, lon: float) -> float:
        if customer_id not in self._last:
            self._last[customer_id] = (lat, lon)
            return 0.0

        prev_lat, prev_lon = self._last[customer_id]
        self._last[customer_id] = (lat, lon)

        dlat = math.radians(lat - prev_lat)
        dlon = math.radians(lon - prev_lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(prev_lat)) * math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2
        return self._EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


class FeatureEngineer:
    """Enrich a raw Transaction into an EnrichedTransaction with ML-ready features."""

    def __init__(self) -> None:
        self._velocity = VelocityTracker()
        self._geo = GeoTracker()
        self._seen_devices: Dict[str, set] = defaultdict(set)
        self._seen_ips: Dict[str, set] = defaultdict(set)
        self._merchant_risk: Dict[str, float] = {}

    def set_merchant_risk(self, merchant_id: str, score: float) -> None:
        self._merchant_risk[merchant_id] = score

    def enrich(self, txn: Transaction) -> EnrichedTransaction:
        ts = txn.timestamp
        v1h, v24h, zscore = self._velocity.record_and_count(txn.customer_id, ts, txn.amount)

        geo = txn.geolocation or {}
        dist_km = 0.0
        if geo:
            dist_km = self._geo.distance_km(txn.customer_id, geo.get("lat", 0.0), geo.get("lon", 0.0))

        device_seen = txn.device_id in self._seen_devices[txn.customer_id] if txn.device_id else True
        ip_seen = txn.ip_address in self._seen_ips[txn.customer_id] if txn.ip_address else True

        if txn.device_id:
            self._seen_devices[txn.customer_id].add(txn.device_id)
        if txn.ip_address:
            self._seen_ips[txn.customer_id].add(txn.ip_address)

        merchant_risk = self._merchant_risk.get(txn.merchant_id, 0.2)

        data = txn.model_dump()
        data.update(
            {
                "hour_of_day": ts.hour,
                "day_of_week": ts.weekday(),
                "amount_log": math.log1p(txn.amount),
                "velocity_1h": v1h,
                "velocity_24h": v24h,
                "amount_zscore": round(zscore, 4),
                "merchant_risk_score": merchant_risk,
                "device_seen_before": device_seen,
                "ip_seen_before": ip_seen,
                "geo_distance_km": round(dist_km, 2),
                "is_cross_border": dist_km > 1000,
                "customer_age_days": 0,
                "account_balance_ratio": 0.0,
            }
        )
        return EnrichedTransaction(**data)
