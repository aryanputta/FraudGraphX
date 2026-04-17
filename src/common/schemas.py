"""Shared Pydantic schemas used across all services."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import re

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudType(str, Enum):
    CARD_NOT_PRESENT = "card_not_present"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    ACCOUNT_TAKEOVER = "account_takeover"
    MULE_ACCOUNT = "mule_account"
    MONEY_LAUNDERING = "money_laundering"
    FRAUD_RING = "fraud_ring"
    UNKNOWN = "unknown"


# ── Transaction ────────────────────────────────────────────────────────────────

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _validate_safe_id(v: str) -> str:
    """Reject IDs that could be used for injection or log-forging."""
    if not _SAFE_ID_RE.match(v):
        raise ValueError("ID must be 1-128 alphanumeric/dash/underscore characters")
    return v


class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    amount: float = Field(gt=0, le=10_000_000, description="Transaction amount in currency units")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    customer_id: str = Field(min_length=1, max_length=128)
    account_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=128)
    merchant_category: str = Field(default="", max_length=64)
    device_id: Optional[str] = Field(default=None, max_length=128)
    ip_address: Optional[str] = Field(default=None, max_length=45)  # max IPv6 length
    geolocation: Optional[Dict[str, float]] = None  # {lat, lon}
    card_type: Optional[str] = Field(default=None, max_length=32)
    is_online: bool = True
    channel: str = Field(default="web", max_length=32)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("customer_id", "account_id", "merchant_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        return _validate_safe_id(v)

    @field_validator("ip_address")
    @classmethod
    def validate_ip(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Basic sanity – full validation handled at network boundary
        if not re.match(r"^[0-9a-fA-F.:]{2,45}$", v):
            raise ValueError("Invalid IP address format")
        return v

    @field_validator("geolocation")
    @classmethod
    def validate_geolocation(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        lat = v.get("lat", 0.0)
        lon = v.get("lon", 0.0)
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Geolocation out of valid range")
        return v


class EnrichedTransaction(Transaction):
    """Transaction after feature engineering."""
    hour_of_day: int = 0
    day_of_week: int = 0
    amount_log: float = 0.0
    velocity_1h: int = 0          # txns in last 1h for this customer
    velocity_24h: int = 0
    amount_zscore: float = 0.0    # relative to customer mean
    merchant_risk_score: float = 0.0
    device_seen_before: bool = True
    ip_seen_before: bool = True
    geo_distance_km: float = 0.0  # from last txn
    is_cross_border: bool = False
    customer_age_days: int = 0
    account_balance_ratio: float = 0.0


# ── Fraud Score & Alert ────────────────────────────────────────────────────────

class ModelScore(BaseModel):
    model_name: str
    score: float                  # 0-1 probability of fraud
    features_used: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class FraudAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transaction_id: str
    customer_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ensemble_score: float
    risk_level: RiskLevel
    fraud_type: FraudType = FraudType.UNKNOWN
    model_scores: List[ModelScore] = Field(default_factory=list)
    graph_path: Optional[List[str]] = None   # suspicious graph path
    shap_values: Optional[Dict[str, float]] = None
    action: str = "review"                   # block | review | pass
    latency_ms: float = 0.0
    investigator_notes: Optional[str] = None


# ── Case Management ────────────────────────────────────────────────────────────

class CaseStatus(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    CONFIRMED_FRAUD = "confirmed_fraud"
    CLEARED = "cleared"
    ESCALATED = "escalated"


class FraudCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_id: str
    transaction_id: str
    customer_id: str
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: CaseStatus = CaseStatus.OPEN
    assigned_to: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.HIGH
    fraud_type: FraudType = FraudType.UNKNOWN
    notes: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    resolution: Optional[str] = None
    fraud_amount: float = 0.0


# ── Graph Entities ─────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    node_id: str
    node_type: str               # customer | device | merchant | account | ip
    properties: Dict[str, Any] = Field(default_factory=dict)
    risk_score: float = 0.0


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str            # MADE_TRANSACTION | USES_DEVICE | ACCESSED_FROM_IP
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class FraudSubgraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    community_id: Optional[str] = None
    suspicion_score: float = 0.0
    fraud_ring_detected: bool = False


# ── Benchmark ─────────────────────────────────────────────────────────────────

class BenchmarkResult(BaseModel):
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_latency_ms: float
    p99_latency_ms: float
    false_positive_rate: float
    fraud_dollars_prevented: float
    threshold: float
