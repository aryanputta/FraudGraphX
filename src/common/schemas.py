"""Shared Pydantic schemas used across all services."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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

class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    amount: float
    currency: str = "USD"
    customer_id: str
    account_id: str
    merchant_id: str
    merchant_category: str = ""
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    geolocation: Optional[Dict[str, float]] = None  # {lat, lon}
    card_type: Optional[str] = None
    is_online: bool = True
    channel: str = "web"
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
