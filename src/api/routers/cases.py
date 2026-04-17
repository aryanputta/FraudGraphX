"""Case management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger
from src.common.schemas import CaseStatus, FraudCase, FraudType, RiskLevel

logger = get_logger(__name__)
router = APIRouter()


class CreateCaseRequest(BaseModel):
    alert_id: str
    transaction_id: str
    customer_id: str
    risk_level: RiskLevel = RiskLevel.HIGH
    fraud_type: FraudType = FraudType.UNKNOWN
    assigned_to: Optional[str] = None
    fraud_amount: float = 0.0
    notes: List[str] = []


class UpdateCaseRequest(BaseModel):
    status: Optional[CaseStatus] = None
    assigned_to: Optional[str] = None
    notes: Optional[List[str]] = None
    resolution: Optional[str] = None
    fraud_type: Optional[FraudType] = None


@router.post("/cases", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_case(request: Request, payload: CreateCaseRequest) -> Dict[str, Any]:
    deps = get_dependencies(request)
    case = FraudCase(
        case_id=str(uuid.uuid4()),
        alert_id=payload.alert_id,
        transaction_id=payload.transaction_id,
        customer_id=payload.customer_id,
        risk_level=payload.risk_level,
        fraud_type=payload.fraud_type,
        assigned_to=payload.assigned_to,
        fraud_amount=payload.fraud_amount,
        notes=payload.notes,
    )
    if deps.cassandra:
        deps.cassandra.save_case(case)
    return case.model_dump()


@router.put("/cases/{case_id}", response_model=Dict[str, Any])
async def update_case(request: Request, case_id: str, payload: UpdateCaseRequest) -> Dict[str, Any]:
    # In production this would load the case from Cassandra and patch it.
    # For this implementation we return the update payload as confirmation.
    return {"case_id": case_id, "updated": payload.model_dump(exclude_none=True), "updated_at": datetime.utcnow().isoformat()}
