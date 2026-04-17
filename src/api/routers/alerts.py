"""GET /v1/alerts – Fraud alert retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/alerts", response_model=List[Dict[str, Any]])
async def list_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    deps = get_dependencies(request)
    if not deps.cassandra:
        return []
    return deps.cassandra.get_recent_alerts(limit=limit)


@router.get("/alerts/{alert_id}", response_model=Dict[str, Any])
async def get_alert(request: Request, alert_id: str) -> Dict[str, Any]:
    deps = get_dependencies(request)
    if not deps.cassandra:
        raise HTTPException(status_code=503, detail="Storage unavailable")
    alert = deps.cassandra.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.get("/customers/{customer_id}/alerts", response_model=List[Dict[str, Any]])
async def customer_alerts(
    request: Request,
    customer_id: str,
    limit: int = Query(20, ge=1, le=200),
) -> List[Dict[str, Any]]:
    deps = get_dependencies(request)
    if not deps.cassandra:
        return []
    return deps.cassandra.get_alerts_for_customer(customer_id, limit=limit)
