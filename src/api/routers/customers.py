"""Customer graph and risk endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/customers/{customer_id}/graph")
async def customer_fraud_graph(
    request: Request,
    customer_id: str,
    hops: int = Query(2, ge=1, le=4),
) -> Dict[str, Any]:
    deps = get_dependencies(request)
    if not deps.graph_builder:
        raise HTTPException(status_code=503, detail="Graph database unavailable")
    subgraph = deps.graph_builder.get_fraud_subgraph(customer_id, hops=hops)
    return subgraph.model_dump()


@router.get("/customers/{customer_id}/risk")
async def customer_risk_summary(request: Request, customer_id: str) -> Dict[str, Any]:
    deps = get_dependencies(request)
    alerts = []
    if deps.cassandra:
        alerts = deps.cassandra.get_alerts_for_customer(customer_id, limit=10)
    ring_members = []
    if deps.graph_path_explainer:
        ring_members = deps.graph_path_explainer.fraud_ring_members(customer_id)
    return {
        "customer_id": customer_id,
        "recent_alerts": alerts,
        "potential_ring_members": ring_members,
        "ring_member_count": len(ring_members),
    }
