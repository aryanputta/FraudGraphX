"""POST /v1/feedback – Analyst feedback for RL threshold adaptation."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    alert_id: str
    predicted_fraud: bool
    actual_fraud: bool
    analyst_notes: str = ""


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(request: Request, payload: FeedbackRequest) -> None:
    deps = get_dependencies(request)
    deps.ensemble.feedback(payload.predicted_fraud, payload.actual_fraud)
    logger.info(
        "Analyst feedback received",
        alert_id=payload.alert_id,
        predicted=payload.predicted_fraud,
        actual=payload.actual_fraud,
        threshold=deps.ensemble._threshold_agent.threshold,
    )
