"""
POST /v1/score – Real-time transaction scoring.
Target: <150ms end-to-end latency.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger
from src.common.schemas import EnrichedTransaction, FraudAlert, FraudType, RiskLevel, Transaction
from src.ingestion.feature_engineering import FeatureEngineer

logger = get_logger(__name__)
router = APIRouter()

# Module-level feature engineer (stateful – keeps velocity/geo history)
_feature_engineer = FeatureEngineer()


class ScoreRequest(BaseModel):
    transaction: Transaction
    explain: bool = False      # include SHAP values
    graph_path: bool = False   # include graph path reasoning


class ScoreResponse(BaseModel):
    alert_id: str
    transaction_id: str
    ensemble_score: float
    risk_level: str
    action: str
    latency_ms: float
    model_scores: list = Field(default_factory=list)
    shap_values: Optional[dict] = None
    graph_path: Optional[List[str]] = None
    investigator_notes: Optional[str] = None


async def _persist_alert(alert: FraudAlert, deps) -> None:  # type: ignore[misc]
    """Background task: save alert to Cassandra + upsert graph."""
    try:
        if deps.cassandra:
            deps.cassandra.save_alert(alert)
    except Exception as exc:
        logger.error("Failed to persist alert", error=str(exc))


@router.post("/score", response_model=ScoreResponse, status_code=status.HTTP_200_OK)
async def score_transaction(
    request: Request,
    payload: ScoreRequest,
    background_tasks: BackgroundTasks,
) -> ScoreResponse:
    t0 = time.perf_counter()
    deps = get_dependencies(request)
    txn = payload.transaction

    # Feature engineering
    enriched: EnrichedTransaction = _feature_engineer.enrich(txn)
    features_df = pd.DataFrame([enriched.model_dump()])

    # Ensemble scoring
    result = deps.ensemble.score(features_df)
    total_latency = (time.perf_counter() - t0) * 1000

    if total_latency > 150:
        logger.warning("Latency SLA breach", latency_ms=total_latency, transaction_id=txn.transaction_id)

    alert = FraudAlert(
        alert_id=str(uuid.uuid4()),
        transaction_id=txn.transaction_id,
        customer_id=txn.customer_id,
        ensemble_score=result.ensemble_score,
        risk_level=result.risk_level,
        model_scores=result.model_scores,
        action=result.action,
        latency_ms=round(total_latency, 2),
    )

    # Optional: SHAP explanation
    shap_dict = None
    if payload.explain and deps.shap_explainer:
        shap_list = deps.shap_explainer.explain(features_df)
        if shap_list:
            shap_dict = shap_list[0]
            alert.shap_values = shap_dict

    # Optional: graph path
    graph_path_list = None
    if payload.graph_path and deps.graph_path_explainer:
        graph_path_list = deps.graph_path_explainer.suspicious_path(txn.customer_id, txn.transaction_id)
        alert.graph_path = graph_path_list

    # LLM notes for high-risk alerts
    if result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and deps.llm_investigator:
        top_shap = deps.shap_explainer.top_features(shap_dict) if shap_dict and deps.shap_explainer else []
        notes = deps.llm_investigator.generate_notes(alert, top_shap, graph_path_list or [])
        alert.investigator_notes = notes

    background_tasks.add_task(_persist_alert, alert, deps)

    return ScoreResponse(
        alert_id=alert.alert_id,
        transaction_id=txn.transaction_id,
        ensemble_score=alert.ensemble_score,
        risk_level=alert.risk_level.value,
        action=alert.action,
        latency_ms=alert.latency_ms,
        model_scores=[s.model_dump() for s in alert.model_scores],
        shap_values=alert.shap_values,
        graph_path=alert.graph_path,
        investigator_notes=alert.investigator_notes,
    )


@router.post("/score/batch", response_model=List[ScoreResponse])
async def score_batch(
    request: Request,
    transactions: List[Transaction],
    background_tasks: BackgroundTasks,
) -> List[ScoreResponse]:
    """Score up to 100 transactions in a single call."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100 transactions")
    results = []
    for txn in transactions:
        resp = await score_transaction(request, ScoreRequest(transaction=txn), background_tasks)
        results.append(resp)
    return results
