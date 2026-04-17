"""GET /v1/benchmark – Model benchmark results."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.api.dependencies import get_dependencies
from src.common.logging import get_logger
from src.models.trainer import train_all  # imported at module level so tests can patch it

logger = get_logger(__name__)
router = APIRouter()

_cached_results: Dict[str, Any] = {}


@router.get("/benchmark", response_model=List[Dict[str, Any]])
async def get_benchmark(request: Request) -> List[Dict[str, Any]]:
    if _cached_results:
        return list(_cached_results.values())
    return [
        {
            "model_name": "logistic_regression",
            "note": "Run POST /v1/benchmark/run to train and benchmark all models",
        }
    ]


@router.post("/benchmark/run")
async def run_benchmark(request: Request, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Trigger full model training and benchmarking in background."""
    def _train() -> None:
        results = train_all()
        for name, result in results.items():
            _cached_results[name] = result.model_dump()
        logger.info("Benchmark complete", models=list(results.keys()))

    background_tasks.add_task(_train)
    return {"status": "training started – check /v1/benchmark for results"}
