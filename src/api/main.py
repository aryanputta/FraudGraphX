"""
FraudGraphX FastAPI application.

Endpoints:
  POST /v1/score           – Real-time transaction scoring (<150 ms)
  GET  /v1/alerts          – Recent fraud alerts
  GET  /v1/alerts/{id}     – Alert detail with SHAP + graph path
  GET  /v1/cases           – Case list
  POST /v1/cases           – Create case from alert
  PUT  /v1/cases/{id}      – Update case status
  GET  /v1/customers/{id}/graph – Customer fraud subgraph
  POST /v1/feedback        – Analyst feedback for RL threshold adaptation
  GET  /v1/benchmark       – Model benchmark results
  GET  /healthz            – Health check
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from src.api.dependencies import get_dependencies, AppDependencies
from src.api.routers import alerts, benchmark, cases, customers, feedback, scoring
from src.common.config import get_settings
from src.common.logging import configure_logging, get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Metrics ───────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("fraudgraphx_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("fraudgraphx_request_latency_ms", "Request latency in ms", buckets=[10, 25, 50, 100, 150, 250, 500, 1000])
ALERT_COUNT = Counter("fraudgraphx_alerts_total", "Total fraud alerts generated", ["risk_level"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()
    logger.info("FraudGraphX API starting up")
    deps = AppDependencies()
    await deps.initialise()
    app.state.deps = deps
    yield
    await deps.shutdown()
    logger.info("FraudGraphX API shut down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FraudGraphX",
        description="Production-grade enterprise fraud detection platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):  # type: ignore[misc]
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        REQUEST_LATENCY.observe(latency_ms)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
        return response

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Routers
    app.include_router(scoring.router, prefix="/v1", tags=["scoring"])
    app.include_router(alerts.router, prefix="/v1", tags=["alerts"])
    app.include_router(cases.router, prefix="/v1", tags=["cases"])
    app.include_router(customers.router, prefix="/v1", tags=["customers"])
    app.include_router(feedback.router, prefix="/v1", tags=["feedback"])
    app.include_router(benchmark.router, prefix="/v1", tags=["benchmark"])

    @app.get("/healthz", tags=["ops"])
    async def health() -> dict:
        return {"status": "ok", "version": "1.0.0"}

    return app


app = create_app()
