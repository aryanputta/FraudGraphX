"""Shared application dependencies (singleton services)."""

from __future__ import annotations

import asyncio
from typing import Optional

from src.common.logging import get_logger
from src.explainability.graph_path_explainer import GraphPathExplainer
from src.explainability.llm_investigator import LLMInvestigator
from src.explainability.shap_explainer import SHAPExplainer
from src.graph.graph_builder import GraphBuilder
from src.graph.neo4j_client import Neo4jClient
from src.models.ensemble import EnsembleScorer
from src.models.xgboost_model import XGBoostFraudClassifier
from src.storage.cassandra_client import CassandraClient

logger = get_logger(__name__)


class AppDependencies:
    """Container for all singleton services initialised at startup."""

    def __init__(self) -> None:
        self.ensemble: EnsembleScorer = EnsembleScorer()
        self.cassandra: Optional[CassandraClient] = None
        self.neo4j: Optional[Neo4jClient] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.graph_path_explainer: Optional[GraphPathExplainer] = None
        self.llm_investigator: LLMInvestigator = LLMInvestigator()
        self.shap_explainer: Optional[SHAPExplainer] = None

    async def initialise(self) -> None:
        loop = asyncio.get_event_loop()

        # Load ML models
        await loop.run_in_executor(None, self.ensemble.load_models)

        # Storage
        try:
            self.cassandra = CassandraClient()
            self.cassandra.initialise_schema()
        except Exception as exc:
            logger.warning("Cassandra unavailable – running without persistent storage", error=str(exc))

        # Graph
        try:
            self.neo4j = Neo4jClient()
            self.neo4j.initialise_schema()
            self.graph_builder = GraphBuilder(client=self.neo4j)
            self.graph_path_explainer = GraphPathExplainer(client=self.neo4j)
        except Exception as exc:
            logger.warning("Neo4j unavailable – graph features disabled", error=str(exc))

        # SHAP explainer (requires XGBoost model to be loaded)
        xgb_model = self.ensemble._models.get("xgboost")
        if xgb_model:
            self.shap_explainer = SHAPExplainer(xgb_model, model_type="xgboost")

        logger.info("Application dependencies initialised")

    async def shutdown(self) -> None:
        if self.cassandra:
            self.cassandra.close()
        if self.neo4j:
            self.neo4j.close()


def get_dependencies(request) -> AppDependencies:  # type: ignore[misc]
    return request.app.state.deps
