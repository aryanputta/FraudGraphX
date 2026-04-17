"""
Nightly batch risk rescoring.

Runs as a Kubernetes CronJob (02:00 UTC daily).
Re-evaluates customer risk scores based on:
  - Recent transaction patterns
  - Graph neighbourhood changes
  - Model drift detection
  - Analyst feedback incorporation
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.graph.graph_builder import GraphBuilder
from src.graph.neo4j_client import Neo4jClient
from src.models.ensemble import EnsembleScorer
from src.storage.cassandra_client import CassandraClient

logger = get_logger(__name__)
settings = get_settings()


class NightlyRescorer:
    """Batch rescoring job that updates customer risk scores in Neo4j."""

    def __init__(self) -> None:
        self._cassandra = CassandraClient()
        self._neo4j = Neo4jClient()
        self._graph = GraphBuilder(client=self._neo4j)
        self._ensemble = EnsembleScorer()

    def run(self) -> None:
        t0 = time.perf_counter()
        logger.info("Nightly rescoring started", timestamp=datetime.utcnow().isoformat())

        try:
            self._ensemble.load_models()
            customers = self._get_active_customers()
            logger.info("Customers to rescore", count=len(customers))

            updated = 0
            for customer_id in customers:
                new_score = self._compute_risk_score(customer_id)
                self._graph.update_customer_risk(customer_id, new_score)
                updated += 1

            # Detect money laundering chains
            chains = self._graph.find_money_laundering_chains()
            logger.info("Money laundering chains detected", count=len(chains))

            elapsed = (time.perf_counter() - t0) / 60
            logger.info("Nightly rescoring complete", customers_rescored=updated, elapsed_minutes=round(elapsed, 2))
        except Exception as exc:
            logger.error("Nightly rescoring failed", error=str(exc))
            raise

    def _get_active_customers(self) -> List[str]:
        """Fetch customers with transactions in the past 7 days."""
        rows = self._neo4j.run(
            """
            MATCH (c:Customer)-[:OWNS]->(a:Account)-[:MADE_TRANSACTION]->(t:Transaction)
            WHERE t.timestamp >= datetime() - duration({days: 7})
            RETURN DISTINCT c.customer_id AS customer_id
            LIMIT 50000
            """
        )
        return [r["customer_id"] for r in rows]

    def _compute_risk_score(self, customer_id: str) -> float:
        """Combine model score + graph neighbourhood suspicion."""
        subgraph = self._graph.get_fraud_subgraph(customer_id, hops=2)
        graph_score = subgraph.suspicion_score

        # Use alert history as a signal
        alerts = self._cassandra.get_alerts_for_customer(customer_id, limit=30)
        if alerts:
            recent_scores = [a.get("ensemble_score", 0.0) for a in alerts[:10]]
            alert_score = sum(recent_scores) / len(recent_scores)
        else:
            alert_score = 0.0

        # Weighted combination
        risk = 0.4 * graph_score + 0.6 * alert_score
        return round(min(1.0, risk), 4)


def main() -> None:
    NightlyRescorer().run()


if __name__ == "__main__":
    main()
