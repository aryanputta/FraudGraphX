"""Neo4j driver wrapper with connection pooling and schema initialisation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from neo4j import GraphDatabase, Session
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Cypher for schema bootstrap
SCHEMA_CYPHER = [
    # Constraints
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
    "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
    "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE",
    "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
    "CREATE CONSTRAINT ip_address IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.address IS UNIQUE",
    "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
    # Indexes
    "CREATE INDEX customer_risk IF NOT EXISTS FOR (c:Customer) ON (c.risk_score)",
    "CREATE INDEX transaction_ts IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
    "CREATE INDEX transaction_fraud IF NOT EXISTS FOR (t:Transaction) ON (t.is_fraud_predicted)",
]


class Neo4jClient:
    """Thread-safe Neo4j client with retry logic."""

    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )

    def initialise_schema(self) -> None:
        with self._driver.session(database=settings.neo4j_database) as session:
            for cypher in SCHEMA_CYPHER:
                try:
                    session.run(cypher)
                except Exception as exc:
                    logger.warning("Schema statement warning", cypher=cypher[:60], error=str(exc))
        logger.info("Neo4j schema initialised")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self._driver.session(database=settings.neo4j_database) as s:
            yield s

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def run(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        with self.session() as s:
            result = s.run(cypher, **params)
            return [dict(r) for r in result]

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
