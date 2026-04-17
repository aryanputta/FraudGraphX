"""Graph path reasoning: find the shortest suspicious path in the fraud graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common.logging import get_logger
from src.graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphPathExplainer:
    """Explain fraud decisions via graph path analysis in Neo4j."""

    def __init__(self, client: Optional[Neo4jClient] = None) -> None:
        self._client = client or Neo4jClient()

    def suspicious_path(self, customer_id: str, transaction_id: str) -> List[str]:
        """Return a human-readable path of suspicious connections."""
        cypher = """
        MATCH path = (c:Customer {customer_id: $cid})-[*1..3]-(suspicious)
        WHERE suspicious.risk_score > 0.6
           OR suspicious.is_fraud_predicted = true
        RETURN [n IN nodes(path) | COALESCE(n.customer_id, n.merchant_id, n.device_id, n.address, 'node')]
                 AS path_labels,
               length(path) AS hops
        ORDER BY hops ASC
        LIMIT 1
        """
        rows = self._client.run(cypher, cid=customer_id)
        if not rows:
            return [f"Customer {customer_id[:8]}… → Transaction {transaction_id[:8]}…"]
        return rows[0].get("path_labels", [])

    def shared_device_fraud(self, device_id: str) -> List[Dict[str, Any]]:
        """Find other customers who used the same device and had fraud."""
        cypher = """
        MATCH (d:Device {device_id: $did})<-[:USES_DEVICE]-(c:Customer)
        OPTIONAL MATCH (c)-[:OWNS]->(a:Account)-[:MADE_TRANSACTION]->(t:Transaction)
        WHERE t.is_fraud_predicted = true
        RETURN c.customer_id AS customer_id, count(t) AS fraud_txn_count
        ORDER BY fraud_txn_count DESC
        LIMIT 20
        """
        return self._client.run(cypher, did=device_id)

    def fraud_ring_members(self, customer_id: str) -> List[str]:
        """Identify customers in the same suspected fraud ring."""
        cypher = """
        MATCH (c:Customer {customer_id: $cid})-[:USES_DEVICE]->(d:Device)<-[:USES_DEVICE]-(peer:Customer)
        WHERE peer.customer_id <> $cid
        RETURN DISTINCT peer.customer_id AS peer_id
        LIMIT 50
        """
        rows = self._client.run(cypher, cid=customer_id)
        return [r["peer_id"] for r in rows]
