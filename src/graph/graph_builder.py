"""
Build and maintain the customer/device/merchant/IP graph in Neo4j.
Called after each transaction is processed.
"""

from __future__ import annotations

from typing import List, Optional

from src.common.logging import get_logger
from src.common.schemas import EnrichedTransaction, FraudSubgraph, GraphEdge, GraphNode
from src.graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphBuilder:
    """Upserts graph nodes and edges for every processed transaction."""

    def __init__(self, client: Optional[Neo4jClient] = None) -> None:
        self._client = client or Neo4jClient()

    def upsert_transaction(self, txn: EnrichedTransaction) -> None:
        """Merge all nodes and relationships for a single transaction."""
        cypher = """
        MERGE (c:Customer {customer_id: $customer_id})
        ON CREATE SET c.risk_score = 0.0, c.created_at = datetime()
        ON MATCH SET c.last_seen = datetime()

        MERGE (a:Account {account_id: $account_id})
        ON CREATE SET a.customer_id = $customer_id, a.created_at = datetime()

        MERGE (m:Merchant {merchant_id: $merchant_id})
        ON CREATE SET m.category = $merchant_category, m.risk_score = $merchant_risk_score

        MERGE (t:Transaction {transaction_id: $transaction_id})
        SET t.amount = $amount,
            t.timestamp = $timestamp,
            t.fraud_score = $fraud_score,
            t.is_fraud_predicted = $is_fraud_predicted,
            t.channel = $channel

        MERGE (c)-[:OWNS]->(a)
        MERGE (a)-[:MADE_TRANSACTION]->(t)
        MERGE (t)-[:AT_MERCHANT]->(m)
        """
        params = {
            "customer_id": txn.customer_id,
            "account_id": txn.account_id,
            "merchant_id": txn.merchant_id,
            "merchant_category": txn.merchant_category,
            "merchant_risk_score": txn.merchant_risk_score,
            "transaction_id": txn.transaction_id,
            "amount": txn.amount,
            "timestamp": txn.timestamp.isoformat(),
            "fraud_score": getattr(txn, "fraud_score", 0.0),
            "is_fraud_predicted": getattr(txn, "is_fraud_predicted", False),
            "channel": txn.channel,
        }
        self._client.run(cypher, **params)

        if txn.device_id:
            self._client.run(
                """
                MERGE (d:Device {device_id: $device_id})
                MERGE (c:Customer {customer_id: $customer_id})
                MERGE (t:Transaction {transaction_id: $transaction_id})
                MERGE (c)-[:USES_DEVICE]->(d)
                MERGE (t)-[:FROM_DEVICE]->(d)
                """,
                device_id=txn.device_id,
                customer_id=txn.customer_id,
                transaction_id=txn.transaction_id,
            )

        if txn.ip_address:
            self._client.run(
                """
                MERGE (ip:IPAddress {address: $ip_address})
                MERGE (c:Customer {customer_id: $customer_id})
                MERGE (t:Transaction {transaction_id: $transaction_id})
                MERGE (c)-[:ACCESSED_FROM_IP]->(ip)
                MERGE (t)-[:FROM_IP]->(ip)
                """,
                ip_address=txn.ip_address,
                customer_id=txn.customer_id,
                transaction_id=txn.transaction_id,
            )

    def update_customer_risk(self, customer_id: str, risk_score: float) -> None:
        self._client.run(
            "MATCH (c:Customer {customer_id: $cid}) SET c.risk_score = $score",
            cid=customer_id,
            score=risk_score,
        )

    def get_fraud_subgraph(self, customer_id: str, hops: int = 2) -> FraudSubgraph:
        """Return neighbourhood graph for a given customer (fraud ring detection)."""
        cypher = f"""
        MATCH path = (c:Customer {{customer_id: $cid}})-[*1..{hops}]-(neighbour)
        WHERE neighbour.risk_score IS NOT NULL OR neighbour.is_fraud_predicted = true
        RETURN DISTINCT id(startNode(relationships(path)[0])) AS src_id,
               id(endNode(relationships(path)[size(relationships(path))-1])) AS dst_id,
               [n IN nodes(path) | {{id: id(n), labels: labels(n), props: properties(n)}}] AS node_list,
               [r IN relationships(path) | {{type: type(r), src: id(startNode(r)), dst: id(endNode(r))}}] AS rel_list
        LIMIT 200
        """
        rows = self._client.run(cypher, cid=customer_id)

        seen_nodes: dict = {}
        edges: List[GraphEdge] = []

        for row in rows:
            for n in row.get("node_list", []):
                nid = str(n["id"])
                if nid not in seen_nodes:
                    seen_nodes[nid] = GraphNode(
                        node_id=nid,
                        node_type=(n.get("labels") or ["unknown"])[0],
                        properties=n.get("props", {}),
                        risk_score=n.get("props", {}).get("risk_score", 0.0),
                    )
            for r in row.get("rel_list", []):
                edges.append(
                    GraphEdge(
                        source_id=str(r["src"]),
                        target_id=str(r["dst"]),
                        relationship=r["type"],
                    )
                )

        suspicion = sum(n.risk_score for n in seen_nodes.values()) / max(len(seen_nodes), 1)
        return FraudSubgraph(
            nodes=list(seen_nodes.values()),
            edges=edges,
            suspicion_score=round(suspicion, 4),
            fraud_ring_detected=len(seen_nodes) >= 10 and suspicion > 0.5,
        )

    def find_money_laundering_chains(self, min_amount: float = 10_000) -> List[dict]:
        """Detect rapid large-inbound / large-outbound sequences (layering)."""
        cypher = """
        MATCH (a:Account)-[:MADE_TRANSACTION]->(t1:Transaction)
        WHERE t1.amount >= $min_amount
        WITH a, t1
        MATCH (a)-[:MADE_TRANSACTION]->(t2:Transaction)
        WHERE t2.amount >= $min_amount * 0.7
          AND t2.timestamp > t1.timestamp
          AND duration.between(datetime(t1.timestamp), datetime(t2.timestamp)).minutes < 60
        RETURN a.account_id AS account_id,
               t1.transaction_id AS inbound_txn,
               t2.transaction_id AS outbound_txn,
               t1.amount AS inbound_amount,
               t2.amount AS outbound_amount
        LIMIT 100
        """
        return self._client.run(cypher, min_amount=min_amount)
