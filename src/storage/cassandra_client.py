"""Cassandra storage client for transactions, alerts, and cases."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from cassandra.cluster import Cluster, Session
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy
from cassandra.query import BatchStatement, SimpleStatement

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import FraudAlert, FraudCase

logger = get_logger(__name__)
settings = get_settings()

# DDL statements
SCHEMA_DDL = [
    f"""
    CREATE KEYSPACE IF NOT EXISTS {settings.cassandra_keyspace}
    WITH replication = {{
        'class': 'NetworkTopologyStrategy',
        'datacenter1': {settings.cassandra_replication_factor}
    }}
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {settings.cassandra_keyspace}.enriched_transactions (
        customer_id   TEXT,
        year_month    TEXT,
        timestamp     TIMESTAMP,
        transaction_id UUID,
        amount        DOUBLE,
        currency      TEXT,
        merchant_id   TEXT,
        merchant_category TEXT,
        fraud_score   DOUBLE,
        is_fraud_predicted BOOLEAN,
        channel       TEXT,
        metadata      MAP<TEXT,TEXT>,
        PRIMARY KEY ((customer_id, year_month), timestamp, transaction_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC)
      AND default_time_to_live = 7776000
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {settings.cassandra_keyspace}.fraud_alerts (
        alert_id      UUID,
        customer_id   TEXT,
        transaction_id TEXT,
        timestamp     TIMESTAMP,
        ensemble_score DOUBLE,
        risk_level    TEXT,
        fraud_type    TEXT,
        action        TEXT,
        latency_ms    DOUBLE,
        shap_values   TEXT,
        graph_path    TEXT,
        investigator_notes TEXT,
        PRIMARY KEY (alert_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {settings.cassandra_keyspace}.fraud_alerts_by_customer (
        customer_id   TEXT,
        timestamp     TIMESTAMP,
        alert_id      UUID,
        risk_level    TEXT,
        ensemble_score DOUBLE,
        PRIMARY KEY (customer_id, timestamp, alert_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC)
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {settings.cassandra_keyspace}.fraud_cases (
        case_id       UUID,
        alert_id      TEXT,
        customer_id   TEXT,
        status        TEXT,
        risk_level    TEXT,
        fraud_type    TEXT,
        opened_at     TIMESTAMP,
        updated_at    TIMESTAMP,
        assigned_to   TEXT,
        fraud_amount  DOUBLE,
        notes         LIST<TEXT>,
        PRIMARY KEY (case_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {settings.cassandra_keyspace}.model_metrics (
        model_name    TEXT,
        eval_date     TEXT,
        precision     DOUBLE,
        recall        DOUBLE,
        f1            DOUBLE,
        roc_auc       DOUBLE,
        avg_latency_ms DOUBLE,
        threshold     DOUBLE,
        PRIMARY KEY (model_name, eval_date)
    ) WITH CLUSTERING ORDER BY (eval_date DESC)
    """,
]


class CassandraClient:
    """Cassandra client with prepared statements and schema bootstrapping."""

    def __init__(self) -> None:
        self._cluster = Cluster(
            contact_points=settings.cassandra_contact_points,
            port=settings.cassandra_port,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc="datacenter1"),
            default_retry_policy=RetryPolicy(),
        )
        self._session: Optional[Session] = None

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = self._cluster.connect()
        return self._session

    def initialise_schema(self) -> None:
        for ddl in SCHEMA_DDL:
            self.session.execute(SimpleStatement(ddl.strip()))
        logger.info("Cassandra schema initialised")

    def save_alert(self, alert: FraudAlert) -> None:
        import json

        ks = settings.cassandra_keyspace
        self.session.execute(
            f"""
            INSERT INTO {ks}.fraud_alerts
            (alert_id, customer_id, transaction_id, timestamp, ensemble_score,
             risk_level, fraud_type, action, latency_ms, shap_values, graph_path, investigator_notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                uuid.UUID(alert.alert_id),
                alert.customer_id,
                alert.transaction_id,
                alert.timestamp,
                alert.ensemble_score,
                alert.risk_level.value,
                alert.fraud_type.value,
                alert.action,
                alert.latency_ms,
                json.dumps(alert.shap_values) if alert.shap_values else None,
                json.dumps(alert.graph_path) if alert.graph_path else None,
                alert.investigator_notes,
            ),
        )
        self.session.execute(
            f"""
            INSERT INTO {ks}.fraud_alerts_by_customer
            (customer_id, timestamp, alert_id, risk_level, ensemble_score)
            VALUES (%s,%s,%s,%s,%s)
            """,
            (alert.customer_id, alert.timestamp, uuid.UUID(alert.alert_id), alert.risk_level.value, alert.ensemble_score),
        )

    def save_case(self, case: FraudCase) -> None:
        ks = settings.cassandra_keyspace
        self.session.execute(
            f"""
            INSERT INTO {ks}.fraud_cases
            (case_id, alert_id, customer_id, status, risk_level, fraud_type,
             opened_at, updated_at, assigned_to, fraud_amount, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                uuid.UUID(case.case_id),
                case.alert_id,
                case.customer_id,
                case.status.value,
                case.risk_level.value,
                case.fraud_type.value,
                case.opened_at,
                case.updated_at,
                case.assigned_to,
                case.fraud_amount,
                case.notes,
            ),
        )

    def get_alerts_for_customer(self, customer_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        ks = settings.cassandra_keyspace
        rows = self.session.execute(
            f"SELECT * FROM {ks}.fraud_alerts_by_customer WHERE customer_id=%s LIMIT %s",
            (customer_id, limit),
        )
        return [dict(r._asdict()) for r in rows]

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        ks = settings.cassandra_keyspace
        rows = self.session.execute(
            f"SELECT * FROM {ks}.fraud_alerts WHERE alert_id=%s",
            (uuid.UUID(alert_id),),
        )
        row = rows.one()
        return dict(row._asdict()) if row else None

    def get_recent_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        ks = settings.cassandra_keyspace
        rows = self.session.execute(
            f"SELECT * FROM {ks}.fraud_alerts LIMIT %s ALLOW FILTERING",
            (limit,),
        )
        return [dict(r._asdict()) for r in rows]

    def close(self) -> None:
        if self._session:
            self._session.shutdown()
        self._cluster.shutdown()

    def __enter__(self) -> "CassandraClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
