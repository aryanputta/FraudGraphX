"""Kafka transaction producer – publishes raw transactions to Kafka."""

import json
import time
from typing import Callable, Optional

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Transaction

logger = get_logger(__name__)
settings = get_settings()


def _build_producer_config() -> dict:
    return {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "security.protocol": settings.kafka_security_protocol,
        "acks": "all",
        "retries": 5,
        "retry.backoff.ms": 300,
        "compression.type": "lz4",
        "linger.ms": 5,
        "batch.size": 65536,
    }


class TransactionProducer:
    """Thread-safe Kafka producer for fraud-detection transactions."""

    def __init__(self) -> None:
        self._producer = Producer(_build_producer_config())
        self._topic = settings.kafka_topic_transactions

    def _delivery_report(self, err: Optional[Exception], msg: object) -> None:
        if err:
            logger.error("Kafka delivery failed", error=str(err))
        else:
            logger.debug("Delivered message", topic=msg.topic(), partition=msg.partition())

    def publish(self, txn: Transaction) -> None:
        payload = txn.model_dump_json().encode("utf-8")
        self._producer.produce(
            topic=self._topic,
            key=txn.customer_id.encode("utf-8"),
            value=payload,
            callback=self._delivery_report,
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        self._producer.flush(timeout)

    def __enter__(self) -> "TransactionProducer":
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()


def ensure_topics_exist() -> None:
    """Create Kafka topics if they do not already exist."""
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    topics = [
        NewTopic(settings.kafka_topic_transactions, num_partitions=12, replication_factor=3),
        NewTopic(settings.kafka_topic_enriched, num_partitions=12, replication_factor=3),
        NewTopic(settings.kafka_topic_alerts, num_partitions=4, replication_factor=3),
    ]
    futures = admin.create_topics(topics)
    for topic, future in futures.items():
        try:
            future.result()
            logger.info("Topic created", topic=topic)
        except Exception as exc:
            if "already exists" not in str(exc):
                logger.warning("Topic creation warning", topic=topic, error=str(exc))
