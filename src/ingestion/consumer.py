"""Kafka consumer – reads raw transactions and dispatches to processing pipeline."""

import json
import signal
import time
from typing import Callable, List

from confluent_kafka import Consumer, KafkaError, KafkaException, Message

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import Transaction

logger = get_logger(__name__)
settings = get_settings()


def _build_consumer_config() -> dict:
    return {
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "security.protocol": settings.kafka_security_protocol,
        "group.id": settings.kafka_consumer_group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "max.poll.interval.ms": 300_000,
        "session.timeout.ms": 45_000,
        "fetch.min.bytes": 1,
        "fetch.wait.max.ms": 500,
    }


class TransactionConsumer:
    """Confluent-Kafka consumer with manual commit and graceful shutdown."""

    def __init__(self, topics: List[str] | None = None) -> None:
        self._consumer = Consumer(_build_consumer_config())
        self._topics = topics or [settings.kafka_topic_transactions]
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, *_: object) -> None:
        logger.info("Shutdown signal received – stopping consumer")
        self._running = False

    def _deserialize(self, msg: Message) -> Transaction | None:
        try:
            data = json.loads(msg.value().decode("utf-8"))
            return Transaction(**data)
        except Exception as exc:
            logger.error("Failed to deserialize message", error=str(exc))
            return None

    def consume(self, handler: Callable[[Transaction], None], batch_size: int = 100) -> None:
        self._consumer.subscribe(self._topics)
        logger.info("Consumer subscribed", topics=self._topics)

        try:
            while self._running:
                messages = self._consumer.consume(batch_size, timeout=1.0)
                if not messages:
                    continue

                for msg in messages:
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logger.error("Kafka error", error=msg.error())
                        continue

                    txn = self._deserialize(msg)
                    if txn:
                        try:
                            handler(txn)
                        except Exception as exc:
                            logger.error("Handler failed", txn_id=txn.transaction_id, error=str(exc))

                self._consumer.commit(asynchronous=False)

        finally:
            self._consumer.close()
            logger.info("Consumer closed")
