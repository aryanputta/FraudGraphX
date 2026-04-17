"""
Alert dispatcher – consumes fraud alerts from Kafka and dispatches
to downstream systems (webhook, PagerDuty, Slack, etc.) under 150 ms.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
from confluent_kafka import Consumer

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class AlertDispatcher:
    """Consume fraud.alerts topic and dispatch with <150ms latency."""

    def __init__(self) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": f"{settings.kafka_consumer_group}-alert-dispatcher",
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "fetch.wait.max.ms": 5,   # aggressive polling for low latency
                "fetch.min.bytes": 1,
            }
        )
        self._handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._http = httpx.AsyncClient(timeout=5.0)

    def add_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._handlers.append(handler)

    async def run(self) -> None:
        self._consumer.subscribe([settings.kafka_topic_alerts])
        logger.info("Alert dispatcher running", topic=settings.kafka_topic_alerts)

        while True:
            msg = self._consumer.poll(timeout=0.01)   # 10ms poll
            if msg is None:
                await asyncio.sleep(0.001)
                continue
            if msg.error():
                logger.error("Kafka error", error=str(msg.error()))
                continue

            t0 = time.perf_counter()
            try:
                alert = json.loads(msg.value().decode("utf-8"))
                for handler in self._handlers:
                    handler(alert)
                latency_ms = (time.perf_counter() - t0) * 1000
                if latency_ms > settings.alert_latency_target_ms:
                    logger.warning("Alert SLA breach", latency_ms=latency_ms)
                else:
                    logger.debug("Alert dispatched", latency_ms=latency_ms, txn_id=alert.get("transaction_id"))
            except Exception as exc:
                logger.error("Alert dispatch error", error=str(exc))

    async def webhook_notify(self, url: str, alert: Dict[str, Any]) -> None:
        try:
            await self._http.post(url, json=alert)
        except Exception as exc:
            logger.error("Webhook notification failed", url=url, error=str(exc))


def log_alert_handler(alert: Dict[str, Any]) -> None:
    logger.warning(
        "FRAUD ALERT",
        transaction_id=alert.get("transaction_id"),
        customer_id=alert.get("customer_id"),
        fraud_score=alert.get("fraud_score"),
    )


async def run_dispatcher() -> None:
    dispatcher = AlertDispatcher()
    dispatcher.add_handler(log_alert_handler)
    await dispatcher.run()
