"""
Spark Structured Streaming pipeline.

Flow:
  Kafka (transactions.raw)
    → feature engineering (pandas UDF)
    → XGBoost scoring UDF
    → Cassandra sink (enriched transactions)
    → Kafka sink (fraud.alerts for real-time alerting)
"""

from __future__ import annotations

import json
import pickle
from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Spark session ──────────────────────────────────────────────────────────────

def build_spark_session() -> SparkSession:
    packages = ",".join(
        [
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
            "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0",
        ]
    )
    return (
        SparkSession.builder.appName(settings.spark_app_name)
        .master(settings.spark_master)
        .config("spark.jars.packages", packages)
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.cassandra.connection.host", settings.cassandra_hosts)
        .config("spark.cassandra.connection.port", str(settings.cassandra_port))
        .config("spark.cassandra.auth.username", settings.cassandra_username)
        .config("spark.cassandra.auth.password", settings.cassandra_password)
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


# ── Schema ─────────────────────────────────────────────────────────────────────

TRANSACTION_SCHEMA = T.StructType(
    [
        T.StructField("transaction_id", T.StringType()),
        T.StructField("timestamp", T.StringType()),
        T.StructField("amount", T.DoubleType()),
        T.StructField("currency", T.StringType()),
        T.StructField("customer_id", T.StringType()),
        T.StructField("account_id", T.StringType()),
        T.StructField("merchant_id", T.StringType()),
        T.StructField("merchant_category", T.StringType()),
        T.StructField("device_id", T.StringType()),
        T.StructField("ip_address", T.StringType()),
        T.StructField("geolocation", T.MapType(T.StringType(), T.DoubleType())),
        T.StructField("card_type", T.StringType()),
        T.StructField("is_online", T.BooleanType()),
        T.StructField("channel", T.StringType()),
    ]
)

FEATURE_SCHEMA = T.StructType(
    TRANSACTION_SCHEMA.fields
    + [
        T.StructField("hour_of_day", T.IntegerType()),
        T.StructField("day_of_week", T.IntegerType()),
        T.StructField("amount_log", T.DoubleType()),
        T.StructField("velocity_1h", T.IntegerType()),
        T.StructField("velocity_24h", T.IntegerType()),
        T.StructField("amount_zscore", T.DoubleType()),
        T.StructField("merchant_risk_score", T.DoubleType()),
        T.StructField("device_seen_before", T.BooleanType()),
        T.StructField("ip_seen_before", T.BooleanType()),
        T.StructField("geo_distance_km", T.DoubleType()),
        T.StructField("is_cross_border", T.BooleanType()),
        T.StructField("fraud_score", T.DoubleType()),
        T.StructField("is_fraud_predicted", T.BooleanType()),
    ]
)


# ── Pandas UDFs ───────────────────────────────────────────────────────────────

@F.pandas_udf(FEATURE_SCHEMA, F.PandasUDFType.MAP_ITER)  # type: ignore[misc]
def enrich_and_score(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Stateless feature engineering + XGBoost scoring UDF."""
    import math
    import pickle

    import xgboost as xgb

    try:
        model = xgb.XGBClassifier()
        model.load_model(settings.xgboost_model_path)
    except Exception:
        model = None

    FEATURE_COLS = [
        "amount_log", "hour_of_day", "day_of_week",
        "velocity_1h", "velocity_24h", "amount_zscore",
        "merchant_risk_score", "geo_distance_km",
    ]

    for batch in iterator:
        batch["timestamp"] = pd.to_datetime(batch["timestamp"])
        batch["hour_of_day"] = batch["timestamp"].dt.hour.astype("int32")
        batch["day_of_week"] = batch["timestamp"].dt.dayofweek.astype("int32")
        batch["amount_log"] = batch["amount"].apply(math.log1p)
        batch["velocity_1h"] = 0
        batch["velocity_24h"] = 0
        batch["amount_zscore"] = 0.0
        batch["merchant_risk_score"] = 0.2
        batch["device_seen_before"] = True
        batch["ip_seen_before"] = True
        batch["geo_distance_km"] = 0.0
        batch["is_cross_border"] = False

        if model is not None:
            X = batch[FEATURE_COLS].fillna(0.0)
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = [0.0] * len(batch)

        batch["fraud_score"] = scores
        batch["is_fraud_predicted"] = batch["fraud_score"] >= settings.fraud_threshold
        batch["timestamp"] = batch["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        yield batch


# ── Pipeline ───────────────────────────────────────────────────────────────────

class FraudStreamingPipeline:
    """End-to-end Spark Structured Streaming fraud detection pipeline."""

    def __init__(self) -> None:
        self.spark = build_spark_session()

    def _read_kafka_stream(self) -> DataFrame:
        return (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
            .option("subscribe", settings.kafka_topic_transactions)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("kafka.group.id", f"{settings.kafka_consumer_group}-spark")
            .load()
        )

    def _parse_transactions(self, raw: DataFrame) -> DataFrame:
        return raw.select(
            F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("txn")
        ).select("txn.*")

    def _write_to_cassandra(self, df: DataFrame) -> object:
        return (
            df.writeStream.format("org.apache.spark.sql.cassandra")
            .option("keyspace", settings.cassandra_keyspace)
            .option("table", "enriched_transactions")
            .option("checkpointLocation", f"{settings.spark_checkpoint_dir}/cassandra")
            .outputMode("append")
            .start()
        )

    def _write_alerts_to_kafka(self, df: DataFrame) -> object:
        alert_df = df.filter(F.col("is_fraud_predicted") == True).select(  # noqa: E712
            F.col("customer_id").alias("key"),
            F.to_json(
                F.struct(
                    "transaction_id", "customer_id", "fraud_score",
                    "merchant_id", "amount", "timestamp",
                )
            ).alias("value"),
        )
        return (
            alert_df.writeStream.format("kafka")
            .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
            .option("topic", settings.kafka_topic_alerts)
            .option("checkpointLocation", f"{settings.spark_checkpoint_dir}/kafka-alerts")
            .outputMode("append")
            .start()
        )

    def run(self) -> None:
        logger.info("Starting Spark Structured Streaming pipeline")

        raw = self._read_kafka_stream()
        parsed = self._parse_transactions(raw)
        enriched = enrich_and_score(parsed)

        cassandra_query = self._write_to_cassandra(enriched)
        kafka_query = self._write_alerts_to_kafka(enriched)

        logger.info("Pipeline running – awaiting termination")
        self.spark.streams.awaitAnyTermination()


def run_pipeline() -> None:
    FraudStreamingPipeline().run()


if __name__ == "__main__":
    run_pipeline()
