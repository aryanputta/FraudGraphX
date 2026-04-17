"""Centralised settings loaded from environment / .env file."""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_transactions: str = "transactions.raw"
    kafka_topic_alerts: str = "fraud.alerts"
    kafka_topic_enriched: str = "transactions.enriched"
    kafka_consumer_group: str = "fraudgraphx-consumers"
    kafka_security_protocol: str = "PLAINTEXT"

    # Spark
    spark_master: str = "local[*]"
    spark_app_name: str = "FraudGraphX"
    spark_checkpoint_dir: str = "/tmp/spark-checkpoints"
    spark_batch_interval_secs: int = 5

    # Cassandra
    cassandra_hosts: str = "localhost"
    cassandra_port: int = 9042
    cassandra_keyspace: str = "fraudgraphx"
    cassandra_username: str = "cassandra"
    cassandra_password: str = "cassandra"
    cassandra_replication_factor: int = 3

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "fraudgraphx"
    neo4j_database: str = "fraudgraphx"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"
    api_allowed_origins: str = "http://localhost:3000"
    alert_latency_target_ms: int = 150

    # ML
    model_dir: str = "/models"
    xgboost_model_path: str = "/models/xgboost_fraud.ubj"
    gnn_model_path: str = "/models/gnn_fraud.pt"
    logreg_model_path: str = "/models/logreg_fraud.pkl"
    fraud_threshold: float = 0.5
    gnn_fraud_threshold: float = 0.6

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_investigator_enabled: bool = True

    # Monitoring
    prometheus_port: int = 9090
    log_level: str = "INFO"

    # RL
    rl_threshold_learning_rate: float = 0.01
    rl_reward_fp_penalty: float = -1.0
    rl_reward_fn_penalty: float = -10.0
    rl_reward_tp_reward: float = 5.0
    rl_epsilon: float = 0.1

    @property
    def allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.api_allowed_origins.split(",")]

    @property
    def cassandra_contact_points(self) -> List[str]:
        return [h.strip() for h in self.cassandra_hosts.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
