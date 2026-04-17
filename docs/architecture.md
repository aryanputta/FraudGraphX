# FraudGraphX – Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FraudGraphX Platform                                   │
│                                                                                  │
│  ┌─────────────┐    ┌────────────────────────────────────────────────────────┐  │
│  │   Sources   │    │              Ingestion Layer                            │  │
│  │             │    │                                                         │  │
│  │ • Kaggle CC │───▶│  TransactionProducer  ──▶  Kafka (transactions.raw)   │  │
│  │ • IEEE-CIS  │    │  FeatureEngineer      ──▶  EnrichedTransaction        │  │
│  │ • Synthetic │    │                                                         │  │
│  └─────────────┘    └────────────────────────────────────────────────────────┘  │
│                                        │                                         │
│                             ┌──────────▼──────────┐                             │
│                             │  Spark Structured    │                             │
│                             │  Streaming           │                             │
│                             │                      │                             │
│                             │  • Feature UDF       │                             │
│                             │  • XGBoost UDF       │                             │
│                             │  • Cassandra Sink    │                             │
│                             │  • Kafka Alert Sink  │                             │
│                             └──────────┬──────────┘                             │
│                                        │                                         │
│           ┌────────────────────────────┼────────────────────────────┐           │
│           │                            │                            │           │
│  ┌────────▼────────┐        ┌──────────▼──────────┐    ┌──────────▼────────┐  │
│  │   Neo4j Graph   │        │   ML Ensemble        │    │    Cassandra       │  │
│  │                 │        │                      │    │                    │  │
│  │ • Customer node │        │ • LogReg (0.2 wt)    │    │ enriched_txns      │  │
│  │ • Device node   │        │ • XGBoost (0.5 wt)   │    │ fraud_alerts       │  │
│  │ • Merchant node │        │ • GNN/PyG (0.3 wt)   │    │ fraud_cases        │  │
│  │ • IP node       │        │ • Adaptive RL         │    │ model_metrics      │  │
│  │ • Fraud rings   │        │   Threshold           │    │                    │  │
│  └────────┬────────┘        └──────────┬──────────┘    └────────────────────┘  │
│           │                            │                                         │
│           └──────────────┬─────────────┘                                        │
│                          │                                                       │
│              ┌───────────▼───────────┐                                          │
│              │  Explainability Layer  │                                          │
│              │                       │                                          │
│              │ • SHAP (XGBoost/LR)   │                                          │
│              │ • Graph path          │                                          │
│              │ • LLM investigator    │                                          │
│              │   (Claude Sonnet w/   │                                          │
│              │    prompt caching)    │                                          │
│              └───────────┬───────────┘                                          │
│                          │                                                       │
│              ┌───────────▼───────────┐                                          │
│              │    FastAPI Backend    │                                          │
│              │                       │                                          │
│              │ POST /v1/score  <150ms│                                          │
│              │ GET  /v1/alerts       │                                          │
│              │ POST /v1/cases        │                                          │
│              │ GET  /v1/customers    │                                          │
│              │ POST /v1/feedback     │                                          │
│              │ GET  /v1/benchmark    │                                          │
│              └───────────┬───────────┘                                          │
│                          │                                                       │
│              ┌───────────▼───────────┐       ┌──────────────────────┐           │
│              │   React Frontend      │       │  Alert Dispatcher     │           │
│              │                       │       │                       │           │
│              │ • Dashboard           │       │ • Kafka consumer      │           │
│              │ • Alerts (SHAP+path)  │       │ • <150ms dispatch     │           │
│              │ • Case management     │       │ • Webhook / PagerDuty │           │
│              │ • Graph visualisation │       └──────────────────────┘           │
│              │ • Model benchmark     │                                           │
│              └───────────────────────┘       ┌──────────────────────┐           │
│                                              │  Nightly Rescorer     │           │
│                                              │  (K8s CronJob)        │           │
│                                              │ • Re-score customers  │           │
│                                              │ • Detect laundering   │           │
│                                              │ • Update graph risk   │           │
│                                              └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ingestion Layer
- **TransactionProducer**: Confluent-Kafka producer with LZ4 compression, acks=all
- **TransactionConsumer**: Manual-commit consumer with graceful shutdown
- **FeatureEngineer**: Stateful real-time feature computation
  - `VelocityTracker`: Rolling-window transaction count (1h, 24h windows)
  - `GeoTracker`: Haversine distance from last known location
  - Device / IP seen-before flags (per-customer set)

### 2. Spark Structured Streaming
- Reads `transactions.raw` Kafka topic
- Applies `enrich_and_score` Pandas UDF (feature engineering + XGBoost)
- Writes enriched records to Cassandra
- Forwards fraud alerts to `fraud.alerts` Kafka topic

### 3. Graph Layer (Neo4j)
- **Node types**: Customer, Account, Transaction, Merchant, Device, IPAddress
- **Edge types**: OWNS, MADE_TRANSACTION, AT_MERCHANT, USES_DEVICE, FROM_DEVICE, ACCESSED_FROM_IP, FROM_IP
- **Fraud ring detection**: Community detection via shared device/IP connections
- **Money laundering**: Large inbound→outbound within 60 minutes

### 4. ML Ensemble
| Model | Weight | Strengths |
|-------|--------|-----------|
| Logistic Regression | 0.20 | Fast baseline, interpretable |
| XGBoost | 0.50 | Tabular features, best AUROC |
| GNN (GraphSAGE+GAT) | 0.30 | Graph structure, ring detection |

**Adaptive Threshold**: ε-greedy RL agent adjusts decision threshold based on analyst feedback (false-positive penalty = -1, false-negative penalty = -10, true-positive reward = +5).

### 5. Explainability
- **SHAP**: TreeExplainer for XGBoost, KernelExplainer for LogReg
- **Graph Path**: Neo4j Cypher shortest-path to nearest suspicious node
- **LLM Investigator**: Claude Sonnet with prompt caching generates investigation notes

### 6. API (<150ms SLA)
- FastAPI with async handlers and background task persistence
- Prometheus metrics at `/metrics`
- CORS, middleware latency tracking
- Batch scoring (up to 100 transactions)

### 7. Storage
- **Cassandra**: Time-series transaction storage, partitioned by (customer_id, year_month)
  - TTL: 90 days for enriched transactions
  - Replication factor: 3 (NetworkTopologyStrategy)
- **Neo4j**: Graph relationships, 5.x Enterprise with GDS plugin

### 8. Infrastructure
- **Docker Compose**: Local development (all services)
- **Kubernetes**: 3-replica API deployment, HPA (3–20 pods), GPU node group for GNN
- **Terraform**: AWS EKS + MSK (Kafka) + Amazon Keyspaces + Neo4j EC2 cluster
- **Nightly CronJob**: K8s CronJob at 02:00 UTC for risk rescoring

## Latency Budget (p99 target: <150ms)

| Stage | Budget |
|-------|--------|
| Kafka consume + deserialize | ~5ms |
| Feature engineering | ~10ms |
| LogReg inference | ~2ms |
| XGBoost inference | ~15ms |
| GNN inference | ~30ms |
| Ensemble + SHAP | ~20ms |
| Cassandra write (async) | background |
| Total | **~82ms** ✓ |

## Security
- Secrets managed via Kubernetes Secrets (not ConfigMaps)
- Cassandra and Neo4j authentication required
- API: CORS restricted to frontend origin
- TLS via cert-manager + Let's Encrypt
- S3 model bucket: server-side AES256 encryption + versioning
