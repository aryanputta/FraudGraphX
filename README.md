# FraudGraphX

**Production-grade enterprise fraud detection platform** — detects fraud rings, mule accounts, synthetic identity fraud, and money-laundering chains in real time.

## Architecture at a Glance

```
Transactions → Kafka → Spark Streaming → ML Ensemble → FastAPI (<150ms) → React UI
                              ↓                  ↓
                           Neo4j Graph      Cassandra
                           (ring detect)    (time-series)
                              ↓
                        Explainability
                     (SHAP + Graph Path + LLM)
```

See [docs/architecture.md](docs/architecture.md) for the full diagram.

## Stack

| Layer | Technology |
|-------|-----------|
| Streaming | Kafka + Spark Structured Streaming |
| Graph | Neo4j 5.x + GDS |
| ML | LogReg · XGBoost · GNN (PyTorch Geometric) |
| Storage | Cassandra 4.x |
| API | FastAPI + Prometheus |
| Frontend | React + Tailwind + Recharts + Cytoscape |
| Infra | Docker · Kubernetes · Terraform (AWS EKS) |
| LLM | Claude Sonnet (prompt caching) |

## Quick Start (Local)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Generate synthetic training data
python data/scripts/generate_synthetic.py

# 4. Train models
python -m src.models.trainer

# 5. API is live at http://localhost:8000/docs
# 6. UI is live at http://localhost:3000
```

## Development

```bash
# Install Python deps
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v --cov=src

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/score` | POST | Score a transaction (<150ms) |
| `/v1/score/batch` | POST | Score up to 100 transactions |
| `/v1/alerts` | GET | List recent fraud alerts |
| `/v1/alerts/{id}` | GET | Alert detail + SHAP + graph |
| `/v1/cases` | POST | Create fraud case |
| `/v1/cases/{id}` | PUT | Update case status |
| `/v1/customers/{id}/graph` | GET | Fraud subgraph |
| `/v1/customers/{id}/risk` | GET | Customer risk summary |
| `/v1/feedback` | POST | Analyst feedback (RL adaptation) |
| `/v1/benchmark` | GET | Model benchmark results |
| `/v1/benchmark/run` | POST | Trigger model training |
| `/healthz` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Model Benchmark

Compare three approaches:

| Model | Precision | Recall | F1 | Avg Latency |
|-------|-----------|--------|----|-------------|
| Logistic Regression | baseline | baseline | baseline | ~2ms |
| XGBoost | ↑↑ | ↑ | ↑↑ | ~15ms |
| GNN (GraphSAGE+GAT) | ↑ | ↑↑↑ | ↑↑ | ~30ms |

Trigger a fresh benchmark via the UI or `POST /v1/benchmark/run`.

## Fraud Detection Patterns

| Pattern | Detection Method |
|---------|-----------------|
| Fraud Ring | Shared device/IP graph clustering |
| Mule Account | Large inbound → outbound within 60min |
| Synthetic Identity | New account + high-value + cross-border |
| Account Takeover | New device + geo anomaly + velocity spike |
| Money Laundering | Neo4j chain pattern matching |

## Production Deployment (AWS)

```bash
cd infra/terraform
terraform init
terraform plan -var="environment=prod"
terraform apply

# Deploy to EKS
kubectl apply -k infra/kubernetes/overlays/prod/

# Nightly rescoring runs automatically via CronJob at 02:00 UTC
```

## Project Structure

```
FraudGraphX/
├── src/
│   ├── common/              # Config, schemas, logging
│   ├── ingestion/           # Kafka producer/consumer, feature engineering
│   ├── streaming/           # Spark Structured Streaming pipeline
│   ├── graph/               # Neo4j client + graph builder
│   ├── models/              # LogReg, XGBoost, GNN, ensemble, trainer
│   ├── api/                 # FastAPI app + routers
│   ├── storage/             # Cassandra client
│   ├── explainability/      # SHAP, graph path, LLM investigator
│   ├── alerting/            # Real-time alert dispatcher
│   └── rescoring/           # Nightly batch rescorer
├── frontend/                # React + TypeScript + Tailwind
├── tests/
│   ├── unit/                # Fast unit tests (no external deps)
│   └── integration/         # Integration tests
├── infra/
│   ├── docker/              # Dockerfiles + Compose config
│   ├── kubernetes/          # K8s manifests (base + overlays)
│   └── terraform/           # AWS infrastructure
├── data/
│   ├── scripts/             # Synthetic data generator
│   └── sample/              # Generated datasets (gitignored)
└── docs/                    # Architecture + API docs
```

## Metrics & Observability

- **Prometheus** scrapes `/metrics` every 15s
- **Grafana** dashboards at `localhost:3001`
- **CloudWatch alarm**: p99 latency > 150ms triggers SNS alert
- **Structured logging** (JSON in production, human-readable in dev)

## License

MIT
