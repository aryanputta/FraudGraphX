.PHONY: help up down generate-data train test lint typecheck benchmark api-dev frontend-dev clean

PYTHON := python
PIP    := pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## Tail all service logs
	docker-compose logs -f

# ── Setup ─────────────────────────────────────────────────────────────────────
install: ## Install Python dependencies
	$(PIP) install -e ".[dev]"

install-frontend: ## Install frontend dependencies
	cd frontend && npm ci --legacy-peer-deps

# ── Data ──────────────────────────────────────────────────────────────────────
generate-data: ## Generate synthetic transaction dataset
	$(PYTHON) data/scripts/generate_synthetic.py

# ── Models ───────────────────────────────────────────────────────────────────
train: ## Train all models and run benchmark
	$(PYTHON) -m src.models.trainer

train-tune: ## Train all models with Optuna HPO
	$(PYTHON) -c "from src.models.trainer import train_all; train_all(tune=True)"

# ── Development servers ───────────────────────────────────────────────────────
api-dev: ## Run FastAPI dev server
	uvicorn src.api.main:app --reload --port 8000

frontend-dev: ## Run React dev server
	cd frontend && npm run dev

spark-streaming: ## Run Spark streaming pipeline
	$(PYTHON) -m src.streaming.spark_pipeline

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all unit tests
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/unit/ -v -x

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ tests/

lint-fix: ## Fix lint issues
	ruff check --fix src/ tests/

typecheck: ## Run mypy type checker
	mypy src/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .coverage htmlcov/ .mypy_cache/ .pytest_cache/

# ── Production ────────────────────────────────────────────────────────────────
terraform-plan: ## Plan Terraform changes
	cd infra/terraform && terraform plan -var="environment=prod"

terraform-apply: ## Apply Terraform changes
	cd infra/terraform && terraform apply -var="environment=prod"

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -k infra/kubernetes/overlays/prod/

rescore: ## Run nightly rescorer manually
	$(PYTHON) -m src.rescoring.nightly_rescorer
