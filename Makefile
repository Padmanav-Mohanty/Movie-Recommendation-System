# Movie Recommendation System — Developer Makefile
# Usage: make <target>

.PHONY: help install install-dev lint format test test-cov clean \
        data preprocess features train-cf train-svd train-two-tower \
        serve docker-build docker-up docker-down

PYTHON  := python
UV      := uv
API_PORT := 8000

## ── Help ─────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

## ── Setup ─────────────────────────────────────────────────────────────────────
install:  ## Install production dependencies
	$(UV) sync --frozen

install-dev:  ## Install all dependencies including dev tools
	$(UV) sync --frozen
	$(UV) pip install ruff mypy pytest pytest-cov httpx

## ── Code quality ──────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	$(UV) run ruff check src/ api/ tests/ config.py

format:  ## Auto-format with ruff
	$(UV) run ruff format src/ api/ tests/ config.py
	$(UV) run ruff check --fix src/ api/ tests/ config.py

typecheck:  ## Run mypy type checking
	$(UV) run mypy src/ api/ --ignore-missing-imports

## ── Tests ─────────────────────────────────────────────────────────────────────
test:  ## Run the test suite
	$(UV) run pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	$(UV) run pytest tests/ -v \
	  --cov=src --cov=api \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --tb=short
	@echo "Coverage report → htmlcov/index.html"

test-fast:  ## Run only fast tests (exclude slow integration tests)
	$(UV) run pytest tests/ -v -m "not slow" --tb=short

## ── Data pipeline ────────────────────────────────────────────────────────────
data:  ## Download raw dataset from HuggingFace
	$(UV) run python src/data/load_data.py

preprocess:  ## Clean, encode, and split the data
	$(UV) run python src/data/preprocess.py

features:  ## Build user/item feature matrices
	$(UV) run python src/features/build_features.py

## ── Model training ────────────────────────────────────────────────────────────
train-cf:  ## Train User-Based Collaborative Filtering
	$(UV) run python src/models/collaborative_filtering.py

train-svd:  ## Train Matrix Factorization (SVD)
	$(UV) run python src/models/matrix_factorization.py

train-two-tower:  ## Train Two-Tower Neural Model
	$(UV) run python src/models/two_tower.py

train-all: train-cf train-svd train-two-tower  ## Train all models sequentially

## ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:  ## Run offline evaluation on test split
	$(UV) run python src/evaluation/metrics.py

## ── Serving ───────────────────────────────────────────────────────────────────
serve:  ## Start the FastAPI development server
	$(UV) run uvicorn api.main:app \
	  --host 0.0.0.0 \
	  --port $(API_PORT) \
	  --reload \
	  --log-level info

serve-prod:  ## Start the FastAPI production server
	ENV=production $(UV) run uvicorn api.main:app \
	  --host 0.0.0.0 \
	  --port $(API_PORT) \
	  --workers 2 \
	  --log-level warning

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:  ## Build the Docker image
	docker build --target runtime -t movie-rec-api:local .

docker-up:  ## Start Docker Compose stack
	docker compose up --build -d

docker-down:  ## Stop Docker Compose stack
	docker compose down

docker-logs:  ## Follow API container logs
	docker compose logs -f api

## ── Misc ──────────────────────────────────────────────────────────────────────
clean:  ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml .mypy_cache .ruff_cache

full-pipeline: data preprocess features train-svd  ## Run end-to-end (data → train SVD)
