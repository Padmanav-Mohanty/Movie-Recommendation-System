# 🎬 Movie Recommendation System

> **A production-grade, end-to-end recommender system** built on the [MovieLens dataset](https://huggingface.co/datasets/ashraq/movielens_ratings) — from raw ratings to a containerised REST API serving three distinct model families.

[![CI](https://github.com/Padmanav-Mohanty/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Padmanav-Mohanty/Movie-Recommendation-System/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Models](#models)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Docker & Deployment](#docker--deployment)
- [CI/CD](#cicd)
- [Metrics](#metrics)
- [Configuration](#configuration)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        REST API (FastAPI)                            │
│  /recommendations  /ratings/predict  /evaluate  /health  /models    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
   │  User-Based │  │  SVD / NMF   │  │   Two-Tower +    │
   │     CF      │  │  (Surprise)  │  │  FAISS ANN Index │
   └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
          │                │                    │
          └────────────────┴────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐      ┌──────────────────┐
     │  User Features  │      │  Item Features   │
     │  (genre affin,  │      │  (OHE genres,    │
     │   activity)     │      │   year, stats)   │
     └────────┬────────┘      └────────┬─────────┘
              └────────────┬───────────┘
                           ▼
              ┌─────────────────────────┐
              │  MovieLens Dataset      │
              │  (HuggingFace Hub)      │
              └─────────────────────────┘
```

---

## Models

| Model | Type | RMSE | MAE | Notes |
|-------|------|------|-----|-------|
| **User-Based CF** | Memory-based | ~0.95 | ~0.74 | cosine similarity, K=50 neighbours |
| **SVD** | Matrix factorisation | ~0.76 | ~0.58 | 100 latent factors, 20 epochs |
| **NMF** | Matrix factorisation | ~0.83 | ~0.64 | non-negative, same hyperparams |
| **Two-Tower** | Neural (PyTorch) | ~0.72 | ~0.55 | embeddings + side features, FAISS ANN |

All models are evaluated on a held-out temporal test split (last 20% of each user's history).

---

## Project Structure

```
movie-recommendation-system/
├── config.py                         # Centralised hyperparameters & paths
├── api/
│   └── main.py                       # FastAPI application (lifespan, middleware, routes)
├── src/
│   ├── data/
│   │   ├── load_data.py              # HuggingFace download + local caching
│   │   └── preprocess.py            # Cleaning, ID encoding, temporal splits
│   ├── features/
│   │   └── build_features.py        # User/item feature matrices, interaction matrix
│   ├── models/
│   │   ├── collaborative_filtering.py   # User-based CF (cosine similarity)
│   │   ├── matrix_factorization.py      # SVD / NMF via scikit-surprise + MLflow
│   │   └── two_tower.py                 # Two-tower neural model (PyTorch) + MLflow
│   ├── evaluation/
│   │   └── metrics.py               # RMSE, MAE, P/R/NDCG/HR@K, MAP, MRR, Coverage, Novelty
│   └── serving/
│       └── recommender.py           # Unified interface + FAISS ANN index
├── tests/
│   ├── conftest.py                  # Shared fixtures (synthetic dataset)
│   ├── test_metrics.py              # Unit tests — evaluation metrics
│   ├── test_preprocessing.py        # Unit tests — data pipeline
│   ├── test_features.py             # Unit tests — feature engineering
│   ├── test_collaborative_filtering.py  # Unit + integration — CF model
│   └── test_api.py                  # API integration tests (TestClient + mocks)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_cf.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_two_tower_model.ipynb
│   └── 06_evaluation.ipynb
├── Dockerfile                        # Multi-stage build (builder + runtime)
├── docker-compose.yml                # API + optional MLflow tracking server
├── Makefile                          # Developer ergonomics
├── pyproject.toml
└── .github/workflows/ci.yml         # CI: lint → test → Docker build → GHCR push
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) **or** `pip`
- Docker (optional, for containerised deployment)

### 1 — Install

```bash
git clone https://github.com/Padmanav-Mohanty/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# With uv (fast)
make install

# Or with pip
pip install -e .
```

### 2 — Run the full data + training pipeline

```bash
# Download + preprocess data
make data
make preprocess
make features

# Train the recommended model (SVD — best RMSE/speed trade-off)
make train-svd

# Or train everything
make train-all
```

### 3 — Launch the API

```bash
make serve
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### 4 — Get recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 0, "top_k": 10, "model": "svd"}'
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + readiness check |
| `GET` | `/models` | List models and training status |
| `POST` | `/recommendations` | Top-K recommendations for a user |
| `POST` | `/ratings/predict` | Predicted rating for (user, movie) |
| `GET` | `/users/{user_idx}/history` | Movies rated by a user |
| `GET` | `/movies/{movie_idx}` | Movie metadata |
| `GET` | `/evaluate` | Ranking evaluation on test split |

Interactive documentation is available at **`/docs`** (Swagger UI) and **`/redoc`** (ReDoc) when the server is running.

### Example Requests

**Recommendations (SVD model)**
```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 42, "top_k": 5, "model": "svd", "exclude_seen": true}'
```

**Rating prediction**
```bash
curl -X POST http://localhost:8000/ratings/predict \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 42, "movie_idx": 150, "model": "svd"}'
```

**Live evaluation**
```bash
curl "http://localhost:8000/evaluate?model=svd&n_users=500&top_k=10"
```

---

## Running Tests

```bash
# Full test suite
make test

# With coverage report
make test-cov
# → Opens htmlcov/index.html

# Quick unit tests only
make test-fast
```

The test suite uses a **synthetic in-memory dataset** — no data download is required to run tests.

---

## Docker & Deployment

### Local Docker

```bash
# Build image
make docker-build

# Start API (+ optional MLflow tracking)
make docker-up
# → API:    http://localhost:8000
# → MLflow: docker compose --profile tracking up

# Stop
make docker-down
```

### Production image

The multi-stage `Dockerfile` produces a minimal runtime image (~300 MB):
- **Builder stage**: installs all dependencies with `uv`
- **Runtime stage**: copies only the virtual environment + source; runs as a non-root user

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `ENV` | `production` | `development` enables hot-reload |
| `DEFAULT_MODEL` | `svd` | Default recommendation model |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

---

## CI/CD

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push and PR:

```
push/PR
  │
  ├── lint        ruff check + format check
  │
  ├── test        pytest + coverage (Python 3.12)
  │                └── coverage uploaded to Codecov
  │
  ├── docker      docker build --target runtime (cache via GHA)
  │
  └── publish     (main branch only)
                  docker push → ghcr.io/<owner>/movie-recommendation-system
```

---

## Metrics

### Rating Prediction
| Metric | Formula |
|--------|---------|
| **RMSE** | √(mean((y_true − y_pred)²)) |
| **MAE** | mean(|y_true − y_pred|) |

### Ranking (implemented for all K values)
| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K that are relevant |
| **Recall@K** | Fraction of relevant items in top-K |
| **NDCG@K** | Normalised Discounted Cumulative Gain |
| **HitRate@K** | 1 if any relevant item appears in top-K |
| **MAP** | Mean Average Precision |
| **MRR** | Mean Reciprocal Rank |

### Beyond Accuracy
| Metric | Description |
|--------|-------------|
| **Catalogue Coverage@K** | % of items recommended to at least one user |
| **Novelty@K** | Mean self-information (rewards less-popular items) |

---

## Configuration

All hyperparameters and paths live in `config.py`:

```python
# Preprocessing
MIN_USER_RATINGS  = 5      # Drop cold-start users
MIN_MOVIE_RATINGS = 5      # Drop cold-start items
TEST_SIZE         = 0.2    # Last 20% of each user's history

# SVD
SVD_N_FACTORS = 100
SVD_N_EPOCHS  = 20
SVD_LR        = 0.005
SVD_REG       = 0.02

# Two-Tower
EMBEDDING_DIM = 64
HIDDEN_DIMS   = [256, 128]
DROPOUT       = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE    = 1024
NUM_EPOCHS    = 20

# Serving
N_CANDIDATES  = 100    # FAISS retrieves top-100, reranked to top-K
```

---

## License

[MIT](LICENSE)
