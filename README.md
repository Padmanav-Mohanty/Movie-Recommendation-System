# 🎬 Movie Recommendation System

> **Production-grade, end-to-end recommender system** built on the [MovieLens dataset](https://huggingface.co/datasets/ashraq/movielens_ratings) — from raw ratings to a containerised REST API serving three distinct model families across 940k ratings, 29k users, and 7.6k movies.

[![CI](https://github.com/Padmanav-Mohanty/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Padmanav-Mohanty/Movie-Recommendation-System/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1-orange)](https://surpriselib.com/)
[![FAISS](https://img.shields.io/badge/FAISS-ANN-blueviolet)](https://faiss.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What Makes This Different

Most recommendation system projects stop at a Jupyter notebook with a single algorithm. This project mirrors how recommendations are built in industry:

- **3 model families** trained and served behind a single unified API
- **Temporal train/test split** — avoids the data-leakage mistake most tutorials make
- **FAISS Approximate Nearest Neighbour** index for sub-millisecond Two-Tower retrieval
- **MLflow experiment tracking** with logged metrics per run
- **Full evaluation suite** — not just RMSE, but NDCG, MAP, MRR, Coverage, Novelty
- **Containerised** with Docker + CI/CD via GitHub Actions

---

## 🏗️ Architecture

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
              │  940k ratings           │
              │  29,474 users           │
              │  7,642 movies           │
              └─────────────────────────┘
```

---

## 📊 Model Results

All models evaluated on a **temporal held-out test split** (last 20% of each user's rating history).

### Rating Prediction

| Model | RMSE ↓ | MAE ↓ | Train Time |
|-------|--------|-------|------------|
| **User-Based CF** | ~0.95 | ~0.74 | — (memory-based) |
| **SVD** | **0.893** | **0.779** | ~8s |
| **NMF** | ~0.83 | ~0.64 | ~12s |
| **Two-Tower** | ~0.72 | ~0.55 | ~15min (CPU) |

### Ranking Metrics @ K=10 (SVD)

| Precision@10 | Recall@10 | NDCG@10 | MAP | MRR |
|---|---|---|---|---|
| 0.0312 | 0.0587 | 0.0441 | 0.0298 | 0.1023 |

### Sample API Response

```json
POST /recommendations
{
  "user_idx": 42,
  "top_k": 10,
  "model": "svd",
  "exclude_seen": true
}

→ 200 OK
{
  "user_idx": 42,
  "model": "svd",
  "top_k": 10,
  "recommendations": [
    { "movie_idx": 4781, "title": "Planet Earth II (2016)",           "genres": "Documentary" },
    { "movie_idx": 5506, "title": "Blue Planet II (2017)",            "genres": "Documentary" },
    { "movie_idx":  630, "title": "Gone Girl (2014)",                 "genres": "Drama|Thriller" },
    { "movie_idx": 1279, "title": "Story of Film: An Odyssey (2011)", "genres": "Documentary" },
    { "movie_idx":  701, "title": "Coherence (2013)",                 "genres": "Mystery|Sci-Fi|Thriller" }
  ]
}
```

---

## 🧠 Models

### User-Based Collaborative Filtering
Memory-based CF using cosine similarity across the full user-user matrix. K=50 nearest neighbours with weighted rating aggregation. No training required — predicts directly from the interaction matrix.

### SVD / NMF (Matrix Factorisation)
Uses [scikit-surprise](https://surpriselib.com/) with 100 latent factors, trained for 20 epochs. SVD achieves the best RMSE/speed trade-off and is the recommended default model. Cross-validated RMSE: **0.7638**.

### Two-Tower Neural Model (PyTorch + FAISS)
Dual-encoder architecture with separate user and item towers. Each tower combines learned embeddings with side features (genre affinities, activity stats for users; OHE genres, year, popularity for items). At serve time, user vectors are queried against a pre-built **FAISS IndexFlatIP** for sub-millisecond ANN retrieval across 7.6k items.

---

## 📁 Project Structure

```
movie-recommendation-system/
├── config.py                         # Centralised hyperparameters & paths
├── api/
│   └── main.py                       # FastAPI app (lifespan, middleware, routes)
├── src/
│   ├── data/
│   │   ├── load_data.py              # HuggingFace download + local caching
│   │   └── preprocess.py            # Cleaning, ID encoding, temporal splits
│   ├── features/
│   │   └── build_features.py        # User/item feature matrices, interaction matrix
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   ├── matrix_factorization.py  # SVD / NMF via scikit-surprise + MLflow
│   │   └── two_tower.py             # Two-Tower neural model (PyTorch) + MLflow
│   ├── evaluation/
│   │   └── metrics.py               # RMSE, MAE, P/R/NDCG/HR@K, MAP, MRR, Coverage, Novelty
│   └── serving/
│       └── recommender.py           # Unified interface + FAISS ANN index
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_cf.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_two_tower_model.ipynb
│   └── 06_evaluation_fixed.ipynb
├── tests/                            # pytest suite — unit + integration
├── Dockerfile                        # Multi-stage build (builder + runtime)
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml         # lint → test → Docker build → GHCR push
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- Docker (optional)

### 1 — Install

```bash
git clone https://github.com/Padmanav-Mohanty/Movie-Recommendation-System.git
cd Movie-Recommendation-System
pip install uv
uv sync --frozen
```

### 2 — Run the pipeline

```bash
# Download + preprocess data
uv run python -m src.data.load_data
uv run python -m src.data.preprocess
uv run python -m src.features.build_features

# Train SVD (recommended — best speed/accuracy trade-off, ~8s)
$env:PYTHONPATH = (Get-Location).Path   # Windows PowerShell
python src/models/matrix_factorization.py

# Or on Linux/Mac:
PYTHONPATH=. python src/models/matrix_factorization.py
```

### 3 — Launch the API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 4 — Get recommendations

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 42, "top_k": 10, "model": "svd", "exclude_seen": true}'
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + readiness check |
| `GET` | `/models` | List models and training status |
| `POST` | `/recommendations` | Top-K recommendations for a user |
| `POST` | `/ratings/predict` | Predicted rating for (user, movie) pair |
| `GET` | `/users/{user_idx}/history` | Movies rated by a user |
| `GET` | `/movies/{movie_idx}` | Movie metadata |
| `GET` | `/evaluate` | Live ranking evaluation on test split |

Interactive docs: **`/docs`** (Swagger UI) · **`/redoc`** (ReDoc)

---

## 🧪 Running Tests

```bash
uv run pytest tests/ -v --tb=short        # full suite
uv run pytest tests/ -m "not slow"        # fast unit tests only
uv run pytest tests/ --cov=src --cov=api  # with coverage
```

Tests use a **synthetic in-memory dataset** — no data download required.

---

## 🐳 Docker

```bash
# Build and start
docker compose up --build -d

# API → http://localhost:8000
# MLflow UI → docker compose --profile tracking up

docker compose down
```

---

## ⚙️ CI/CD

GitHub Actions runs on every push and PR:

```
push / PR
  ├── lint      ruff check + format
  ├── test      pytest + coverage → Codecov
  ├── docker    build --target runtime
  └── publish   (main only) → ghcr.io/padmanav-mohanty/movie-recommendation-system
```

---

## 📏 Evaluation Metrics

### Rating Prediction
| Metric | Formula |
|--------|---------|
| **RMSE** | √(mean((y_true − y_pred)²)) |
| **MAE** | mean(\|y_true − y_pred\|) |

### Ranking
| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K that are relevant |
| **Recall@K** | Fraction of relevant items found in top-K |
| **NDCG@K** | Normalised Discounted Cumulative Gain |
| **HitRate@K** | 1 if any relevant item in top-K |
| **MAP** | Mean Average Precision |
| **MRR** | Mean Reciprocal Rank |

### Beyond Accuracy
| Metric | Description |
|--------|-------------|
| **Coverage@K** | % of catalogue recommended to ≥1 user |
| **Novelty@K** | Mean self-information (rewards less-popular items) |

---

## ⚙️ Configuration

All hyperparameters in `config.py`:

```python
# Preprocessing
MIN_USER_RATINGS  = 5      # drop cold-start users
MIN_MOVIE_RATINGS = 5      # drop cold-start items
TEST_SIZE         = 0.2    # temporal last-20% split

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

## 📄 License

[MIT](LICENSE)
