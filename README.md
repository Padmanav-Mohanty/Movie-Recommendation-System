# Movie Recommendation System

A full-stack movie recommendation system built on the [MovieLens dataset](https://huggingface.co/datasets/ashraq/movielens-ratings) from HuggingFace.

## Architecture

```
├── config.py               — centralised hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── load_data.py    — HuggingFace dataset download + caching
│   │   └── preprocess.py   — cleaning, ID encoding, temporal splits
│   ├── features/
│   │   └── build_features.py — user/item feature matrices, interaction matrix
│   ├── models/
│   │   ├── collaborative_filtering.py  — user-based CF (cosine similarity)
│   │   ├── matrix_factorization.py     — SVD / NMF via scikit-surprise
│   │   └── two_tower.py                — two-tower neural model (PyTorch)
│   ├── evaluation/
│   │   └── metrics.py      — RMSE, MAE, Precision/Recall/NDCG/HitRate@K, MAP, MRR
│   └── serving/
│       └── recommender.py  — unified recommender + FAISS ANN index
├── api/
│   └── main.py             — FastAPI REST API
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_baseline_cf.ipynb
    ├── 04_matrix_factorization.ipynb
    ├── 05_two_tower_model.ipynb
    └── 06_evaluation.ipynb
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e .        # or: uv sync

# 2. Download & preprocess data
python src/data/load_data.py
python src/data/preprocess.py

# 3. Build features
python src/features/build_features.py

# 4. Train models (choose any)
python src/models/collaborative_filtering.py
python src/models/matrix_factorization.py
python src/models/two_tower.py

# 5. Launch the API
uvicorn api.main:app --reload --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/models` | List available trained models |
| POST | `/recommendations` | Top-K recommendations for a user |
| POST | `/ratings/predict` | Predicted rating for (user, movie) |
| GET | `/users/{user_idx}/history` | Rated movies for a user |
| GET | `/movies/{movie_idx}` | Movie metadata |
| GET | `/evaluate` | Quick ranking evaluation on test split |

### Example

```bash
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_idx": 0, "top_k": 10, "model": "svd"}'
```

## Models

| Model | Type | RMSE (approx) |
|-------|------|---------------|
| User-Based CF | Memory-based | ~0.95 |
| SVD | Matrix factorisation | ~0.76 |
| Two-Tower | Neural (PyTorch) | ~0.72 |

## Metrics Implemented

**Rating prediction:** RMSE, MAE  
**Ranking:** Precision@K, Recall@K, NDCG@K, Hit Rate@K, MAP@K, MRR@K  
**Beyond accuracy:** Catalogue Coverage@K, Novelty@K
