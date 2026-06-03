"""
FastAPI application for the Movie Recommendation System.

Endpoints:
  GET  /health                         — liveness check
  GET  /models                         — list available trained models
  POST /recommendations                — get top-K recs for a user
  POST /ratings/predict                — predict rating for (user, movie)
  GET  /users/{user_idx}/history       — movies rated by a user (from train split)
  GET  /movies/{movie_idx}             — metadata for a movie
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, SPLITS_DIR, PROCESSED_DIR
from src.serving.recommender import load_recommender, BaseRecommender
from src.evaluation.metrics import rmse, mae

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Movie Recommendation API",
    description="Serve recommendations from CF, SVD, and Two-Tower models.",
    version="0.1.0",
)

# ── Startup: load data once ───────────────────────────────────────────────────

_train_df: pd.DataFrame = None
_test_df:  pd.DataFrame = None
_movie_meta: pd.DataFrame = None   # movie_idx → title, genres


@app.on_event("startup")
def startup():
    global _train_df, _test_df, _movie_meta
    try:
        _train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
        _test_df  = pd.read_parquet(SPLITS_DIR / "test.parquet")
        _movie_meta = (
            _train_df[["movie_idx", "movie_id", "title", "genres"]]
            .drop_duplicates("movie_idx")
            .set_index("movie_idx")
        )
        print(f"Loaded train ({len(_train_df):,} rows) and test ({len(_test_df):,} rows)")
    except FileNotFoundError:
        print("WARNING: data splits not found — run preprocessing first.")


# ── Model cache ───────────────────────────────────────────────────────────────

_recommenders: dict[str, BaseRecommender] = {}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "svd")
AVAILABLE_MODELS = ["cf", "svd", "two_tower"]


def get_recommender(model_name: str) -> BaseRecommender:
    """Lazy-load and cache recommender instances."""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}.",
        )
    if model_name not in _recommenders:
        try:
            _recommenders[model_name] = load_recommender(
                model_name, train_df=_train_df
            )
        except (FileNotFoundError, Exception) as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' not available: {e}",
            )
    return _recommenders[model_name]


# ── Schemas ───────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_idx:     int           = Field(...,  ge=0, description="User index (0-based)")
    top_k:        int           = Field(10,   ge=1, le=100)
    model:        str           = Field("svd", description="'cf' | 'svd' | 'two_tower'")
    exclude_seen: bool          = Field(True)


class RecommendResponse(BaseModel):
    user_idx:     int
    model:        str
    top_k:        int
    recommendations: List[dict]   # [{movie_idx, title, genres}]


class PredictRequest(BaseModel):
    user_idx:  int = Field(..., ge=0)
    movie_idx: int = Field(..., ge=0)
    model:     str = Field("svd")


class PredictResponse(BaseModel):
    user_idx:        int
    movie_idx:       int
    predicted_rating: float
    model:           str


# ── Helpers ───────────────────────────────────────────────────────────────────

def enrich_movie(movie_idx: int) -> dict:
    base = {"movie_idx": movie_idx}
    if _movie_meta is not None and movie_idx in _movie_meta.index:
        row = _movie_meta.loc[movie_idx]
        base["title"]  = row.get("title", "Unknown")
        base["genres"] = row.get("genres", "")
    else:
        base["title"]  = "Unknown"
        base["genres"] = ""
    return base


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    available = []
    for m in AVAILABLE_MODELS:
        path_map = {
            "cf":        MODELS_DIR / "user_based_cf.pkl",
            "svd":       MODELS_DIR / "svd_model.pkl",
            "two_tower": MODELS_DIR / "two_tower.pt",
        }
        available.append({
            "name":    m,
            "trained": path_map[m].exists(),
        })
    return {"models": available}


@app.post("/recommendations", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    rec   = get_recommender(req.model)
    recs  = rec.recommend(req.user_idx, top_k=req.top_k,
                          exclude_seen=req.exclude_seen)
    items = [enrich_movie(idx) for idx in recs]
    return RecommendResponse(
        user_idx=req.user_idx,
        model=req.model,
        top_k=req.top_k,
        recommendations=items,
    )


@app.post("/ratings/predict", response_model=PredictResponse)
def predict_rating(req: PredictRequest):
    rec = get_recommender(req.model)

    # Only SVD / CF expose a predict() method
    predict_fn = getattr(rec, "_model", None)
    if predict_fn is None or not hasattr(predict_fn, "predict"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' does not support rating prediction.",
        )
    rating = float(predict_fn.predict(req.user_idx, req.movie_idx))
    return PredictResponse(
        user_idx=req.user_idx,
        movie_idx=req.movie_idx,
        predicted_rating=round(rating, 3),
        model=req.model,
    )


@app.get("/users/{user_idx}/history")
def user_history(
    user_idx: int,
    limit: int = Query(20, ge=1, le=200),
):
    if _train_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    rows = _train_df[_train_df["user_idx"] == user_idx].head(limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"User {user_idx} not found.")
    records = rows[["movie_idx", "title", "rating", "genres"]].to_dict(orient="records")
    return {"user_idx": user_idx, "n_ratings": len(rows), "history": records}


@app.get("/movies/{movie_idx}")
def movie_info(movie_idx: int):
    info = enrich_movie(movie_idx)
    if info["title"] == "Unknown":
        raise HTTPException(status_code=404, detail=f"Movie {movie_idx} not found.")
    return info


@app.get("/evaluate")
def evaluate_model(
    model: str = Query("svd"),
    n_users: int = Query(200, ge=10, le=2000),
    top_k:  int = Query(10, ge=1, le=50),
):
    """
    Quick evaluation of a model on the test split.
    Samples n_users, computes Precision/Recall/NDCG@top_k.
    """
    if _test_df is None:
        raise HTTPException(status_code=503, detail="Test data not loaded.")

    from src.evaluation.metrics import (
        evaluate_recommendations, build_ground_truth
    )

    rec      = get_recommender(model)
    gt       = build_ground_truth(_test_df, min_rating=3.5)
    user_ids = list(gt.keys())[:n_users]

    recs_dict = rec.recommend_batch(user_ids, top_k=top_k)
    results   = evaluate_recommendations(recs_dict, gt, k_values=[top_k])
    return results.reset_index().to_dict(orient="records")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)