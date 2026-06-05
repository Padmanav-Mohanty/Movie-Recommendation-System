"""
FastAPI application — Movie Recommendation System
=================================================

Endpoints
---------
  GET  /health                       — liveness + readiness check
  GET  /models                       — list available trained models
  POST /recommendations              — top-K recs for a user
  POST /ratings/predict              — predicted rating for (user, movie)
  GET  /users/{user_idx}/history     — movies rated by a user
  GET  /movies/{movie_idx}           — movie metadata
  GET  /evaluate                     — ranking evaluation on the test split

Run locally
-----------
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlretrieve

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Make src/ importable when running from repo root ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, SPLITS_DIR, PROCESSED_DIR
from src.evaluation.metrics import build_ground_truth, evaluate_recommendations
from src.serving.recommender import BaseRecommender, load_recommender

# ── Model download (Render: fetch from MODEL_ARTIFACT_BASE_URL at startup) ────
# ── Model download (fetch from MODEL_ARTIFACT_BASE_URL at startup) ────────────
def _download_model_artifacts() -> None:
    """
    If MODEL_ARTIFACT_BASE_URL is set, download any missing files listed
    in MODEL_ARTIFACTS (comma-separated) into the correct local directory:
      - *_model.pkl / *.pt / *.bin  → MODELS_DIR
      - *.parquet / *.npz           → matched by filename to SPLITS_DIR or PROCESSED_DIR
    Safe to call every startup — skips files that already exist on disk.
    """
    import urllib.request

    base_url = os.getenv("MODEL_ARTIFACT_BASE_URL", "").rstrip("/")
    artifacts = os.getenv("MODEL_ARTIFACTS", "")

    if not base_url or not artifacts:
        log.info("MODEL_ARTIFACT_BASE_URL not set — skipping artifact download.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Map filenames to their target directories
    splits_files = {"train.parquet", "test.parquet", "val.parquet"}
    processed_files = {
        "interaction_matrix.npz", "item_features.parquet",
        "movie2idx.parquet", "user2idx.parquet", "user_features.parquet"
    }

    for filename in [f.strip() for f in artifacts.split(",") if f.strip()]:
        if filename in splits_files:
            dest = SPLITS_DIR / filename
        elif filename in processed_files:
            dest = PROCESSED_DIR / filename
        else:
            dest = MODELS_DIR / filename

        if dest.exists():
            log.info("Artifact already present, skipping: %s", filename)
            continue
        url = f"{base_url}/{filename}"
        log.info("Downloading artifact: %s → %s", url, dest)
        try:
            urllib.request.urlretrieve(url, dest)
            log.info("Downloaded: %s (%s bytes)", filename, dest.stat().st_size)
        except Exception as exc:
            log.error("Failed to download %s: %s", filename, exc)
            raise RuntimeError(f"Could not fetch required model artifact: {filename}") from exc
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("api")

# ── Application state ─────────────────────────────────────────────────────────
_train_df: pd.DataFrame | None = None
_test_df: pd.DataFrame | None = None
_movie_meta: pd.DataFrame | None = None  # movie_idx → title, genres
_recommenders: dict[str, BaseRecommender] = {}

# cf excluded (6.9GB model); two_tower available but memory-heavy on free tier
AVAILABLE_MODELS = ["svd", "two_tower"]
LIGHT_MODELS = ["svd"]  # pre-warm only these on free tier
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "svd")
MODEL_ARTIFACT_FILENAMES = {
    "cf": "user_based_cf.pkl",
    "svd": "svd_model.pkl",
    "two_tower": "two_tower.pt",
    "faiss": "faiss_index.bin",
}
_startup_started_at = time.time()
_startup_complete = False
_startup_error: str | None = None



# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared data on startup; release on shutdown."""
    global _train_df, _test_df, _movie_meta
    import gc

    _download_model_artifacts()

    try:
        # Load only essential columns to minimise memory on free-tier hosts
        train_columns = ["user_idx", "movie_idx", "movie_id", "title", "genres", "rating"]
        try:
            _train_df = pd.read_parquet(
                SPLITS_DIR / "train.parquet",
                columns=[*train_columns, "timestamp"],
            )
        except Exception as exc:
            if "timestamp" not in str(exc):
                raise
            _train_df = pd.read_parquet(SPLITS_DIR / "train.parquet", columns=train_columns)
        _test_df = pd.read_parquet(
            SPLITS_DIR / "test.parquet",
            columns=["user_idx", "movie_idx", "rating"],
        )
        _movie_meta = (
            _train_df[["movie_idx", "movie_id", "title", "genres"]]
            .drop_duplicates("movie_idx")
            .set_index("movie_idx")
        )
        gc.collect()
        log.info(
            "Loaded train (%s rows) and test (%s rows)",
            f"{len(_train_df):,}",
            f"{len(_test_df):,}",
        )
    except FileNotFoundError:
        log.warning("Data splits not found — run preprocessing first.")
    yield
    _recommenders.clear()
    log.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "Production-grade REST API serving CF, SVD, and Two-Tower "
        "recommendations from the MovieLens dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request-timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"
    log.info(
        "%s %s  %s  %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ── Model loader (lazy, cached) ───────────────────────────────────────────────
def _configured_model_artifacts() -> list[str]:
    raw = os.getenv("MODEL_ARTIFACTS", MODEL_ARTIFACT_FILENAMES.get(DEFAULT_MODEL, ""))
    artifacts = [item.strip() for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(artifacts))


def download_model_artifacts() -> None:
    """Download missing model artifacts from an HTTP/S3-presigned base URL."""
    base_url = os.getenv("MODEL_ARTIFACT_BASE_URL", "").strip()
    if not base_url:
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in _configured_model_artifacts():
        target = MODELS_DIR / filename
        if target.exists():
            log.info("Model artifact already exists: %s", target)
            continue

        artifact_url = urljoin(f"{base_url.rstrip('/')}/", filename)
        log.info("Downloading model artifact %s", filename)
        try:
            urlretrieve(artifact_url, target)
        except (HTTPError, URLError, OSError) as exc:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download model artifact {filename}: {exc}") from exc

def get_recommender(model_name: str) -> BaseRecommender:
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_name}'. Choose from {AVAILABLE_MODELS}.",
        )
    if model_name not in _recommenders:
        try:
            _recommenders[model_name] = load_recommender(model_name, train_df=_train_df)
            log.info("Loaded recommender: %s", model_name)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' not yet trained. {exc}",
            ) from exc
        except Exception as exc:
            log.exception("Failed to load model '%s'", model_name)
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model_name}' could not be loaded: {exc}",
            ) from exc
    return _recommenders[model_name]


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    user_idx: int = Field(..., ge=0, description="User index (0-based)")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    model: str = Field("svd", description="'cf' | 'svd' | 'two_tower'")
    exclude_seen: bool = Field(True, description="Exclude already-rated movies")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_idx": 0,
                "top_k": 10,
                "model": "svd",
                "exclude_seen": True,
            }
        }
    }


class MovieResult(BaseModel):
    movie_idx: int
    title: str
    genres: str
    score: float = 0.0


class RecommendResponse(BaseModel):
    user_idx: int
    model: str
    top_k: int
    recommendations: list[MovieResult]


class PredictRequest(BaseModel):
    user_idx: int = Field(..., ge=0)
    movie_idx: int = Field(..., ge=0)
    model: str = Field("svd")

    model_config = {
        "json_schema_extra": {"example": {"user_idx": 0, "movie_idx": 50, "model": "svd"}}
    }


class PredictResponse(BaseModel):
    user_idx: int
    movie_idx: int
    predicted_rating: float
    model: str


class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    models_ready: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────


def enrich_movie(movie_idx: int) -> MovieResult:
    if _movie_meta is not None and movie_idx in _movie_meta.index:
        row = _movie_meta.loc[movie_idx]
        return MovieResult(
            movie_idx=movie_idx,
            title=str(row.get("title", "Unknown")),
            genres=str(row.get("genres", "")),
        )
    return MovieResult(movie_idx=movie_idx, title="Unknown", genres="")


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness + lightweight readiness check."""
    return HealthResponse(
        status="ok",
        data_loaded=_train_df is not None,
        models_ready=list(_recommenders.keys()),
    )


@app.get("/models", tags=["System"])
def list_models():
    """List all models and whether their artefacts exist on disk."""
    path_map = {
        "cf": MODELS_DIR / "user_based_cf.pkl",
        "svd": MODELS_DIR / "svd_model.pkl",
        "two_tower": MODELS_DIR / "two_tower.pt",
    }
    return {
        "models": [
            {"name": m, "trained": path_map[m].exists(), "loaded": m in _recommenders}
            for m in AVAILABLE_MODELS
        ]
    }


@app.post("/recommendations", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    rec   = get_recommender(req.model)
    recs  = rec.recommend(req.user_idx, top_k=req.top_k, exclude_seen=req.exclude_seen)

    items = []
    for idx in recs:
        movie = enrich_movie(idx)
        score = 0.0
        inner = getattr(rec, "_model", None)
        if inner and hasattr(inner, "predict"):
            try:
                score = round(float(inner.predict(req.user_idx, idx)), 3)
            except Exception as e:
                log.warning("Score prediction failed for user=%s movie=%s: %s", req.user_idx, idx, e)
                score = 0.0
        items.append(MovieResult(**movie.model_dump(), score=score))

    return RecommendResponse(
        user_idx=req.user_idx,
        model=req.model,
        top_k=req.top_k,
        recommendations=items,
    )

@app.post("/ratings/predict", response_model=PredictResponse, tags=["Recommendations"])
def predict_rating(req: PredictRequest):
    """Predict the rating a user would give a specific movie."""
    rec = get_recommender(req.model)

    inner_model = getattr(rec, "_model", None)
    if inner_model is None or not hasattr(inner_model, "predict"):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' does not expose a rating-prediction method.",
        )
    rating = float(inner_model.predict(req.user_idx, req.movie_idx))
    return PredictResponse(
        user_idx=req.user_idx,
        movie_idx=req.movie_idx,
        predicted_rating=round(rating, 3),
        model=req.model,
    )


@app.get("/users/{user_idx}/history", tags=["Users"])
def user_history(
    user_idx: int,
    limit: int = Query(20, ge=1, le=200),
):
    """Return movies rated by a user (from the training split)."""
    if _train_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    rows = _train_df[_train_df["user_idx"] == user_idx]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"User {user_idx} not found.")
    if "timestamp" in rows.columns:
        rows = rows.sort_values("timestamp", ascending=False, kind="stable")
    rows = rows.head(limit)
    records = rows[["movie_idx", "title", "rating", "genres"]].to_dict(orient="records")
    return {"user_idx": user_idx, "n_ratings": len(rows), "history": records}


@app.get("/movies/{movie_idx}", tags=["Movies"])
def movie_info(movie_idx: int):
    """Return metadata for a single movie."""
    info = enrich_movie(movie_idx)
    if info.title == "Unknown":
        raise HTTPException(status_code=404, detail=f"Movie {movie_idx} not found.")
    return info.model_dump()


@app.get("/evaluate", tags=["Evaluation"])
def evaluate_model(
    model: str = Query("svd", description="Model to evaluate"),
    n_users: int = Query(200, ge=10, le=2000),
    top_k: int = Query(10, ge=1, le=50),
    min_rating: float = Query(3.5, ge=0.5, le=5.0, description="Min rating to count as 'relevant'"),
):
    """
    Evaluate a model on the held-out test split.

    Samples up to ``n_users`` users with at least one relevant item
    and returns Precision/Recall/NDCG/HitRate@K, MAP, and MRR.
    """
    if _test_df is None:
        raise HTTPException(status_code=503, detail="Test data not loaded.")

    rec = get_recommender(model)
    gt = build_ground_truth(_test_df, min_rating=min_rating)

    if not gt:
        raise HTTPException(
            status_code=422,
            detail=f"No users have ratings >= {min_rating} in the test split.",
        )

    user_ids = list(gt.keys())[:n_users]
    recs_dict = rec.recommend_batch(user_ids, top_k=top_k)

    # Optional beyond-accuracy metrics
    n_items = int(_movie_meta.index.max()) + 1 if _movie_meta is not None else None

    results_df = evaluate_recommendations(
        recs_dict,
        gt,
        k_values=[top_k],
        n_items=n_items,
    )
    return {
        "model": model,
        "n_users": len(user_ids),
        "top_k": top_k,
        "metrics": results_df.reset_index().to_dict(orient="records"),
    }


# ── A/B Testing ───────────────────────────────────────────────────────────────


class ABTestRequest(BaseModel):
    user_idx: int = Field(..., ge=0, description="User index (0-based)")
    top_k: int = Field(10, ge=1, le=100)
    model_a: str = Field("svd", description="First model: 'cf' | 'svd' | 'two_tower'")
    model_b: str = Field("two_tower", description="Second model: 'svd' | 'two_tower'")
    exclude_seen: bool = Field(True)

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_idx": 42,
                "top_k": 10,
                "model_a": "svd",
                "model_b": "cf",
                "exclude_seen": True,
            }
        }
    }


class ABTestResponse(BaseModel):
    user_idx: int
    top_k: int
    model_a: str
    model_b: str
    results_a: list[MovieResult]
    results_b: list[MovieResult]
    overlap: list[MovieResult]
    overlap_pct: float
    unique_to_a: list[MovieResult]
    unique_to_b: list[MovieResult]


@app.post("/ab-test", response_model=ABTestResponse, tags=["Recommendations"])
def ab_test(req: ABTestRequest):
    """
    Compare two models side by side for the same user.

    Returns recommendations from both models plus an overlap analysis —
    which movies both models agree on, and which are unique to each.
    Overlap % is a diversity signal: low overlap means the models are
    capturing different signals and may complement each other.
    """
    if req.model_a == req.model_b:
        raise HTTPException(
            status_code=400,
            detail="model_a and model_b must be different.",
        )

    rec_a = get_recommender(req.model_a)
    rec_b = get_recommender(req.model_b)

    recs_a = rec_a.recommend(req.user_idx, top_k=req.top_k, exclude_seen=req.exclude_seen)
    recs_b = rec_b.recommend(req.user_idx, top_k=req.top_k, exclude_seen=req.exclude_seen)

    set_a, set_b = set(recs_a), set(recs_b)
    overlap_idxs = set_a & set_b
    unique_a_idxs = set_a - set_b
    unique_b_idxs = set_b - set_a
    overlap_pct = len(overlap_idxs) / req.top_k * 100

    return ABTestResponse(
        user_idx=req.user_idx,
        top_k=req.top_k,
        model_a=req.model_a,
        model_b=req.model_b,
        results_a=[enrich_movie(i) for i in recs_a],
        results_b=[enrich_movie(i) for i in recs_b],
        overlap=[enrich_movie(i) for i in overlap_idxs],
        overlap_pct=round(overlap_pct, 1),
        unique_to_a=[enrich_movie(i) for i in unique_a_idxs],
        unique_to_b=[enrich_movie(i) for i in unique_b_idxs],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )
