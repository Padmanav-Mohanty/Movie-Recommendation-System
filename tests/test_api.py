"""
Integration tests for api/main.py using FastAPI's TestClient.

These tests use a mock recommender so no trained model artefacts are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "movie_idx": [0, 1, 1, 2, 0],
            "movie_id": [10, 20, 20, 30, 10],
            "title": ["Movie A", "Movie B", "Movie B", "Movie C", "Movie A"],
            "genres": ["Action", "Drama", "Drama", "Comedy", "Action"],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
        }
    )


@pytest.fixture(scope="module")
def mock_test_df(mock_train_df) -> pd.DataFrame:
    return mock_train_df.copy()


class MockRecommender:
    """Deterministic recommender for testing — returns first N movie indices."""

    def recommend(
        self, user_idx: int, top_k: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        return list(range(top_k))

    def recommend_batch(self, user_idxs, top_k=10, exclude_seen=True, seen_dict=None):
        return {u: self.recommend(u, top_k) for u in user_idxs}


@pytest.fixture(scope="module")
def client(mock_train_df, mock_test_df):
    """TestClient with all external I/O mocked out."""
    import api.main as main_module

    # Patch data loading and model loading at module level
    main_module._train_df = mock_train_df
    main_module._test_df = mock_test_df
    main_module._movie_meta = (
        mock_train_df[["movie_idx", "movie_id", "title", "genres"]]
        .drop_duplicates("movie_idx")
        .set_index("movie_idx")
    )
    main_module._recommenders = {}

    # Patch load_recommender to return the mock
    with patch("api.main.load_recommender", return_value=MockRecommender()):
        yield TestClient(main_module.app)


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_data_loaded_flag(self, client):
        r = client.get("/health")
        assert r.json()["data_loaded"] is True

    def test_response_time_header(self, client):
        r = client.get("/health")
        assert "X-Process-Time-Ms" in r.headers


# ── Models ────────────────────────────────────────────────────────────────────


class TestModels:
    def test_returns_list(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_all_models_present(self, client):
        r = client.get("/models")
        names = {m["name"] for m in r.json()["models"]}
        assert {"svd", "two_tower"} == names


# ── Recommendations ───────────────────────────────────────────────────────────


class TestRecommendations:
    def test_basic_request(self, client):
        r = client.post(
            "/recommendations", json={"user_idx": 0, "top_k": 3, "model": "svd"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["user_idx"] == 0
        assert body["model"] == "svd"
        assert len(body["recommendations"]) == 3

    def test_recommendation_schema(self, client):
        r = client.post(
            "/recommendations", json={"user_idx": 0, "top_k": 2, "model": "svd"}
        )
        recs = r.json()["recommendations"]
        for rec in recs:
            assert "movie_idx" in rec
            assert "title" in rec
            assert "genres" in rec

    def test_top_k_respected(self, client):
        for k in (1, 5, 10):
            r = client.post(
                "/recommendations", json={"user_idx": 0, "top_k": k, "model": "svd"}
            )
            assert len(r.json()["recommendations"]) == k

    def test_invalid_model_400(self, client):
        r = client.post(
            "/recommendations",
            json={"user_idx": 0, "top_k": 5, "model": "does_not_exist"},
        )
        assert r.status_code == 400

    def test_negative_user_idx_422(self, client):
        r = client.post(
            "/recommendations", json={"user_idx": -1, "top_k": 5, "model": "svd"}
        )
        assert r.status_code == 422

    def test_top_k_over_limit_422(self, client):
        r = client.post(
            "/recommendations", json={"user_idx": 0, "top_k": 9999, "model": "svd"}
        )
        assert r.status_code == 422


# ── User history ──────────────────────────────────────────────────────────────


class TestUserHistory:
    def test_known_user(self, client):
        r = client.get("/users/0/history")
        assert r.status_code == 200
        body = r.json()
        assert body["user_idx"] == 0
        assert "history" in body
        assert len(body["history"]) > 0

    def test_unknown_user_404(self, client):
        r = client.get("/users/99999/history")
        assert r.status_code == 404

    def test_limit_respected(self, client):
        r = client.get("/users/0/history?limit=1")
        assert r.status_code == 200
        assert len(r.json()["history"]) <= 1


# ── Movie info ────────────────────────────────────────────────────────────────


class TestMovieInfo:
    def test_known_movie(self, client):
        r = client.get("/movies/0")
        assert r.status_code == 200
        body = r.json()
        assert body["movie_idx"] == 0
        assert "title" in body
        assert "genres" in body

    def test_unknown_movie_404(self, client):
        r = client.get("/movies/99999")
        assert r.status_code == 404


# ── OpenAPI docs ──────────────────────────────────────────────────────────────


class TestOpenAPI:
    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "Movie Recommendation API"
