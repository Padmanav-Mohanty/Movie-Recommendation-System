"""
Shared pytest fixtures.

All fixtures are scoped to the session where possible to avoid
expensive re-computation in every test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# ── Make repo root importable ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Tiny synthetic dataset ────────────────────────────────────────────────────

N_USERS  = 50
N_MOVIES = 30
N_RATINGS = 500
RATING_SCALE = (0.5, 5.0)
RANDOM_SEED  = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def ratings_df(rng) -> pd.DataFrame:
    """Synthetic ratings dataframe that mirrors the real schema."""
    user_ids  = rng.integers(0, N_USERS,  size=N_RATINGS)
    movie_ids = rng.integers(0, N_MOVIES, size=N_RATINGS)
    ratings   = rng.uniform(0.5, 5.0, size=N_RATINGS).round(1)

    genres_pool = ["Action|Adventure", "Drama", "Comedy", "Sci-Fi", "Romance"]
    genres = [genres_pool[i % len(genres_pool)] for i in range(N_RATINGS)]
    titles = [f"Movie {mid} (200{mid % 10})" for mid in movie_ids]

    df = pd.DataFrame({
        "user_id":   user_ids,
        "movie_id":  movie_ids,
        "user_idx":  user_ids,
        "movie_idx": movie_ids,
        "rating":    ratings,
        "title":     titles,
        "genres":    genres,
        "genre_list": [g.split("|") for g in genres],
    })
    return df.drop_duplicates(["user_idx", "movie_idx"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def train_df(ratings_df) -> pd.DataFrame:
    return ratings_df.sample(frac=0.8, random_state=RANDOM_SEED).reset_index(drop=True)


@pytest.fixture(scope="session")
def test_df(ratings_df, train_df) -> pd.DataFrame:
    return ratings_df.drop(train_df.index).reset_index(drop=True)


@pytest.fixture(scope="session")
def interaction_matrix(train_df) -> csr_matrix:
    return csr_matrix(
        (train_df["rating"].values,
         (train_df["user_idx"].values, train_df["movie_idx"].values)),
        shape=(N_USERS, N_MOVIES),
    )


@pytest.fixture(scope="session")
def user_features(train_df) -> pd.DataFrame:
    from src.features.build_features import build_user_features
    return build_user_features(train_df)


@pytest.fixture(scope="session")
def item_features(train_df) -> pd.DataFrame:
    from src.features.build_features import build_item_features
    return build_item_features(train_df)
