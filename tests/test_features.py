"""
Unit tests for src/features/build_features.py
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy.sparse import issparse

from src.features.build_features import (
    ALL_GENRES,
    build_interaction_matrix,
    build_item_features,
    build_user_features,
)


class TestUserFeatures:
    def test_one_row_per_user(self, train_df):
        uf = build_user_features(train_df)
        assert uf["user_idx"].is_unique

    def test_required_columns(self, train_df):
        uf = build_user_features(train_df)
        for col in ("user_idx", "mean_rating", "rating_std", "n_ratings"):
            assert col in uf.columns, f"Missing column: {col}"

    def test_genre_affinity_columns(self, train_df):
        uf = build_user_features(train_df)
        genre_cols = [c for c in uf.columns if c.startswith("genre_affinity_")]
        assert len(genre_cols) == len(ALL_GENRES)

    def test_no_nulls(self, train_df):
        uf = build_user_features(train_df)
        assert uf.isna().sum().sum() == 0

    def test_mean_rating_in_range(self, train_df):
        uf = build_user_features(train_df)
        assert (uf["mean_rating"] >= 0.5).all()
        assert (uf["mean_rating"] <= 5.0).all()

    def test_std_nonnegative(self, train_df):
        uf = build_user_features(train_df)
        assert (uf["rating_std"] >= 0).all()


class TestItemFeatures:
    def test_one_row_per_item(self, train_df):
        itf = build_item_features(train_df)
        assert itf["movie_idx"].is_unique

    def test_required_columns(self, train_df):
        itf = build_item_features(train_df)
        for col in ("movie_idx", "mean_rating", "rating_std", "n_ratings", "year"):
            assert col in itf.columns, f"Missing column: {col}"

    def test_genre_ohe_columns(self, train_df):
        itf = build_item_features(train_df)
        genre_cols = [c for c in itf.columns if c.startswith("genre_")]
        assert len(genre_cols) == len(ALL_GENRES)

    def test_genre_ohe_binary(self, train_df):
        itf = build_item_features(train_df)
        genre_cols = [c for c in itf.columns if c.startswith("genre_")]
        ohe = itf[genre_cols]
        assert ohe.isin([0, 1]).all().all()

    def test_no_nulls(self, train_df):
        itf = build_item_features(train_df)
        assert itf.isna().sum().sum() == 0

    def test_year_extraction(self):
        df = pd.DataFrame(
            {
                "movie_idx": [0, 1],
                "user_idx": [0, 0],
                "rating": [4.0, 3.0],
                "title": ["Toy Story (1995)", "Unknown Movie"],
                "genre_list": [["Animation"], ["Drama"]],
            }
        )
        itf = build_item_features(df)
        row_ts = itf[itf["movie_idx"] == 0].iloc[0]
        assert row_ts["year"] == pytest.approx(1995.0)
        row_uk = itf[itf["movie_idx"] == 1].iloc[0]
        assert row_uk["year"] == pytest.approx(0.0)


class TestInteractionMatrix:
    def test_shape(self, train_df):
        from tests.conftest import N_MOVIES, N_USERS

        mat = build_interaction_matrix(train_df, N_USERS, N_MOVIES)
        assert mat.shape == (N_USERS, N_MOVIES)

    def test_sparse(self, train_df):
        from tests.conftest import N_MOVIES, N_USERS

        mat = build_interaction_matrix(train_df, N_USERS, N_MOVIES)
        assert issparse(mat)

    def test_values_match(self, train_df):
        from tests.conftest import N_MOVIES, N_USERS

        mat = build_interaction_matrix(train_df, N_USERS, N_MOVIES)
        # All stored values should be valid ratings
        assert (mat.data >= 0.5).all()
        assert (mat.data <= 5.0).all()
