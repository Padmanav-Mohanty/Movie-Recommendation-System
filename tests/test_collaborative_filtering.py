"""
Unit tests for src/models/collaborative_filtering.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.collaborative_filtering import UserBasedCF
from tests.conftest import N_USERS, N_MOVIES


class TestUserBasedCF:
    @pytest.fixture(scope="class")
    def fitted_model(self, interaction_matrix):
        model = UserBasedCF(n_neighbors=10)
        model.fit(interaction_matrix)
        return model

    # ── fit ───────────────────────────────────────────────────────────────────

    def test_fit_stores_matrix(self, fitted_model):
        assert fitted_model.interaction_mat is not None

    def test_fit_dimensions(self, fitted_model):
        assert fitted_model.n_users  == N_USERS
        assert fitted_model.n_movies == N_MOVIES

    def test_similarity_matrix_shape(self, fitted_model):
        assert fitted_model.user_sim.shape == (N_USERS, N_USERS)

    def test_self_similarity_is_zero(self, fitted_model):
        diag = np.diag(fitted_model.user_sim)
        np.testing.assert_array_equal(diag, 0.0)

    def test_similarity_in_range(self, fitted_model):
        assert fitted_model.user_sim.min() >= -1.0
        assert fitted_model.user_sim.max() <=  1.0

    # ── predict ───────────────────────────────────────────────────────────────

    def test_predict_returns_float(self, fitted_model):
        p = fitted_model.predict(0, 0)
        assert isinstance(p, float)

    def test_predict_in_rating_range(self, fitted_model, interaction_matrix):
        global_min = interaction_matrix.data.min()
        global_max = interaction_matrix.data.max()
        p = fitted_model.predict(0, 0)
        # Result may be weighted average, so should be close to valid range
        assert global_min - 1 <= p <= global_max + 1

    def test_predict_batch_length(self, fitted_model, train_df):
        sample = train_df.head(10)
        preds  = fitted_model.predict_batch(sample)
        assert len(preds) == 10

    # ── recommend ─────────────────────────────────────────────────────────────

    def test_recommend_length(self, fitted_model):
        recs = fitted_model.recommend(user_idx=0, top_k=5)
        assert len(recs) == 5

    def test_recommend_no_duplicates(self, fitted_model):
        recs = fitted_model.recommend(user_idx=0, top_k=10)
        assert len(recs) == len(set(recs))

    def test_recommend_excludes_seen(self, fitted_model, interaction_matrix):
        seen = interaction_matrix[0].indices.tolist()
        recs = fitted_model.recommend(user_idx=0, top_k=5, exclude_seen=True)
        assert not set(recs).intersection(seen)

    def test_recommend_valid_movie_indices(self, fitted_model):
        recs = fitted_model.recommend(user_idx=0, top_k=10)
        for r in recs:
            assert 0 <= r < N_MOVIES

    # ── persistence ───────────────────────────────────────────────────────────

    def test_save_load_roundtrip(self, fitted_model, tmp_path):
        path = tmp_path / "cf_model.pkl"
        fitted_model.save(path)
        loaded = UserBasedCF.load(path)
        assert loaded.n_users  == fitted_model.n_users
        assert loaded.n_movies == fitted_model.n_movies
        recs_orig   = fitted_model.recommend(0, top_k=5)
        recs_loaded = loaded.recommend(0, top_k=5)
        assert recs_orig == recs_loaded
