"""
Unit tests for src/evaluation/metrics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    average_precision,
    build_ground_truth,
    catalogue_coverage_at_k,
    evaluate_recommendations,
    hit_rate_at_k,
    mae,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)

# ── Rating-prediction metrics ─────────────────────────────────────────────────


class TestRatingPrediction:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([3.0, 3.0, 3.0, 3.0])
        y_pred = np.array([2.0, 4.0, 2.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([3.0, 3.0])
        y_pred = np.array([2.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_rmse_always_nonnegative(self, rng=None):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0.5, 5.0, 100)
        y_pred = rng.uniform(0.5, 5.0, 100)
        assert rmse(y_true, y_pred) >= 0.0

    def test_rmse_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        assert rmse(a, b) == pytest.approx(rmse(b, a))


# ── Per-user ranking metrics ──────────────────────────────────────────────────


class TestPrecisionRecall:
    REC = [1, 2, 3, 4, 5]
    RELEVANT = [2, 4, 6]

    def test_precision_at_k_basic(self):
        # 2 hits in top-4: items 2 and 4
        assert precision_at_k(self.REC, self.RELEVANT, k=4) == pytest.approx(2 / 4)

    def test_recall_at_k_basic(self):
        # 2 of 3 relevant items found in top-5
        assert recall_at_k(self.REC, self.RELEVANT, k=5) == pytest.approx(2 / 3)

    def test_precision_at_k_zero(self):
        assert precision_at_k([10, 11, 12], [1, 2, 3], k=3) == pytest.approx(0.0)

    def test_precision_at_k_perfect(self):
        assert precision_at_k([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)

    def test_recall_empty_relevant(self):
        assert recall_at_k([1, 2, 3], [], k=3) == pytest.approx(0.0)

    def test_precision_k_zero(self):
        assert precision_at_k([1, 2], [1], k=0) == pytest.approx(0.0)


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k([1, 2, 3], [3, 5], k=3) == 1.0

    def test_no_hit(self):
        assert hit_rate_at_k([1, 2, 3], [4, 5], k=3) == 0.0

    def test_hit_outside_k(self):
        # relevant item is at position 4, k=3 → no hit
        assert hit_rate_at_k([1, 2, 3, 4], [4], k=3) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        # All relevant items at top of list
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == pytest.approx(1.0)

    def test_ndcg_zero(self):
        assert ndcg_at_k([1, 2, 3], [4, 5, 6], k=3) == pytest.approx(0.0)

    def test_ndcg_ordering_matters(self):
        # Relevant item first → higher NDCG than relevant item last
        relevant = [1]
        ndcg_first = ndcg_at_k([1, 2, 3, 4], relevant, k=4)
        ndcg_last = ndcg_at_k([2, 3, 4, 1], relevant, k=4)
        assert ndcg_first > ndcg_last

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k([1, 2, 3], [], k=3) == pytest.approx(0.0)

    def test_ndcg_bounded(self):
        rng = np.random.default_rng(1)
        rec = rng.integers(0, 100, 20).tolist()
        rel = rng.integers(0, 100, 10).tolist()
        score = ndcg_at_k(rec, rel, k=10)
        assert 0.0 <= score <= 1.0


# ── Aggregate metrics ─────────────────────────────────────────────────────────


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_zero(self):
        assert average_precision([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert average_precision([1, 2], []) == pytest.approx(0.0)

    def test_partial(self):
        # [1,2,3,4,5], relevant=[1,3,5]
        # hits at positions 1,3,5 → P@1=1, P@3=2/3, P@5=3/5
        # AP = (1 + 2/3 + 3/5) / 3
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert average_precision([1, 2, 3, 4, 5], [1, 3, 5]) == pytest.approx(
            expected, rel=1e-6
        )


class TestMAP:
    def test_perfect(self):
        recs = {0: [1, 2], 1: [3, 4]}
        gt = {0: [1, 2], 1: [3, 4]}
        assert mean_average_precision(recs, gt) == pytest.approx(1.0)

    def test_zero(self):
        recs = {0: [5, 6], 1: [7, 8]}
        gt = {0: [1, 2], 1: [3, 4]}
        assert mean_average_precision(recs, gt) == pytest.approx(0.0)


class TestMRR:
    def test_first_hit_at_rank_1(self):
        recs = {0: [1, 2, 3]}
        gt = {0: [1]}
        assert mrr(recs, gt) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        recs = {0: [9, 1, 2, 3]}
        gt = {0: [1]}
        assert mrr(recs, gt) == pytest.approx(0.5)

    def test_no_hit(self):
        recs = {0: [9, 8, 7]}
        gt = {0: [1, 2]}
        assert mrr(recs, gt) == pytest.approx(0.0)


# ── Beyond-accuracy metrics ───────────────────────────────────────────────────


class TestCoverage:
    def test_full_coverage(self):
        recs = {0: [0, 1, 2], 1: [3, 4, 5]}
        assert catalogue_coverage_at_k(recs, n_items=6, k=3) == pytest.approx(1.0)

    def test_partial_coverage(self):
        recs = {0: [0, 1], 1: [0, 1]}
        # only 2 out of 10 unique items recommended
        assert catalogue_coverage_at_k(recs, n_items=10, k=2) == pytest.approx(0.2)

    def test_zero_items(self):
        assert catalogue_coverage_at_k({0: [1]}, n_items=0, k=1) == pytest.approx(0.0)


class TestNovelty:
    def test_novelty_positive(self):
        pop = {0: 1, 1: 2, 2: 3}
        recs = {0: [0, 1, 2]}
        score = novelty_at_k(recs, pop, n_users=100, k=3)
        assert score > 0.0

    def test_novelty_decreases_with_popularity(self):
        # Lower popularity → higher novelty
        pop_niche = {0: 1}
        pop_popular = {0: 100}
        recs = {0: [0]}
        n = novelty_at_k(recs, pop_niche, n_users=1000, k=1)
        p = novelty_at_k(recs, pop_popular, n_users=1000, k=1)
        assert n > p


# ── Helpers ───────────────────────────────────────────────────────────────────


class TestBuildGroundTruth:
    def test_filters_by_min_rating(self):
        df = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1],
                "movie_idx": [0, 1, 2, 3],
                "rating": [5.0, 2.0, 4.0, 1.0],
            }
        )
        gt = build_ground_truth(df, min_rating=4.0)
        assert gt == {0: [0], 1: [2]}

    def test_empty_if_no_high_ratings(self):
        df = pd.DataFrame({"user_idx": [0], "movie_idx": [0], "rating": [1.0]})
        gt = build_ground_truth(df, min_rating=4.0)
        assert gt == {}


class TestEvaluateRecommendations:
    RECS = {0: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]}
    GT = {0: [2, 4], 1: [6, 10]}

    def test_returns_dataframe(self):
        df = evaluate_recommendations(self.RECS, self.GT, k_values=[5])
        assert isinstance(df, pd.DataFrame)

    def test_index_is_k(self):
        df = evaluate_recommendations(self.RECS, self.GT, k_values=[5, 10])
        assert list(df.index) == [5, 10]

    def test_metric_columns_present(self):
        df = evaluate_recommendations(self.RECS, self.GT, k_values=[5])
        for col in ("Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "MAP", "MRR"):
            assert col in df.columns, f"Missing column: {col}"

    def test_coverage_column_optional(self):
        df = evaluate_recommendations(self.RECS, self.GT, k_values=[5])
        assert "Coverage@K" not in df.columns

        df2 = evaluate_recommendations(self.RECS, self.GT, k_values=[5], n_items=20)
        assert "Coverage@K" in df2.columns

    def test_all_metrics_bounded(self):
        df = evaluate_recommendations(self.RECS, self.GT, k_values=[5, 10])
        for col in ("Precision@K", "Recall@K", "NDCG@K", "HitRate@K"):
            assert (df[col] >= 0).all() and (df[col] <= 1).all(), f"{col} out of [0,1]"
