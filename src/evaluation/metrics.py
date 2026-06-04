"""
Evaluation metrics for the Movie Recommendation System.

Rating-prediction : RMSE, MAE
Ranking           : Precision@K, Recall@K, NDCG@K, Hit Rate@K, MAP, MRR
Beyond-accuracy   : Catalogue Coverage@K, Novelty@K
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── Rating-prediction ─────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Per-user ranking metrics ──────────────────────────────────────────────────

def precision_at_k(recommended: List[int],
                   relevant: List[int], k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int],
                relevant: List[int], k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def hit_rate_at_k(recommended: List[int],
                  relevant: List[int], k: int) -> float:
    """1 if any of top-K recommendations is relevant, else 0."""
    return float(bool(set(recommended[:k]) & set(relevant)))


def ndcg_at_k(recommended: List[int],
              relevant: List[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K."""
    rec_k = recommended[:k]
    dcg   = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(rec_k)
        if item in set(relevant)
    )
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: List[int],
                      relevant: List[int]) -> float:
    """Average Precision (area under the precision-recall curve)."""
    relevant_set = set(relevant)
    hits, score  = 0, 0.0
    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0


# ── Aggregate ranking metrics ─────────────────────────────────────────────────

def mean_average_precision(recommendations: Dict[int, List[int]],
                           ground_truth: Dict[int, List[int]]) -> float:
    """Mean Average Precision across all users."""
    return float(np.mean([
        average_precision(recommendations.get(u, []), ground_truth.get(u, []))
        for u in ground_truth
    ]))


def mrr(recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, List[int]]) -> float:
    """Mean Reciprocal Rank."""
    rr_scores = []
    for u, relevant in ground_truth.items():
        relevant_set = set(relevant)
        rr = 0.0
        for i, item in enumerate(recommendations.get(u, [])):
            if item in relevant_set:
                rr = 1.0 / (i + 1)
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores))


# ── Beyond-accuracy metrics ───────────────────────────────────────────────────

def catalogue_coverage_at_k(recommendations: Dict[int, List[int]],
                             n_items: int, k: int) -> float:
    """
    Percentage of the catalogue recommended to at least one user.

    Parameters
    ----------
    recommendations : user_idx → list of movie_idx
    n_items         : total number of items in the catalogue
    k               : cutoff
    """
    recommended_items: set[int] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / n_items if n_items > 0 else 0.0


def novelty_at_k(recommendations: Dict[int, List[int]],
                 item_popularity: Dict[int, int],
                 n_users: int, k: int) -> float:
    """
    Mean Self-Information (novelty) of top-K recommendations.
    High novelty → model recommends less-popular / more surprising items.

    novelty(i) = -log2( pop(i) / n_users )
    """
    novelty_scores = []
    for recs in recommendations.values():
        for item in recs[:k]:
            pop = item_popularity.get(item, 1)
            novelty_scores.append(-np.log2(pop / n_users + 1e-10))
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0


# ── Convenience helpers ───────────────────────────────────────────────────────

def build_ground_truth(df: pd.DataFrame,
                       min_rating: float = 4.0) -> Dict[int, List[int]]:
    """
    Build a ground-truth dict from a ratings DataFrame.

    Returns
    -------
    {user_idx: [movie_idx, ...]}  — items rated >= min_rating
    """
    return (
        df[df["rating"] >= min_rating]
        .groupby("user_idx")["movie_idx"]
        .apply(list)
        .to_dict()
    )


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = (5, 10, 20),
    n_items: Optional[int] = None,
    item_popularity: Optional[Dict[int, int]] = None,
    n_users: Optional[int] = None,
) -> pd.DataFrame:
    """
    Full ranking evaluation across multiple K values.

    Returns a DataFrame indexed by K with columns:
        Precision@K, Recall@K, NDCG@K, HitRate@K, MAP, MRR,
        Coverage@K (if n_items provided), Novelty@K (if item_popularity provided)
    """
    results = []
    for k in k_values:
        p_scores, r_scores, n_scores, hr_scores = [], [], [], []

        for u, relevant in ground_truth.items():
            recs = recommendations.get(u, [])
            p_scores.append(precision_at_k(recs, relevant, k))
            r_scores.append(recall_at_k(recs, relevant, k))
            n_scores.append(ndcg_at_k(recs, relevant, k))
            hr_scores.append(hit_rate_at_k(recs, relevant, k))

        row: Dict[str, float] = {
            "K":           k,
            "Precision@K": float(np.mean(p_scores)),
            "Recall@K":    float(np.mean(r_scores)),
            "NDCG@K":      float(np.mean(n_scores)),
            "HitRate@K":   float(np.mean(hr_scores)),
            "MAP":         mean_average_precision(recommendations, ground_truth),
            "MRR":         mrr(recommendations, ground_truth),
        }

        if n_items is not None:
            row["Coverage@K"] = catalogue_coverage_at_k(
                recommendations, n_items, k
            )
        if item_popularity is not None and n_users is not None:
            row["Novelty@K"] = novelty_at_k(
                recommendations, item_popularity, n_users, k
            )

        results.append(row)

    return pd.DataFrame(results).set_index("K")


# Alias kept for backwards-compat with older callers
evaluate_ranking = evaluate_recommendations


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import SPLITS_DIR, MODELS_DIR
    import pickle

    print("Loading test set and SVD model...")
    test  = pd.read_parquet(SPLITS_DIR / "test.parquet")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")

    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        model = pickle.load(f)

    ground_truth = build_ground_truth(test, min_rating=4.0)

    print(f"Evaluating on {min(500, len(ground_truth))} users...")
    recommendations: Dict[int, List[int]] = {}
    seen_per_user = train.groupby("user_idx")["movie_idx"].apply(list).to_dict()

    for i, user_idx in enumerate(list(ground_truth.keys())[:500]):
        seen = seen_per_user.get(user_idx, [])
        recommendations[user_idx] = model.recommend(
            user_idx=user_idx, top_k=20, seen_movie_idxs=seen
        )
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/500 users done")

    # Rating-prediction metrics
    known_users  = set(train["user_idx"])
    known_movies = set(train["movie_idx"])
    test_filtered = test[
        test["user_idx"].isin(known_users) &
        test["movie_idx"].isin(known_movies)
    ].sample(5000, random_state=42)

    preds  = model.predict_batch(test_filtered)
    y_true = test_filtered["rating"].values

    print(f"\nRating Prediction (SVD)")
    print(f"  RMSE : {rmse(y_true, preds):.4f}")
    print(f"  MAE  : {mae(y_true, preds):.4f}")

    print(f"\nRanking Metrics (SVD) @ top-500 users")
    ranking_df = evaluate_recommendations(recommendations, ground_truth)
    print(ranking_df.to_string())
