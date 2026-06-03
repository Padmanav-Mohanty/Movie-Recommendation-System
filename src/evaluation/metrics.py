"""
Ranking and rating-prediction metrics for recommendation systems.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


# ── Rating-prediction metrics ─────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ── Ranking metrics ───────────────────────────────────────────────────────────

def precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Average Precision@K (area under precision-recall curve up to K)."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.
    Assumes binary relevance (item is relevant or not).
    """
    def dcg(items):
        return sum(
            (1.0 / np.log2(i + 2))
            for i, item in enumerate(items)
            if item in relevant
        )

    top_k = recommended[:k]
    ideal = sorted(relevant)[:k]           # best possible ordering
    idcg  = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(top_k) / idcg


def hit_rate_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """1 if at least one relevant item is in top-K, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


def mrr_at_k(recommended: List[int], relevant: set, k: int) -> float:
    """Reciprocal rank of the first relevant item in top-K."""
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth:    Dict[int, set],
    k_values:        List[int] = None,
) -> pd.DataFrame:
    """
    Compute ranking metrics for every user across multiple K values.

    Parameters
    ----------
    recommendations : {user_idx: [ordered list of movie_idx]}
    ground_truth    : {user_idx: {set of relevant movie_idx}}
    k_values        : list of cutoffs, e.g. [5, 10, 20]

    Returns
    -------
    pd.DataFrame with one row per K, columns:
        precision, recall, ndcg, hit_rate, map, mrr
    """
    if k_values is None:
        from config import TOP_K
        k_values = TOP_K

    rows = []
    for k in k_values:
        metrics_per_user = []
        for user_idx, recs in recommendations.items():
            relevant = ground_truth.get(user_idx, set())
            if not relevant:
                continue
            metrics_per_user.append({
                "precision": precision_at_k(recs, relevant, k),
                "recall":    recall_at_k(recs, relevant, k),
                "ndcg":      ndcg_at_k(recs, relevant, k),
                "hit_rate":  hit_rate_at_k(recs, relevant, k),
                "ap":        average_precision_at_k(recs, relevant, k),
                "mrr":       mrr_at_k(recs, relevant, k),
            })

        if not metrics_per_user:
            continue

        agg = pd.DataFrame(metrics_per_user).mean()
        rows.append({
            "K":         k,
            "Precision": agg["precision"],
            "Recall":    agg["recall"],
            "NDCG":      agg["ndcg"],
            "HitRate":   agg["hit_rate"],
            "MAP":       agg["ap"],
            "MRR":       agg["mrr"],
            "n_users":   len(metrics_per_user),
        })

    return pd.DataFrame(rows).set_index("K")


def build_ground_truth(
    test_df: pd.DataFrame,
    min_rating: float = 3.5,
) -> Dict[int, set]:
    """
    Build a ground-truth dict from test split.
    A movie is 'relevant' if the user gave it >= min_rating.
    """
    gt = (
        test_df[test_df["rating"] >= min_rating]
        .groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )
    return gt


def coverage_at_k(
    recommendations: Dict[int, List[int]],
    n_items: int,
    k: int = 10,
) -> float:
    """Catalogue coverage: fraction of items that appear in any top-K list."""
    seen = set()
    for recs in recommendations.values():
        seen.update(recs[:k])
    return len(seen) / n_items if n_items > 0 else 0.0


def novelty_at_k(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, float],
    k: int = 10,
) -> float:
    """
    Mean self-information (novelty) of recommendations.
    item_popularity maps movie_idx → probability of being rated.
    """
    scores = []
    for recs in recommendations.values():
        for item in recs[:k]:
            p = item_popularity.get(item, 1e-9)
            scores.append(-np.log2(p + 1e-9))
    return float(np.mean(scores)) if scores else 0.0