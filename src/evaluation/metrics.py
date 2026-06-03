import numpy as np
import pandas as pd
from typing import List, Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: List[int],
                   relevant: List[int], k: int) -> float:
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List[int],
                relevant: List[int], k: int) -> float:
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: List[int],
              relevant: List[int], k: int) -> float:
    rec_k = recommended[:k]
    dcg   = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(rec_k)
        if item in set(relevant)
    )
    idcg  = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(recommended: List[int],
                      relevant: List[int]) -> float:
    relevant_set = set(relevant)
    hits, score  = 0, 0.0
    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0


def mean_average_precision(recommendations: Dict[int, List[int]],
                           ground_truth:    Dict[int, List[int]]) -> float:
    return float(np.mean([
        average_precision(recommendations.get(u, []), ground_truth.get(u, []))
        for u in ground_truth
    ]))


def mrr(recommendations: Dict[int, List[int]],
        ground_truth:    Dict[int, List[int]]) -> float:
    rr_scores = []
    for u, relevant in ground_truth.items():
        relevant_set = set(relevant)
        rr = 0.0
        for i, item in enumerate(recommendations.get(u, [])):
            if item in relevant_set:
                rr = 1 / (i + 1)
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores))


def evaluate_ranking(recommendations: Dict[int, List[int]],
                     ground_truth:    Dict[int, List[int]],
                     k_values: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Full ranking evaluation across multiple K values.
    Returns a DataFrame with Precision, Recall, NDCG per K.
    """
    results = []
    for k in k_values:
        p_scores, r_scores, n_scores = [], [], []
        for u, relevant in ground_truth.items():
            recs = recommendations.get(u, [])
            p_scores.append(precision_at_k(recs, relevant, k))
            r_scores.append(recall_at_k(recs, relevant, k))
            n_scores.append(ndcg_at_k(recs, relevant, k))
        results.append({
            "K":           k,
            "Precision@K": np.mean(p_scores),
            "Recall@K":    np.mean(r_scores),
            "NDCG@K":      np.mean(n_scores),
        })

    df = pd.DataFrame(results).set_index("K")
    df["MAP"]  = mean_average_precision(recommendations, ground_truth)
    df["MRR"]  = mrr(recommendations, ground_truth)
    return df


if __name__ == "__main__":
    from config import SPLITS_DIR, MODELS_DIR, PROCESSED_DIR
    from src.models.matrix_factorization import MatrixFactorization
    import pickle

    print("Loading test set and model...")
    test  = pd.read_parquet(SPLITS_DIR / "test.parquet")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")

    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Build ground truth — items rated >= 4.0 are "relevant"
    ground_truth = (
        test[test["rating"] >= 4.0]
        .groupby("user_idx")["movie_idx"]
        .apply(list)
        .to_dict()
    )

    # Generate recommendations for users in ground truth
    print(f"Evaluating on {len(ground_truth)} users...")
    recommendations = {}
    seen_per_user   = train.groupby("user_idx")["movie_idx"].apply(list).to_dict()

    for i, user_idx in enumerate(list(ground_truth.keys())[:500]):
        seen = seen_per_user.get(user_idx, [])
        recommendations[user_idx] = model.recommend(
            user_idx=user_idx, top_k=20, seen_movie_idxs=seen
        )
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/500 users done")

    # Rating prediction metrics
    known_users  = set(train["user_idx"])
    known_movies = set(train["movie_idx"])
    test_filtered = test[
        test["user_idx"].isin(known_users) &
        test["movie_idx"].isin(known_movies)
    ].sample(5000, random_state=42)

    preds  = model.predict_batch(test_filtered)
    y_true = test_filtered["rating"].values

    print(f"\nRating Prediction (SVD) on test set")
    print(f"  RMSE : {rmse(y_true, preds):.4f}")
    print(f"  MAE  : {mae(y_true, preds):.4f}")

    print(f"\nRanking Metrics (SVD) @ top-500 users")
    ranking_df = evaluate_ranking(recommendations, ground_truth)
    print(ranking_df.to_string())