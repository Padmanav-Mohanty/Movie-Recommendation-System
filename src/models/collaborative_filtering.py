import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle

from config import PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, TOP_K


class UserBasedCF:
    """
    User-based Collaborative Filtering using cosine similarity.
    For each user, find K most similar users, aggregate their ratings
    as weighted predictions.
    """

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors     = n_neighbors
        self.interaction_mat = None   # (n_users, n_movies) sparse
        self.user_sim        = None   # (n_users, n_users) dense — computed lazily
        self.n_users         = 0
        self.n_movies        = 0

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, interaction_mat: csr_matrix) -> "UserBasedCF":
        self.interaction_mat = interaction_mat
        self.n_users, self.n_movies = interaction_mat.shape
        print(f"UserBasedCF fit on {self.n_users:,} users × {self.n_movies:,} movies")

        print("Computing user-user cosine similarity...")
        # Chunked to avoid OOM on large matrices
        self.user_sim = cosine_similarity(interaction_mat, dense_output=True)
        np.fill_diagonal(self.user_sim, 0)   # user is not their own neighbour
        print("Similarity matrix ready.")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating for a single (user, movie) pair."""
        sim_scores = self.user_sim[user_idx]                        # (n_users,)
        top_n_idx  = np.argsort(sim_scores)[::-1][:self.n_neighbors]

        neighbour_ratings = self.interaction_mat[top_n_idx, movie_idx].toarray().flatten()
        neighbour_sims    = sim_scores[top_n_idx]

        mask = neighbour_ratings > 0
        if mask.sum() == 0:
            # Cold fallback — global mean
            return self.interaction_mat.data.mean()

        return np.dot(neighbour_sims[mask], neighbour_ratings[mask]) / (
            neighbour_sims[mask].sum() + 1e-9
        )

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for a DataFrame with user_idx and movie_idx columns."""
        return np.array([
            self.predict(row.user_idx, row.movie_idx)
            for row in df.itertuples(index=False)
        ])

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        """Return top-K movie indices for a user."""
        sim_scores = self.user_sim[user_idx]
        top_n_idx  = np.argsort(sim_scores)[::-1][:self.n_neighbors]

        # Weighted sum of neighbour rating vectors
        neighbour_mat    = self.interaction_mat[top_n_idx].toarray()   # (n_neighbours, n_movies)
        weights          = sim_scores[top_n_idx].reshape(-1, 1)
        weighted_scores  = (weights * neighbour_mat).sum(axis=0)       # (n_movies,)

        if exclude_seen:
            seen = self.interaction_mat[user_idx].indices
            weighted_scores[seen] = -np.inf

        top_k_idx = np.argsort(weighted_scores)[::-1][:top_k]
        return top_k_idx.tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "UserBasedCF":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # Load data
    print("Loading splits and interaction matrix...")
    train    = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val      = pd.read_parquet(SPLITS_DIR / "val.parquet")
    int_mat  = load_npz(str(PROCESSED_DIR / "interaction_matrix.npz"))

    # Fit
    model = UserBasedCF(n_neighbors=50)
    t0 = time.time()
    model.fit(int_mat)
    print(f"Fit time: {time.time() - t0:.1f}s")

    # Evaluate on a sample (full val is slow for CF)
    print("\nEvaluating on 2,000 validation samples...")
    val_sample = val.sample(2000, random_state=42)
    t0 = time.time()
    preds = model.predict_batch(val_sample)
    print(f"Predict time: {time.time() - t0:.1f}s")

    y_true = val_sample["rating"].values
    print(f"\nUser-Based CF Results")
    print(f"  RMSE : {rmse(y_true, preds):.4f}")
    print(f"  MAE  : {mae(y_true, preds):.4f}")

    # Sample recommendations
    print("\nSample recommendations for user_idx=0:")
    recs = model.recommend(user_idx=0, top_k=10)
    print(f"  Top-10 movie indices: {recs}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "user_based_cf.pkl")