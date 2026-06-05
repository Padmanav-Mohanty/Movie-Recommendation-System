import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from config import PROCESSED_DIR, SPLITS_DIR

# ── Genre list (all unique genres in MovieLens) ───────────────────────────────
ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "Unknown",
]


# ── User features ─────────────────────────────────────────────────────────────


def build_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user features derived from training ratings only (no leakage).

    Features:
        - mean_rating       : user's average rating
        - rating_std        : how opinionated they are
        - n_ratings         : activity level (log-scaled)
        - genre_affinity_*  : mean rating per genre (20 features)
    """
    # Basic stats
    user_stats = (
        train.groupby("user_idx")["rating"]
        .agg(mean_rating="mean", rating_std="std", n_ratings="count")
        .reset_index()
    )
    user_stats["rating_std"] = user_stats["rating_std"].fillna(0)
    user_stats["log_n_ratings"] = np.log1p(user_stats["n_ratings"])

    # Genre affinity — explode genre_list then pivot
    train_genres = train.explode("genre_list")
    genre_affinity = (
        train_genres.groupby(["user_idx", "genre_list"])["rating"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=ALL_GENRES, fill_value=0)
    )
    genre_affinity.columns = [
        f"genre_affinity_{g.lower().replace('-', '_')}" for g in genre_affinity.columns
    ]
    genre_affinity = genre_affinity.reset_index()

    user_features = user_stats.merge(genre_affinity, on="user_idx", how="left")

    print(f"User features  : {user_features.shape[0]:,} users × {user_features.shape[1]} features")
    return user_features


# ── Item features ─────────────────────────────────────────────────────────────


def build_item_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Per-item features derived from training data only.

    Features:
        - mean_rating       : item's average rating
        - rating_std        : rating variance
        - n_ratings         : popularity (log-scaled)
        - year              : release year extracted from title
        - genre_*           : one-hot genre flags (20 features)
    """
    # Basic stats
    item_stats = (
        train.groupby("movie_idx")["rating"]
        .agg(mean_rating="mean", rating_std="std", n_ratings="count")
        .reset_index()
    )
    item_stats["rating_std"] = item_stats["rating_std"].fillna(0)
    item_stats["log_n_ratings"] = np.log1p(item_stats["n_ratings"])

    # Release year from title e.g. "Toy Story (1995)"
    movie_meta = train[["movie_idx", "title", "genre_list"]].drop_duplicates("movie_idx").copy()
    movie_meta["year"] = movie_meta["title"].str.extract(r"\((\d{4})\)$").astype(float).fillna(0)

    # One-hot genres
    mlb = MultiLabelBinarizer(classes=ALL_GENRES)
    genre_ohe = pd.DataFrame(
        mlb.fit_transform(movie_meta["genre_list"]),
        columns=[f"genre_{g.lower().replace('-', '_')}" for g in ALL_GENRES],
        index=movie_meta.index,
    )
    movie_meta = pd.concat([movie_meta[["movie_idx", "year"]], genre_ohe], axis=1)

    item_features = item_stats.merge(movie_meta, on="movie_idx", how="left")

    print(f"Item features  : {item_features.shape[0]:,} movies × {item_features.shape[1]} features")
    return item_features


# ── Interaction matrix ────────────────────────────────────────────────────────


def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_movies: int) -> np.ndarray:
    """Sparse-friendly dense interaction matrix (users × movies)."""
    from scipy.sparse import csr_matrix

    mat = csr_matrix(
        (df["rating"].values, (df["user_idx"].values, df["movie_idx"].values)),
        shape=(n_users, n_movies),
    )
    print(f"Interaction matrix : {mat.shape} | sparsity {1 - mat.nnz / (n_users * n_movies):.4%}")
    return mat


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_feature_pipeline(train: pd.DataFrame, n_users: int, n_movies: int) -> dict:
    user_features = build_user_features(train)
    item_features = build_item_features(train)
    interaction_mat = build_interaction_matrix(train, n_users, n_movies)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    user_features.to_parquet(PROCESSED_DIR / "user_features.parquet", index=False)
    item_features.to_parquet(PROCESSED_DIR / "item_features.parquet", index=False)

    from scipy.sparse import save_npz

    save_npz(str(PROCESSED_DIR / "interaction_matrix.npz"), interaction_mat)

    print(f"Saved features → {PROCESSED_DIR}")

    return {
        "user_features": user_features,
        "item_features": item_features,
        "interaction_mat": interaction_mat,
    }


if __name__ == "__main__":
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")

    # genre_list is stored as string in parquet — re-parse it
    if isinstance(train["genre_list"].iloc[0], str):
        import ast

        train["genre_list"] = train["genre_list"].apply(ast.literal_eval)

    n_users = train["user_idx"].max() + 1
    n_movies = train["movie_idx"].max() + 1

    features = run_feature_pipeline(train, n_users, n_movies)
