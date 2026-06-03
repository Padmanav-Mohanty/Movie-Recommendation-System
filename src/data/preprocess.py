import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import (
    MIN_USER_RATINGS, MIN_MOVIE_RATINGS,
    TEST_SIZE, RANDOM_SEED, PROCESSED_DIR, SPLITS_DIR
)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, fix types, remove low-activity users/movies."""
    df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
    df["rating"]   = df["rating"].astype(float)
    df["user_id"]  = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)

    # Remove cold-start users and items
    user_counts  = df["user_id"].value_counts()
    movie_counts = df["movie_id"].value_counts()

    df = df[df["user_id"].isin(user_counts[user_counts  >= MIN_USER_RATINGS].index)]
    df = df[df["movie_id"].isin(movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index)]

    print(f"After cleaning: {df.shape[0]:,} rows | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['movie_id'].nunique():,} movies")
    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map user_id and movie_id to contiguous 0-based indices."""
    users  = sorted(df["user_id"].unique())
    movies = sorted(df["movie_id"].unique())

    user2idx  = {u: i for i, u in enumerate(users)}
    movie2idx = {m: i for i, m in enumerate(movies)}

    df = df.copy()
    df["user_idx"]  = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)

    print(f"Encoded: {len(user2idx):,} users | {len(movie2idx):,} movies")
    return df, user2idx, movie2idx


def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Expand pipe-separated genres into a list column."""
    df = df.copy()
    df["genre_list"] = df["genres"].fillna("Unknown").str.split("|")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal-style split:
      - per user, hold out the last 20% of ratings as test
      - from remaining, hold out 10% as validation
    """
    df = df.sort_values(["user_id", "rating"])  # proxy for time if no timestamp

    test_rows, val_rows, train_rows = [], [], []

    for _, group in df.groupby("user_id"):
        n = len(group)
        if n < 5:
            train_rows.append(group)
            continue
        n_test = max(1, int(n * TEST_SIZE))
        n_val  = max(1, int((n - n_test) * 0.1))

        test_rows.append(group.iloc[-n_test:])
        val_rows.append(group.iloc[-(n_test + n_val):-n_test])
        train_rows.append(group.iloc[:-(n_test + n_val)])

    train = pd.concat(train_rows).reset_index(drop=True)
    val   = pd.concat(val_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)

    print(f"Split → train: {len(train):,} | val: {len(val):,} | test: {len(test):,}")
    return train, val, test


def save_processed(df: pd.DataFrame, name: str, folder=PROCESSED_DIR) -> None:
    folder = folder if isinstance(folder, __builtins__.__class__) else __import__('pathlib').Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {name} → {path}")


def run_pipeline(train_raw: pd.DataFrame,
                 val_raw: pd.DataFrame) -> dict:
    """Full preprocessing pipeline."""
    # Combine for global encoding, then re-split
    combined = pd.concat([train_raw, val_raw], ignore_index=True)
    combined = basic_clean(combined)
    combined = parse_genres(combined)
    combined, user2idx, movie2idx = encode_ids(combined)

    train, val, test = split_data(combined)

    # Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR  / "val.parquet",   index=False)
    test.to_parquet(SPLITS_DIR / "test.parquet",  index=False)
    print("All splits saved.")

    return {
        "train": train, "val": val, "test": test,
        "user2idx": user2idx, "movie2idx": movie2idx,
        "n_users": len(user2idx), "n_movies": len(movie2idx),
    }


if __name__ == "__main__":
    from src.data.load_data import get_data
    train_raw, val_raw = get_data()
    data = run_pipeline(train_raw, val_raw)
    print("\nReady:", data["n_users"], "users,", data["n_movies"], "movies")