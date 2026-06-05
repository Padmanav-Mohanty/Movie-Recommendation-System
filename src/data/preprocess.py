import pandas as pd

from config import (
    MIN_MOVIE_RATINGS,
    MIN_USER_RATINGS,
    PROCESSED_DIR,
    SPLITS_DIR,
    TEST_SIZE,
)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, fix types, remove cold-start users/movies."""
    df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
    df["rating"] = df["rating"].astype(float)
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)

    user_counts = df["user_id"].value_counts()
    movie_counts = df["movie_id"].value_counts()

    df = df[df["user_id"].isin(user_counts[user_counts >= MIN_USER_RATINGS].index)]
    df = df[df["movie_id"].isin(movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index)]

    print(
        f"After cleaning : {df.shape[0]:,} rows | "
        f"{df['user_id'].nunique():,} users | "
        f"{df['movie_id'].nunique():,} movies"
    )
    return df


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map user_id / movie_id to contiguous 0-based indices."""
    users = sorted(df["user_id"].unique())
    movies = sorted(df["movie_id"].unique())

    user2idx = {u: i for i, u in enumerate(users)}
    movie2idx = {m: i for i, m in enumerate(movies)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)

    print(f"Encoded        : {len(user2idx):,} users | {len(movie2idx):,} movies")
    return df, user2idx, movie2idx


def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Expand pipe-separated genres into a list column."""
    df = df.copy()
    df["genre_list"] = df["genres"].fillna("Unknown").str.split("|")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal split:
      - last 20% of each user's ratings → test
      - next 10%                        → validation
      - remainder                       → train
    """
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    test_rows, val_rows, train_rows = [], [], []

    for _, group in df.groupby("user_id"):
        n = len(group)
        if n < 5:
            train_rows.append(group)
            continue
        n_test = max(1, int(n * TEST_SIZE))
        n_val = max(1, int((n - n_test) * 0.1))

        test_rows.append(group.iloc[-n_test:])
        val_rows.append(group.iloc[-(n_test + n_val) : -n_test])
        train_rows.append(group.iloc[: -(n_test + n_val)])

    empty = pd.DataFrame(columns=df.columns)
    train = pd.concat(train_rows).reset_index(drop=True) if train_rows else empty
    val = pd.concat(val_rows).reset_index(drop=True) if val_rows else empty
    test = pd.concat(test_rows).reset_index(drop=True) if test_rows else empty

    print(
        f"Split          : train {len(train):,} | val {len(val):,} | test {len(test):,}"
    )
    return train, val, test


def run_pipeline(train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> dict:
    """Full preprocessing pipeline — returns splits + mappings."""
    combined = pd.concat([train_raw, val_raw], ignore_index=True)
    combined = basic_clean(combined)
    combined = parse_genres(combined)
    combined, user2idx, movie2idx = encode_ids(combined)

    train, val, test = split_data(combined)

    # Persist splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test.to_parquet(SPLITS_DIR / "test.parquet", index=False)
    print(f"Saved splits   → {SPLITS_DIR}")

    # Persist mappings
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(user2idx).to_frame("idx").to_parquet(PROCESSED_DIR / "user2idx.parquet")
    pd.Series(movie2idx).to_frame("idx").to_parquet(PROCESSED_DIR / "movie2idx.parquet")
    print(f"Saved mappings → {PROCESSED_DIR}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "n_users": len(user2idx),
        "n_movies": len(movie2idx),
    }


if __name__ == "__main__":
    from src.data.load_data import get_data

    train_raw, val_raw = get_data()
    print("\nRunning preprocessing pipeline...")
    data = run_pipeline(train_raw, val_raw)

    print(f"\n✓ Done — {data['n_users']:,} users | {data['n_movies']:,} movies")
    print("\nTrain sample:")
    print(
        data["train"][["user_id", "movie_id", "rating", "user_idx", "movie_idx"]].head()
    )
