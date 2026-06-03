import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time

from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import cross_validate

from config import (
    SPLITS_DIR, MODELS_DIR,
    SVD_N_FACTORS, SVD_N_EPOCHS, SVD_LR, SVD_REG,
)


def df_to_surprise(df: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(df[["user_idx", "movie_idx", "rating"]], reader)


class MatrixFactorization:
    def __init__(self, algorithm="svd", n_factors=SVD_N_FACTORS,
                 n_epochs=SVD_N_EPOCHS, lr_all=SVD_LR, reg_all=SVD_REG):
        self.algorithm = algorithm
        self.n_movies  = 0
        self.params    = dict(n_factors=n_factors, n_epochs=n_epochs,
                              lr_all=lr_all, reg_all=reg_all)
        if algorithm == "svd":
            self.model = SVD(**self.params, random_state=42, verbose=False)
        elif algorithm == "nmf":
            self.model = NMF(n_factors=n_factors, n_epochs=n_epochs,
                             random_state=42, verbose=False)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, train_df: pd.DataFrame) -> "MatrixFactorization":
        self.n_movies = train_df["movie_idx"].max() + 1
        dataset       = df_to_surprise(train_df)
        trainset      = dataset.build_full_trainset()
        print(f"Fitting {self.algorithm.upper()} "
              f"(factors={self.params['n_factors']}, epochs={self.params['n_epochs']})...")
        t0 = time.time()
        self.model.fit(trainset)
        print(f"Fit time: {time.time() - t0:.1f}s")
        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        return self.model.predict(user_idx, movie_idx).est

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([
            self.predict(row.user_idx, row.movie_idx)
            for row in df.itertuples(index=False)
        ])

    def recommend(self, user_idx: int, top_k: int = 10,
                  seen_movie_idxs: list = None) -> list:
        seen       = set(seen_movie_idxs or [])
        candidates = [m for m in range(self.n_movies) if m not in seen]
        scores     = [(m, self.predict(user_idx, m)) for m in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scores[:top_k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "MatrixFactorization":
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate(model, val_df: pd.DataFrame, train_df: pd.DataFrame, sample_n: int = 5000) -> dict:
    known_users  = set(train_df["user_idx"].unique())
    known_movies = set(train_df["movie_idx"].unique())

    filtered = val_df[
        val_df["user_idx"].isin(known_users) &
        val_df["movie_idx"].isin(known_movies)
    ]
    sample = filtered.sample(min(sample_n, len(filtered)), random_state=42)
    preds  = model.predict_batch(sample)
    y_true = sample["rating"].values
    return {
        "rmse": float(np.sqrt(np.mean((y_true - preds) ** 2))),
        "mae":  float(np.mean(np.abs(y_true - preds))),
    }


def compare_algorithms(train_df: pd.DataFrame) -> pd.DataFrame:
    reader  = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(
        train_df[["user_idx", "movie_idx", "rating"]], reader
    )
    results = []
    for name, algo in [
        ("SVD", SVD(n_factors=SVD_N_FACTORS, n_epochs=SVD_N_EPOCHS,
                    lr_all=SVD_LR, reg_all=SVD_REG, random_state=42, verbose=False)),
        ("NMF", NMF(n_factors=SVD_N_FACTORS, n_epochs=SVD_N_EPOCHS,
                    random_state=42, verbose=False)),
    ]:
        print(f"Cross-validating {name}...")
        cv = cross_validate(algo, dataset, measures=["RMSE", "MAE"],
                            cv=3, verbose=False, n_jobs=-1)
        results.append({
            "algorithm": name,
            "rmse_mean": cv["test_rmse"].mean(),
            "rmse_std":  cv["test_rmse"].std(),
            "mae_mean":  cv["test_mae"].mean(),
            "mae_std":   cv["test_mae"].std(),
            "fit_time":  np.mean(cv["fit_time"]),
        })
    return pd.DataFrame(results).sort_values("rmse_mean")


if __name__ == "__main__":
    import mlflow

    print("Loading data...")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val   = pd.read_parquet(SPLITS_DIR / "val.parquet")

    mlflow.set_experiment("matrix-factorization")
    with mlflow.start_run(run_name="SVD-fixed"):
        mlflow.log_params(dict(
            n_factors=SVD_N_FACTORS, n_epochs=SVD_N_EPOCHS,
            lr_all=SVD_LR, reg_all=SVD_REG
        ))
        model = MatrixFactorization("svd")
        model.fit(train)

        metrics = evaluate(model, val, train)
        mlflow.log_metrics(metrics)

        print(f"\nSVD Results")
        print(f"  RMSE : {metrics['rmse']:.4f}  (CV score was 0.7638)")
        print(f"  MAE  : {metrics['mae']:.4f}")

        seen = train[train["user_idx"] == 0]["movie_idx"].tolist()
        recs = model.recommend(user_idx=0, top_k=10, seen_movie_idxs=seen)
        print(f"\nTop-10 for user_idx=0: {recs}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / "svd_model.pkl")