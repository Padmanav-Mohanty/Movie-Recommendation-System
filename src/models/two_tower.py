import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_DIMS,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    PROCESSED_DIR,
    SPLITS_DIR,
)

# ── Dataset ───────────────────────────────────────────────────────────────────


class RatingsDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame
    ):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        # Numeric user feature matrix (drop id col)
        user_feat_cols = [
            c
            for c in user_features.columns
            if c not in ("user_idx",)
            and user_features[c].dtype in (np.float64, np.int64, float, int)
        ]
        item_feat_cols = [
            c
            for c in item_features.columns
            if c not in ("movie_idx", "title")
            and item_features[c].dtype in (np.float64, np.int64, float, int)
        ]

        uf = user_features.set_index("user_idx")[user_feat_cols].fillna(0)
        itf = item_features.set_index("movie_idx")[item_feat_cols].fillna(0)

        self.user_feat_mat = torch.tensor(uf.values, dtype=torch.float32)
        self.item_feat_mat = torch.tensor(itf.values, dtype=torch.float32)
        self.user_feat_dim = self.user_feat_mat.shape[1]
        self.item_feat_dim = self.item_feat_mat.shape[1]

        # Index lookup
        self.user_feat_idx = {uid: i for i, uid in enumerate(uf.index)}
        self.item_feat_idx = {mid: i for i, mid in enumerate(itf.index)}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        u = self.users[idx].item()
        m = self.movies[idx].item()

        uf_idx = self.user_feat_idx.get(u, 0)
        if_idx = self.item_feat_idx.get(m, 0)

        return {
            "user_idx": self.users[idx],
            "movie_idx": self.movies[idx],
            "user_feat": self.user_feat_mat[uf_idx],
            "item_feat": self.item_feat_mat[if_idx],
            "rating": self.ratings[idx],
        }


# ── Model ─────────────────────────────────────────────────────────────────────


class TwoTowerModel(nn.Module):
    """
    Two-tower architecture:
      - User tower  : embedding + side features → dense vector
      - Item tower  : embedding + side features → dense vector
      - Score       : dot product of the two towers
    """

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        user_feat_dim: int,
        item_feat_dim: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dims: list = None,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        hidden_dims = hidden_dims or HIDDEN_DIMS

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # User tower
        user_input_dim = embedding_dim + user_feat_dim
        self.user_tower = self._build_tower(user_input_dim, hidden_dims, dropout)

        # Item tower
        item_input_dim = embedding_dim + item_feat_dim
        self.item_tower = self._build_tower(item_input_dim, hidden_dims, dropout)

        self._init_weights()

    def _build_tower(
        self, input_dim: int, hidden_dims: list, dropout: float
    ) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, EMBEDDING_DIM))
        return nn.Sequential(*layers)

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_idx, movie_idx, user_feat, item_feat):
        u_emb = self.user_embedding(user_idx)
        m_emb = self.movie_embedding(movie_idx)

        u_vec = self.user_tower(torch.cat([u_emb, user_feat], dim=1))
        m_vec = self.item_tower(torch.cat([m_emb, item_feat], dim=1))

        # Dot product → scale to rating range [0.5, 5.0]
        score = (u_vec * m_vec).sum(dim=1)
        score = torch.sigmoid(score) * 4.5 + 0.5
        return score

    def get_user_vector(
        self, user_idx: torch.Tensor, user_feat: torch.Tensor
    ) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        return self.user_tower(torch.cat([u_emb, user_feat], dim=1))

    def get_item_vector(
        self, movie_idx: torch.Tensor, item_feat: torch.Tensor
    ) -> torch.Tensor:
        m_emb = self.movie_embedding(movie_idx)
        return self.item_tower(torch.cat([m_emb, item_feat], dim=1))


# ── Trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(
        self, model: TwoTowerModel, lr: float = LEARNING_RATE, device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5
        )
        self.criterion = nn.MSELoss()
        print(f"Training on: {self.device}")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            self.optimizer.zero_grad()
            preds = self.model(
                batch["user_idx"].to(self.device),
                batch["movie_idx"].to(self.device),
                batch["user_feat"].to(self.device),
                batch["item_feat"].to(self.device),
            )
            loss = self.criterion(preds, batch["rating"].to(self.device))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        for batch in loader:
            preds = self.model(
                batch["user_idx"].to(self.device),
                batch["movie_idx"].to(self.device),
                batch["user_feat"].to(self.device),
                batch["item_feat"].to(self.device),
            )
            all_preds.append(preds.cpu())
            all_targets.append(batch["rating"])
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
        mae = float(np.mean(np.abs(targets - preds)))
        return rmse, mae

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = NUM_EPOCHS,
    ) -> list[dict]:
        history = []
        best_rmse = float("inf")
        best_state = None

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_rmse, val_mae = self.eval_epoch(val_loader)
            self.scheduler.step(val_rmse)
            elapsed = time.time() - t0

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                }
            )

            print(
                f"Epoch {epoch:>3}/{n_epochs} | "
                f"loss {train_loss:.4f} | "
                f"val RMSE {val_rmse:.4f} | "
                f"val MAE {val_mae:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

        # Restore best weights
        self.model.load_state_dict(best_state)
        print(f"\nBest val RMSE: {best_rmse:.4f}")
        return history


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ast

    import mlflow

    print("Loading data...")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val = pd.read_parquet(SPLITS_DIR / "val.parquet")

    # Re-parse genre_list
    for df in (train, val):
        if isinstance(df["genre_list"].iloc[0], str):
            df["genre_list"] = df["genre_list"].apply(ast.literal_eval)

    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")

    print("Building datasets...")
    train_ds = RatingsDataset(train, user_features, item_features)
    val_ds = RatingsDataset(val, user_features, item_features)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    n_users = train["user_idx"].max() + 1
    n_movies = train["movie_idx"].max() + 1

    print(
        f"n_users={n_users} | n_movies={n_movies} | "
        f"user_feat_dim={train_ds.user_feat_dim} | "
        f"item_feat_dim={train_ds.item_feat_dim}"
    )

    mlflow.set_experiment("two-tower")
    with mlflow.start_run(run_name="two-tower-v1"):
        mlflow.log_params(
            {
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dims": str(HIDDEN_DIMS),
                "dropout": DROPOUT,
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "n_epochs": NUM_EPOCHS,
            }
        )

        model = TwoTowerModel(
            n_users=n_users,
            n_movies=n_movies,
            user_feat_dim=train_ds.user_feat_dim,
            item_feat_dim=train_ds.item_feat_dim,
        )
        trainer = Trainer(model)
        history = trainer.fit(train_loader, val_loader, n_epochs=NUM_EPOCHS)

        best = min(history, key=lambda x: x["val_rmse"])
        mlflow.log_metrics({"val_rmse": best["val_rmse"], "val_mae": best["val_mae"]})

        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODELS_DIR / "two_tower.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "n_users": n_users,
                "n_movies": n_movies,
                "user_feat_dim": train_ds.user_feat_dim,
                "item_feat_dim": train_ds.item_feat_dim,
                "user_feat_idx": train_ds.user_feat_idx,
                "item_feat_idx": train_ds.item_feat_idx,
                "user_feat_mat": train_ds.user_feat_mat,
                "item_feat_mat": train_ds.item_feat_mat,
            },
            save_path,
        )
        print(f"Model saved → {save_path}")
        mlflow.log_artifact(str(save_path))
