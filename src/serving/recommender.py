"""
Unified recommender that can load any trained model and serve top-K recommendations.
Also builds a FAISS ANN index from Two-Tower item vectors for fast retrieval.
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from config import (
    MODELS_DIR, PROCESSED_DIR, SPLITS_DIR,
    FAISS_INDEX_PATH, N_CANDIDATES, TOP_K,
    EMBEDDING_DIM, HIDDEN_DIMS, DROPOUT,
)


# ── Base interface ────────────────────────────────────────────────────────────

class BaseRecommender:
    """Every recommender must implement recommend()."""

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> List[int]:
        raise NotImplementedError

    def recommend_batch(
        self,
        user_idxs:    List[int],
        top_k:        int = 10,
        exclude_seen: bool = True,
        seen_dict:    Optional[Dict[int, List[int]]] = None,
    ) -> Dict[int, List[int]]:
        seen_dict = seen_dict or {}
        return {
            u: self.recommend(u, top_k=top_k, exclude_seen=exclude_seen)
            for u in user_idxs
        }


# ── CF recommender ────────────────────────────────────────────────────────────

class CFRecommender(BaseRecommender):
    """Wraps the UserBasedCF model."""

    def __init__(self, model_path: Path = None):
        path = model_path or (MODELS_DIR / "user_based_cf.pkl")
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        print(f"CFRecommender loaded from {path}")

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> List[int]:
        return self._model.recommend(user_idx, top_k=top_k,
                                     exclude_seen=exclude_seen)


# ── SVD recommender ───────────────────────────────────────────────────────────

class SVDRecommender(BaseRecommender):
    """Wraps the MatrixFactorization (SVD/NMF) model."""

    def __init__(self, model_path: Path = None,
                 train_df: pd.DataFrame = None):
        path = model_path or (MODELS_DIR / "svd_model.pkl")
        with open(path, "rb") as f:
            self._model = pickle.load(f)

        # Build seen-items lookup from train
        if train_df is not None:
            self._seen = (
                train_df.groupby("user_idx")["movie_idx"]
                .apply(list).to_dict()
            )
        else:
            self._seen = {}
        print(f"SVDRecommender loaded from {path}")

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> List[int]:
        seen = self._seen.get(user_idx, []) if exclude_seen else []
        return self._model.recommend(user_idx, top_k=top_k,
                                     seen_movie_idxs=seen)


# ── Two-Tower + FAISS recommender ─────────────────────────────────────────────

class TwoTowerRecommender(BaseRecommender):
    """
    Loads the Two-Tower model and a FAISS index.
    If the FAISS index doesn't exist yet, it builds it from item vectors.
    """

    def __init__(self, model_path: Path = None,
                 train_df: pd.DataFrame = None,
                 device: str = None):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. pip install faiss-cpu")

        from src.models.two_tower import TwoTowerModel

        path  = model_path or (MODELS_DIR / "two_tower.pt")
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TwoTowerModel(
            n_users=ckpt["n_users"],
            n_movies=ckpt["n_movies"],
            user_feat_dim=ckpt["user_feat_dim"],
            item_feat_dim=ckpt["item_feat_dim"],
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(self.device).eval()

        self._user_feat_mat = ckpt["user_feat_mat"]
        self._item_feat_mat = ckpt["item_feat_mat"]
        self._user_feat_idx = ckpt["user_feat_idx"]
        self._item_feat_idx = ckpt["item_feat_idx"]
        self._n_movies      = ckpt["n_movies"]

        # Seen items lookup
        if train_df is not None:
            self._seen = (
                train_df.groupby("user_idx")["movie_idx"]
                .apply(list).to_dict()
            )
        else:
            self._seen = {}

        # Build/load FAISS index
        self._index = self._load_or_build_index()
        print(f"TwoTowerRecommender loaded from {path}")

    # ── FAISS helpers ─────────────────────────────────────────────────────────

    def _build_item_vectors(self) -> np.ndarray:
        """Compute dense item embeddings for all movies."""
        all_idx  = torch.arange(self._n_movies)
        all_feat = torch.zeros(self._n_movies,
                               self._item_feat_mat.shape[1])
        for mid, i in self._item_feat_idx.items():
            if mid < self._n_movies:
                all_feat[mid] = self._item_feat_mat[i]

        with torch.no_grad():
            vecs = self._model.get_item_vector(
                all_idx.to(self.device),
                all_feat.to(self.device),
            ).cpu().numpy()
        return vecs.astype("float32")

    def _load_or_build_index(self):
        if FAISS_INDEX_PATH.exists():
            index = self._faiss.read_index(str(FAISS_INDEX_PATH))
            print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
        else:
            print("Building FAISS index...")
            vecs  = self._build_item_vectors()
            self._faiss.normalize_L2(vecs)
            index = self._faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(index, str(FAISS_INDEX_PATH))
            print(f"FAISS index saved → {FAISS_INDEX_PATH} ({index.ntotal} vectors)")
        return index

    def _get_user_vector(self, user_idx: int) -> np.ndarray:
        uf_i    = self._user_feat_idx.get(user_idx, 0)
        u_feat  = self._user_feat_mat[uf_i].unsqueeze(0)
        u_idx_t = torch.tensor([user_idx])
        with torch.no_grad():
            vec = self._model.get_user_vector(
                u_idx_t.to(self.device),
                u_feat.to(self.device),
            ).cpu().numpy().astype("float32")
        self._faiss.normalize_L2(vec)
        return vec

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> List[int]:
        u_vec      = self._get_user_vector(user_idx)
        _, indices = self._index.search(u_vec, N_CANDIDATES)
        candidates = indices[0].tolist()

        if exclude_seen:
            seen = set(self._seen.get(user_idx, []))
            candidates = [c for c in candidates if c not in seen]

        return candidates[:top_k]


# ── Factory ───────────────────────────────────────────────────────────────────

def load_recommender(
    model_name: str,
    train_df: pd.DataFrame = None,
) -> BaseRecommender:
    """
    Factory function.

    Parameters
    ----------
    model_name : 'cf' | 'svd' | 'two_tower'
    train_df   : training split DataFrame (needed for seen-items filtering)
    """
    model_name = model_name.lower().replace("-", "_")
    if model_name == "cf":
        return CFRecommender()
    elif model_name == "svd":
        return SVDRecommender(train_df=train_df)
    elif model_name in ("two_tower", "twotower"):
        return TwoTowerRecommender(train_df=train_df)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. "
                         f"Choose from 'cf', 'svd', 'two_tower'.")


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "svd"
    print(f"Loading model: {model_name}")

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    rec   = load_recommender(model_name, train_df=train)

    user_idx = 0
    recs = rec.recommend(user_idx, top_k=10)
    print(f"\nTop-10 recommendations for user_idx={user_idx}:")
    print(recs)