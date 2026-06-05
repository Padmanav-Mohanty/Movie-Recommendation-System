from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models" / "saved"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"

# ── Dataset ──────────────────────────────────────────────────────────────────
HF_DATASET_PATH = "ashraq/movielens_ratings"
HF_SPLITS = {
    "train": "data/train-00000-of-00001-8c8c7645a52d95e5.parquet",
    "validation": "data/validation-00000-of-00001-609ec132d91847f9.parquet",
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
MIN_USER_RATINGS = 5  # drop users with fewer ratings
MIN_MOVIE_RATINGS = 5  # drop movies with fewer ratings
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ── Model hyperparameters ─────────────────────────────────────────────────────
# Matrix Factorization (SVD)
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
SVD_LR = 0.005
SVD_REG = 0.02

# Two-Tower Neural Model
EMBEDDING_DIM = 64
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
NUM_EPOCHS = 20

# ── Evaluation ────────────────────────────────────────────────────────────────
TOP_K = [5, 10, 20]

# ── Serving ───────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
N_CANDIDATES = 100  # retrieve top-100, rerank to top-K
