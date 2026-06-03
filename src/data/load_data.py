import pandas as pd
from pathlib import Path
from config import HF_DATASET_PATH, HF_SPLITS, RAW_DIR

def load_from_huggingface() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits from HuggingFace."""
    print("Loading dataset from HuggingFace...")

    train_df = pd.read_parquet(f"hf://datasets/{HF_DATASET_PATH}/{HF_SPLITS['train']}")
    val_df   = pd.read_parquet(f"hf://datasets/{HF_DATASET_PATH}/{HF_SPLITS['validation']}")

    print(f"  Train:      {train_df.shape[0]:,} rows")
    print(f"  Validation: {val_df.shape[0]:,} rows")

    return train_df, val_df


def save_raw(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Save raw splits to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(RAW_DIR / "train.parquet", index=False)
    val_df.to_parquet(RAW_DIR / "validation.parquet", index=False)
    print(f"Saved raw data to {RAW_DIR}")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from local disk (faster than re-downloading)."""
    train_path = RAW_DIR / "train.parquet"
    val_path   = RAW_DIR / "validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError("Raw data not found. Run load_from_huggingface() first.")

    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def get_data(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point — downloads once, then loads from disk."""
    if force_download or not (RAW_DIR / "train.parquet").exists():
        train_df, val_df = load_from_huggingface()
        save_raw(train_df, val_df)
    else:
        print("Loading from local cache...")
        train_df, val_df = load_raw()

    return train_df, val_df


if __name__ == "__main__":
    train, val = get_data()
    print("\nTrain sample:")
    print(train.head())
    print("\nColumns:", train.columns.tolist())
    print("Dtypes:\n", train.dtypes)