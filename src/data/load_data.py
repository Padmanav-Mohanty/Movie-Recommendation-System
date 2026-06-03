import pandas as pd
from datasets import load_dataset
from config import HF_DATASET_PATH, RAW_DIR


def load_from_huggingface() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits from HuggingFace."""
    print("Loading dataset from HuggingFace...")

    dataset = load_dataset(HF_DATASET_PATH)

    train_df = dataset["train"].to_pandas()
    val_df   = dataset["validation"].to_pandas()

    print(f"  Train:      {len(train_df):,} rows")
    print(f"  Validation: {len(val_df):,} rows")

    return train_df, val_df


def save_raw(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Save raw splits to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(RAW_DIR / "train.parquet", index=False)
    val_df.to_parquet(RAW_DIR   / "validation.parquet", index=False)
    print(f"Saved raw data to {RAW_DIR}")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from local disk (faster than re-downloading)."""
    train_path = RAW_DIR / "train.parquet"
    val_path   = RAW_DIR / "validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError("Raw data not found. Run with force_download=True first.")

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