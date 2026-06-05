"""Data loading and preprocessing."""


def get_data(*args, **kwargs):
    from .load_data import get_data as _get_data

    return _get_data(*args, **kwargs)


def load_raw(*args, **kwargs):
    from .load_data import load_raw as _load_raw

    return _load_raw(*args, **kwargs)


from .preprocess import basic_clean, encode_ids, parse_genres, run_pipeline, split_data  # noqa: E402

__all__ = [
    "get_data",
    "load_raw",
    "basic_clean",
    "encode_ids",
    "parse_genres",
    "split_data",
    "run_pipeline",
]
