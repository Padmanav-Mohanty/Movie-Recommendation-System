"""
Unit tests for src/data/preprocess.py
"""

from __future__ import annotations

import pandas as pd

from src.data.preprocess import basic_clean, encode_ids, parse_genres, split_data


class TestBasicClean:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": [1, 2, None, 3, 3, 3, 3, 3, 3],
                "movie_id": [10, 20, 30, 40, 40, 40, 40, 40, 50],
                "rating": [4.0, 3.5, 5.0, 2.0, 3.0, 4.0, 2.5, 3.5, None],
            }
        )

    def test_drops_nulls(self):
        df = self._make_df()
        result = basic_clean(df)
        assert result.isna().sum().sum() == 0

    def test_correct_dtypes(self):
        df = self._make_df()
        result = basic_clean(df)
        assert result["rating"].dtype == float
        assert result["user_id"].dtype == int
        assert result["movie_id"].dtype == int


class TestEncodeIds:
    def test_contiguous_zero_based(self):
        df = pd.DataFrame(
            {
                "user_id": [10, 20, 30],
                "movie_id": [100, 200, 300],
                "rating": [4.0, 3.0, 5.0],
            }
        )
        result, user2idx, movie2idx = encode_ids(df)
        assert set(result["user_idx"]) == {0, 1, 2}
        assert set(result["movie_idx"]) == {0, 1, 2}

    def test_mapping_invertible(self):
        df = pd.DataFrame(
            {
                "user_id": [5, 5, 10],
                "movie_id": [1, 2, 1],
                "rating": [3.0, 4.0, 5.0],
            }
        )
        _, user2idx, _ = encode_ids(df)
        idx2user = {v: k for k, v in user2idx.items()}
        assert len(idx2user) == len(user2idx)

    def test_new_columns_added(self):
        df = pd.DataFrame({"user_id": [1], "movie_id": [1], "rating": [3.0]})
        result, _, _ = encode_ids(df)
        assert "user_idx" in result.columns
        assert "movie_idx" in result.columns


class TestParseGenres:
    def test_splits_pipe_separated(self):
        df = pd.DataFrame({"genres": ["Action|Comedy", "Drama"]})
        result = parse_genres(df)
        assert result["genre_list"].iloc[0] == ["Action", "Comedy"]
        assert result["genre_list"].iloc[1] == ["Drama"]

    def test_handles_nulls(self):
        df = pd.DataFrame({"genres": [None, "Action"]})
        result = parse_genres(df)
        assert result["genre_list"].iloc[0] == ["Unknown"]

    def test_original_column_preserved(self):
        df = pd.DataFrame({"genres": ["Comedy"]})
        result = parse_genres(df)
        assert "genres" in result.columns


class TestSplitData:
    def _make_user_df(self, n_ratings=20) -> pd.DataFrame:
        """Single user with n_ratings ratings."""
        return pd.DataFrame(
            {
                "user_id": [1] * n_ratings,
                "movie_id": list(range(n_ratings)),
                "user_idx": [0] * n_ratings,
                "movie_idx": list(range(n_ratings)),
                "rating": [4.0] * n_ratings,
            }
        )

    def test_total_rows_preserved(self):
        df = self._make_user_df(20)
        train, val, test = split_data(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_duplicate_movie_idxs_across_splits(self):
        """Each movie should appear in at most one split per user."""
        df = self._make_user_df(20)
        train, val, test = split_data(df)
        train_movies = set(train["movie_idx"])
        test_movies = set(test["movie_idx"])
        assert train_movies.isdisjoint(test_movies), "Train and test overlap"

    def test_small_user_goes_to_train(self):
        """Users with < 5 ratings should be assigned entirely to train."""
        df = pd.DataFrame(
            {
                "user_id": [2, 2, 2],
                "movie_id": [1, 2, 3],
                "user_idx": [1, 1, 1],
                "movie_idx": [0, 1, 2],
                "rating": [3.0, 4.0, 5.0],
            }
        )
        train, val, test = split_data(df)
        assert len(train) == 3
        # val and test may be empty lists — concat produces empty DataFrames
        assert len(val) == 0
        assert len(test) == 0

    def test_test_size_approximately_correct(self):
        df = self._make_user_df(100)
        _, _, test = split_data(df)
        # TEST_SIZE=0.2 → ~20 rows in test
        assert 15 <= len(test) <= 25

    def test_split_returns_dataframes(self):
        df = self._make_user_df(10)
        splits = split_data(df)
        assert len(splits) == 3
        for s in splits:
            assert isinstance(s, pd.DataFrame)
