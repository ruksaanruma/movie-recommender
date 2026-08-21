"""
Data preparation for the movie recommender.

Two jobs:
  1. Split ratings PER USER — hold out ~20% of each user's ratings for testing,
     so every user keeps training history (needed by collaborative filtering)
     and has held-out ratings to evaluate against.
  2. Build the sparse user-item rating matrix (SciPy CSR) from the training
     ratings, with id<->index maps (MovieLens ids are not contiguous).

Run:
    python -m src.data_prep
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SEED = 42
TEST_FRAC = 0.20


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    return ratings, movies


def split_ratings(ratings: pd.DataFrame, test_frac: float = TEST_FRAC,
                  seed: int = SEED, save: bool = True):
    """Hold out `test_frac` of each user's ratings for the test set."""
    rng = np.random.RandomState(seed)
    test_idx = []
    for _, grp in ratings.groupby("userId"):
        n_test = max(1, int(round(len(grp) * test_frac)))
        test_idx.extend(rng.choice(grp.index.values, size=n_test, replace=False))
    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train.to_csv(PROCESSED_DIR / "train.csv", index=False)
        test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_user_item_matrix(train: pd.DataFrame):
    """Return (CSR matrix [users x movies], user_index, movie_index)."""
    users = np.sort(train["userId"].unique())
    movies = np.sort(train["movieId"].unique())
    uidx = {u: i for i, u in enumerate(users)}
    midx = {m: j for j, m in enumerate(movies)}
    rows = train["userId"].map(uidx).values
    cols = train["movieId"].map(midx).values
    mat = csr_matrix((train["rating"].values, (rows, cols)),
                     shape=(len(users), len(movies)))
    return mat, uidx, midx


def load_splits():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, test


if __name__ == "__main__":
    ratings, movies = load_data()
    train, test = split_ratings(ratings)
    mat, uidx, midx = build_user_item_matrix(train)

    print(f"Ratings: {len(ratings):,} -> train {len(train):,} / test {len(test):,}")
    print(f"User-item matrix: {mat.shape[0]} users x {mat.shape[1]} movies")
    print(f"  non-zero entries: {mat.nnz:,} "
          f"(density {mat.nnz / (mat.shape[0] * mat.shape[1]) * 100:.2f}%)")

    # How many test pairs are scorable (user & movie both seen in train)?
    seen_movie = test["movieId"].isin(midx)
    print(f"\nTest ratings with a train-seen movie: {seen_movie.mean() * 100:.1f}% "
          f"({(~seen_movie).sum()} cold-item pairs will be skipped in eval)")
    print(f"Every test user is in train: {test['userId'].isin(uidx).all()}")
    print(f"\nSaved splits to {PROCESSED_DIR.relative_to(PROJECT_ROOT)}/")
