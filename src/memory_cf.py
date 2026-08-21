"""
Baselines + memory-based collaborative filtering (item-item).

Baselines (the bar to beat):
  - global mean
  - per-user mean
  - per-movie mean

Item-item CF:
  - adjusted-cosine similarity between movies (ratings centered by user mean so
    a generous vs harsh rater doesn't skew the similarity).
  - predict r(u, i) = user_mean[u] + weighted avg over the user's rated movies of
    similarity(i, j) * (r(u, j) - user_mean[u]), using the top-K similar movies.
  - similarity is computed over movies with >= MIN_ITEM_RATINGS ratings (the
    long tail is too sparse to be reliable); cold items fall back to the movie
    mean.

Run:
    python -m src.memory_cf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.data_prep import load_splits, load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MIN_ITEM_RATINGS = 5
TOP_K = 40


class ItemItemCF:
    def __init__(self, min_item_ratings: int = MIN_ITEM_RATINGS, k: int = TOP_K):
        self.min_item_ratings = min_item_ratings
        self.k = k

    def fit(self, train: pd.DataFrame):
        self.global_mean = train["rating"].mean()
        self.user_mean = train.groupby("userId")["rating"].mean().to_dict()
        self.item_mean = train.groupby("movieId")["rating"].mean().to_dict()

        # Popular items only (long tail is unreliable for similarity)
        counts = train.groupby("movieId").size()
        popular = np.sort(counts[counts >= self.min_item_ratings].index.values)
        self.pop_index = {m: j for j, m in enumerate(popular)}
        self.popular = popular

        # Centered (by user mean) users x popular-items matrix
        pop_ratings = train[train["movieId"].isin(self.pop_index)].copy()
        users = np.sort(train["userId"].unique())
        self.uidx = {u: i for i, u in enumerate(users)}
        rows = pop_ratings["userId"].map(self.uidx).values
        cols = pop_ratings["movieId"].map(self.pop_index).values
        centered = pop_ratings["rating"].values - \
            pop_ratings["userId"].map(self.user_mean).values
        Xc = csr_matrix((centered, (rows, cols)),
                        shape=(len(users), len(popular)))

        # Item-item cosine similarity (float32 to save memory)
        self.sim = cosine_similarity(Xc.T.astype(np.float32), dense_output=True)
        np.fill_diagonal(self.sim, 0.0)

        # Per-user rated popular items + residuals (for fast prediction)
        self._user_rated = {}
        for u, grp in pop_ratings.groupby("userId"):
            js = grp["movieId"].map(self.pop_index).values
            res = grp["rating"].values - self.user_mean[u]
            self._user_rated[u] = (js, res)
        return self

    def predict_one(self, user, movie) -> float:
        base = self.user_mean.get(user, self.global_mean)
        if movie in self.pop_index and user in self._user_rated:
            i = self.pop_index[movie]
            js, res = self._user_rated[user]
            sims = self.sim[i, js]
            if self.k < len(sims):
                top = np.argpartition(np.abs(sims), -self.k)[-self.k:]
                sims, res = sims[top], res[top]
            denom = np.abs(sims).sum()
            if denom > 1e-8:
                return float(np.clip(base + (sims @ res) / denom, 0.5, 5.0))
        return float(np.clip(self.item_mean.get(movie, base), 0.5, 5.0))

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(u, m)
                         for u, m in zip(df["userId"], df["movieId"])])

    def recommend(self, user, movies_df=None, n: int = 10, seen: set | None = None):
        """Top-N movie recommendations for a user (over popular items)."""
        if user not in self._user_rated:
            return []
        seen = seen or set()
        js, res = self._user_rated[user]
        base = self.user_mean.get(user, self.global_mean)
        S = self.sim[:, js]                      # (n_popular, n_rated)
        denom = np.abs(S).sum(axis=1)
        num = S @ res
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 1e-8, num / denom, 0.0)
        scores = base + ratio
        inv_m = {j: m for m, j in self.pop_index.items()}
        out = []
        for j in np.argsort(-scores):
            movie = inv_m[j]
            if movie not in seen:
                out.append((movie, float(np.clip(scores[j], 0.5, 5.0))))
            if len(out) >= n:
                break
        return out


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    train, test = load_splits()

    # Only score test pairs whose movie was seen in train
    seen_movies = set(train["movieId"].unique())
    test_scorable = test[test["movieId"].isin(seen_movies)].copy()
    y = test_scorable["rating"].values

    global_mean = train["rating"].mean()
    user_mean = train.groupby("userId")["rating"].mean()
    item_mean = train.groupby("movieId")["rating"].mean()

    results = {}
    results["Global mean"] = rmse(y, np.full(len(y), global_mean))
    results["User mean"] = rmse(
        y, test_scorable["userId"].map(user_mean).fillna(global_mean).values)
    results["Movie mean"] = rmse(
        y, test_scorable["movieId"].map(item_mean).fillna(global_mean).values)

    print("Fitting item-item CF...")
    cf = ItemItemCF().fit(train)
    results["Item-item CF"] = rmse(y, cf.predict_df(test_scorable))

    print("\n=== Test RMSE (lower is better) ===")
    for name, r in results.items():
        print(f"  {r:.4f}  {name}")
    best = min(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.4f})")

    # Chart
    plt.figure(figsize=(7, 4.5))
    names = list(results.keys())
    vals = [results[n] for n in names]
    colors = ["#bbb", "#bbb", "#bbb", "#4c72b0"]
    bars = plt.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    plt.ylabel("Test RMSE"); plt.title("Baselines vs Item-item CF"); plt.ylim(0.8, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "memory_cf_rmse.png", dpi=90, bbox_inches="tight")
    plt.close()
    (OUTPUTS_DIR / "memory_cf_rmse.json").write_text(
        json.dumps({k: round(v, 4) for k, v in results.items()}, indent=2))
    print(f"\nSaved chart to outputs/memory_cf_rmse.png")


if __name__ == "__main__":
    main()
