"""
Content-based recommender using movie genres.

Each movie is represented by a multi-hot genre vector. A user's taste profile is
the rating-weighted average of the genre vectors of movies they liked. We then
recommend unseen movies whose genres best match that profile (cosine similarity).

Unlike collaborative filtering, this can recommend movies with very few ratings
(the long-tail / cold-start items) because it relies on genre content, not on
other users having rated them.

Run:
    python -m src.content_based
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.data_prep import load_data, load_splits


class ContentRecommender:
    def fit(self, movies: pd.DataFrame, train: pd.DataFrame):
        genres = movies["genres"].str.split("|")
        self.mlb = MultiLabelBinarizer()
        G = self.mlb.fit_transform(genres).astype(float)
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.G = G / norms                       # row-normalized genre vectors
        self.movie_ids = movies["movieId"].values
        self.row_of = {m: i for i, m in enumerate(self.movie_ids)}
        self.train = train
        self._by_user = {u: g for u, g in train.groupby("userId")}
        return self

    def _profile(self, user):
        g = self._by_user.get(user)
        if g is None:
            return None
        liked = g[g["rating"] >= 4.0]
        if len(liked) == 0:
            liked = g
        rows, weights = [], []
        for m, r in zip(liked["movieId"], liked["rating"]):
            if m in self.row_of:
                rows.append(self.row_of[m]); weights.append(r)
        if not rows:
            return None
        w = np.array(weights)
        return (self.G[rows] * w[:, None]).sum(axis=0) / w.sum()

    def recommend(self, user, n: int = 10, seen: set | None = None):
        profile = self._profile(user)
        if profile is None:
            return []
        seen = seen or set()
        scores = self.G @ profile
        out = []
        for i in np.argsort(-scores):
            movie = self.movie_ids[i]
            if movie not in seen:
                out.append((movie, float(scores[i])))
            if len(out) >= n:
                break
        return out


if __name__ == "__main__":
    ratings, movies = load_data()
    train, _ = load_splits()
    rec = ContentRecommender().fit(movies, train)
    title = movies.set_index("movieId")["title"]
    user = 1
    seen = set(train[train["userId"] == user]["movieId"])
    print(f"Genres: {list(rec.mlb.classes_)}\n")
    print(f"Content-based recommendations for user {user}:")
    for movie, score in rec.recommend(user, n=10, seen=seen):
        print(f"  {score:.3f}  {title.get(movie, movie)}")
