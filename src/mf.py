"""
Matrix factorization (bias-aware funkSVD), implemented from scratch with SGD.

Model:
    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

where mu is the global mean, b_u / b_i are user/movie bias terms, and p_u / q_i
are latent factor vectors learned so their dot product captures taste alignment
(e.g. "likes action", "prefers indie"). Learning these latent factors lets the
model generalize across the ~98% sparsity far better than similarity alone.

Trained by stochastic gradient descent minimizing squared error with L2
regularization. No external recommender library required.

Run:
    python -m src.mf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data_prep import load_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class MatrixFactorizationSVD:
    def __init__(self, n_factors=40, n_epochs=25, lr=0.005, reg=0.02, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train: pd.DataFrame, verbose: bool = True):
        rng = np.random.RandomState(self.seed)
        users = np.sort(train["userId"].unique())
        movies = np.sort(train["movieId"].unique())
        self.uidx = {u: i for i, u in enumerate(users)}
        self.midx = {m: j for j, m in enumerate(movies)}

        u = train["userId"].map(self.uidx).values
        i = train["movieId"].map(self.midx).values
        r = train["rating"].values.astype(np.float64)

        self.mu = r.mean()
        n_u, n_i, f = len(users), len(movies), self.n_factors
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)
        self.P = rng.normal(0, 0.1, (n_u, f))
        self.Q = rng.normal(0, 0.1, (n_i, f))
        lr, reg = self.lr, self.reg

        order = np.arange(len(r))
        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            for idx in order:
                uu, ii, rr = u[idx], i[idx], r[idx]
                pred = self.mu + self.bu[uu] + self.bi[ii] + self.P[uu] @ self.Q[ii]
                e = rr - pred
                self.bu[uu] += lr * (e - reg * self.bu[uu])
                self.bi[ii] += lr * (e - reg * self.bi[ii])
                pu = self.P[uu].copy()
                self.P[uu] += lr * (e * self.Q[ii] - reg * self.P[uu])
                self.Q[ii] += lr * (e * pu - reg * self.Q[ii])
            if verbose:
                tr = self._rmse_arrays(u, i, r)
                print(f"  epoch {epoch + 1:2d}/{self.n_epochs}  train RMSE {tr:.4f}")
        return self

    def _rmse_arrays(self, u, i, r):
        pred = (self.mu + self.bu[u] + self.bi[i]
                + np.einsum("ij,ij->i", self.P[u], self.Q[i]))
        return np.sqrt(mean_squared_error(r, pred))

    def predict_one(self, user, movie) -> float:
        pred = self.mu
        known_u = user in self.uidx
        known_i = movie in self.midx
        if known_u:
            pred += self.bu[self.uidx[user]]
        if known_i:
            pred += self.bi[self.midx[movie]]
        if known_u and known_i:
            pred += self.P[self.uidx[user]] @ self.Q[self.midx[movie]]
        return float(np.clip(pred, 0.5, 5.0))

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(u, m)
                         for u, m in zip(df["userId"], df["movieId"])])

    def recommend(self, user, n: int = 10, seen: set | None = None):
        if user not in self.uidx:
            return []
        seen = seen or set()
        uu = self.uidx[user]
        scores = self.mu + self.bu[uu] + self.bi + self.Q @ self.P[uu]
        inv_m = {j: m for m, j in self.midx.items()}
        order = np.argsort(-scores)
        out = []
        for j in order:
            movie = inv_m[j]
            if movie not in seen:
                out.append((movie, float(np.clip(scores[j], 0.5, 5.0))))
            if len(out) >= n:
                break
        return out


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    train, test = load_splits()
    seen_movies = set(train["movieId"].unique())
    test_scorable = test[test["movieId"].isin(seen_movies)].copy()

    print("Training matrix factorization (funkSVD)...")
    mf = MatrixFactorizationSVD().fit(train)

    test_rmse = rmse(test_scorable["rating"].values, mf.predict_df(test_scorable))
    print(f"\nTest RMSE (MF): {test_rmse:.4f}")

    import joblib
    joblib.dump(mf, MODELS_DIR / "mf_model.joblib")
    (OUTPUTS_DIR / "mf_rmse.json").write_text(f'{{"mf_test_rmse": {test_rmse:.4f}}}')
    print(f"Saved model -> {MODELS_DIR / 'mf_model.joblib'}")


if __name__ == "__main__":
    main()
