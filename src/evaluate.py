"""
Final evaluation for the movie recommender.

Two complementary views:
  1. RMSE — how accurately each method predicts held-out ratings.
  2. Ranking (precision@k / recall@k) — of the top-k movies we'd actually
     recommend, how many the user rated highly in the test set. This matters
     more than RMSE for a real recommender, where the top of the list is what
     users see.

Models compared on ranking: Popularity (non-personalized), Content-based,
Item-item CF, and Matrix Factorization.

Run:
    python -m src.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data_prep import load_data, load_splits
from src.memory_cf import ItemItemCF
from src.mf import MatrixFactorizationSVD
from src.content_based import ContentRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
K = 10
RELEVANCE = 4.0


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def ranking_eval(recommend_fn, train, test, k=K, relevance=RELEVANCE):
    """Mean precision@k and recall@k over users with >=1 relevant test item."""
    train_seen = train.groupby("userId")["movieId"].apply(set)
    relevant_by_user = (test[test["rating"] >= relevance]
                        .groupby("userId")["movieId"].apply(set))
    precs, recs = [], []
    for user, relevant in relevant_by_user.items():
        seen = train_seen.get(user, set())
        recommended = recommend_fn(user, k, seen)
        rec_ids = [m for m, _ in recommended] if recommended and isinstance(
            recommended[0], tuple) else list(recommended)
        hits = len(set(rec_ids) & relevant)
        precs.append(hits / k)
        recs.append(hits / len(relevant))
    return float(np.mean(precs)), float(np.mean(recs))


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    ratings, movies = load_data()
    train, test = load_splits()
    seen_movies = set(train["movieId"].unique())
    test_scorable = test[test["movieId"].isin(seen_movies)].copy()
    y = test_scorable["rating"].values

    global_mean = train["rating"].mean()
    user_mean = train.groupby("userId")["rating"].mean()
    item_mean = train.groupby("movieId")["rating"].mean()

    # --- Fit / load models ---
    print("Fitting item-item CF...")
    cf = ItemItemCF().fit(train)
    mf_path = MODELS_DIR / "mf_model.joblib"
    if mf_path.exists():
        mf = joblib.load(mf_path)
    else:
        print("Training MF...")
        mf = MatrixFactorizationSVD().fit(train, verbose=False)
    content = ContentRecommender().fit(movies, train)

    # --- RMSE ---
    rmse_results = {
        "Global mean": rmse(y, np.full(len(y), global_mean)),
        "User mean": rmse(y, test_scorable["userId"].map(user_mean).fillna(global_mean)),
        "Movie mean": rmse(y, test_scorable["movieId"].map(item_mean).fillna(global_mean)),
        "Item-item CF": rmse(y, cf.predict_df(test_scorable)),
        "Matrix Factorization": rmse(y, mf.predict_df(test_scorable)),
    }
    print("\n=== RMSE (lower is better) ===")
    for k_, v in rmse_results.items():
        print(f"  {v:.4f}  {k_}")

    # --- Ranking (precision@k / recall@k) ---
    pop = train.groupby("movieId").size().sort_values(ascending=False)

    def pop_rec(user, n, seen):
        out = []
        for m in pop.index:
            if m not in seen:
                out.append(m)
            if len(out) >= n:
                break
        return out

    rankers = {
        "Popularity": pop_rec,
        "Content-based": lambda u, n, s: content.recommend(u, n, s),
        "Item-item CF": lambda u, n, s: cf.recommend(u, n=n, seen=s),
        "Matrix Factorization": lambda u, n, s: mf.recommend(u, n=n, seen=s),
    }
    print(f"\n=== Ranking @ {K} (higher is better) ===")
    ranking_results = {}
    for name, fn in rankers.items():
        p, r = ranking_eval(fn, train, test)
        ranking_results[name] = {"precision": p, "recall": r}
        print(f"  {name:<22} precision@{K}={p:.3f}  recall@{K}={r:.3f}")

    # --- Charts ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(rmse_results.keys())
    vals = [rmse_results[n] for n in names]
    colors = ["#bbb", "#bbb", "#bbb", "#4c72b0", "#55a868"]
    a1.bar(names, vals, color=colors)
    a1.set_title("Rating accuracy — RMSE"); a1.set_ylim(0.8, 1.1); a1.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals): a1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    rnames = list(ranking_results.keys())
    rvals = [ranking_results[n]["precision"] for n in rnames]
    a2.bar(rnames, rvals, color=["#bbb", "#dd8452", "#4c72b0", "#55a868"])
    a2.set_title(f"Ranking quality — precision@{K}"); a2.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rvals): a2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "final_comparison.png", dpi=90, bbox_inches="tight")
    plt.close()

    (OUTPUTS_DIR / "final_metrics.json").write_text(json.dumps({
        "rmse": {k_: round(v, 4) for k_, v in rmse_results.items()},
        "ranking": {k_: {m: round(x, 4) for m, x in v.items()}
                    for k_, v in ranking_results.items()},
    }, indent=2))

    # --- Example recommendations (MF) for a sample user ---
    title = movies.set_index("movieId")["title"]
    user = 1
    seen = set(train[train["userId"] == user]["movieId"])
    print(f"\n=== Top-5 recommendations for user {user} (Matrix Factorization) ===")
    for movie, score in mf.recommend(user, n=5, seen=seen):
        print(f"  {score:.2f}  {title.get(movie, movie)}")
    print(f"\nSaved comparison chart to outputs/final_comparison.png")


if __name__ == "__main__":
    main()
