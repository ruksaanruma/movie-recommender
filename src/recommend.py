"""
Generate personalized top-N movie recommendations for a user.

Trains the matrix-factorization model (best on RMSE) on the ratings and prints
the top recommendations with titles, alongside a few of the user's own
top-rated movies for context.

Run:
    python -m src.recommend --user 1 --n 10
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data_prep import load_data, split_ratings
from src.mf import MatrixFactorizationSVD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", type=int, default=1, help="userId to recommend for.")
    ap.add_argument("--n", type=int, default=10, help="Number of recommendations.")
    args = ap.parse_args()

    ratings, movies = load_data()
    title = movies.set_index("movieId")["title"]

    if args.user not in ratings["userId"].unique():
        raise SystemExit(f"User {args.user} not in the dataset "
                         f"(valid: 1..{ratings['userId'].max()}).")

    # Train on all of this user's data (use full ratings for the live model)
    print("Training matrix-factorization model...")
    mf = MatrixFactorizationSVD().fit(ratings, verbose=False)

    user_ratings = ratings[ratings["userId"] == args.user]
    seen = set(user_ratings["movieId"])

    print(f"\nUser {args.user} has rated {len(seen)} movies. A few they loved:")
    for _, row in user_ratings.sort_values("rating", ascending=False).head(5).iterrows():
        print(f"  {row['rating']:.1f}  {title.get(row['movieId'], row['movieId'])}")

    print(f"\nTop {args.n} recommendations:")
    for movie, score in mf.recommend(args.user, n=args.n, seen=seen):
        print(f"  {score:.2f}  {title.get(movie, movie)}")


if __name__ == "__main__":
    main()
