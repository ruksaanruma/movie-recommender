# Data

`ratings.csv` and `movies.csv` (~2.9 MB total) are committed directly to this
repo, so no download is needed.

## Source
**MovieLens (ml-latest-small)** — 100,836 ratings and 3,683 tag applications
across 9,742 movies by 610 users (1996–2018). A standard benchmark for
recommender systems, from GroupLens at the University of Minnesota.

- GroupLens: https://grouplens.org/datasets/movielens/
- Mirror used here: https://github.com/smanihwr/ml-latest-small

## Files
- `ratings.csv` — `userId, movieId, rating, timestamp` (ratings 0.5–5.0).
- `movies.csv` — `movieId, title, genres` (genres are pipe-separated).

## Key characteristics
- **Very sparse:** the 610 × 9,724 user–movie matrix is ~98.3% empty — the
  central challenge that collaborative filtering addresses.
- **Long tail:** ~62% of movies have fewer than 5 ratings (cold-start).
- Every user has rated at least 20 movies.
