#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The MovieLens ml-latest-small CSVs (~2.9 MB) are committed to this repo at
# data/ratings.csv and data/movies.csv, so you normally do NOT need this
# script. It's here only to re-fetch them from a mirror if needed.
#
#   bash scripts/download_data.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
BASE="https://raw.githubusercontent.com/smanihwr/ml-latest-small/master"
echo ">> Fetching MovieLens ml-latest-small CSVs..."
curl -sSL -o data/ratings.csv "$BASE/ratings.csv"
curl -sSL -o data/movies.csv  "$BASE/movies.csv"
echo ">> Done. Ratings: $(($(wc -l < data/ratings.csv) - 1)) | Movies: $(($(wc -l < data/movies.csv) - 1))"
