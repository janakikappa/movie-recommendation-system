"""Stages 1-4: data collection, understanding and cleaning.

Run:  python -m src.data_preprocessing
"""
import pandas as pd

from src.utils import (CLEAN_CSV, CREDITS_CSV, MOVIES_CSV, PROCESSED_DIR,
                       get_director, parse_json_names)


def load_raw_data():
    """Load the two TMDB 5000 CSV files and merge them into one table."""
    if not MOVIES_CSV.exists() or not CREDITS_CSV.exists():
        raise FileNotFoundError(
            "Dataset not found. Download the TMDB 5000 Movie Dataset from "
            "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata and place "
            f"tmdb_5000_movies.csv and tmdb_5000_credits.csv in {MOVIES_CSV.parent}"
        )
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    if movies.empty or credits.empty:
        raise ValueError("One of the dataset files is empty.")
    # movies.id and credits.movie_id are the same TMDB id -> safe merge key
    merged = movies.merge(credits[["movie_id", "cast", "crew"]],
                          left_on="id", right_on="movie_id", how="left")
    return movies, credits, merged


def understand_data(df):
    """Print the basic data-understanding checks."""
    print("Shape (rows, columns):", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nSample records:\n", df.head(3))
    print("\nData types:\n", df.dtypes)
    missing = df.isnull().sum()
    print("\nMissing values:\n", missing[missing > 0])
    print("\nDuplicate movie ids:", df.duplicated(subset=["id"]).sum())


def clean_data(df):
    """Clean the merged table. Each step explains WHY it is done."""
    df = df.copy()
    start = len(df)

    # 1. Duplicates: the same movie twice would be recommended to itself.
    df = df.drop_duplicates(subset="id")

    # 2. Invalid records: a movie without a title cannot be shown or searched.
    df = df.dropna(subset=["title"])
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""]
    # Two different movies can share a title; keep the more popular one so the
    # title -> row lookup in the app is unambiguous.
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df = df.sort_values("popularity", ascending=False).drop_duplicates("title")

    # 3. Missing text: fill with "" instead of dropping the movie, because
    #    genres/cast/keywords may still describe it well.
    df["overview"] = df["overview"].fillna("").astype(str)

    # 4. Wrong data types: make numbers numeric (bad values become 0).
    for col in ["vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    # 5. Turn JSON strings into usable Python lists / strings.
    df["genres_list"] = df["genres"].apply(parse_json_names)
    df["keywords_list"] = df["keywords"].apply(parse_json_names)
    df["cast_list"] = df["cast"].apply(lambda v: parse_json_names(v, limit=3))
    df["director"] = df["crew"].apply(get_director)

    # 6. Drop movies with no usable information at all (nothing to compare).
    has_info = ((df["genres_list"].str.len() > 0) | (df["keywords_list"].str.len() > 0)
                | (df["overview"].str.strip() != "") | (df["cast_list"].str.len() > 0))
    df = df[has_info]

    # 7. Remove columns that are not useful for recommendation.
    keep = ["id", "title", "overview", "genres_list", "keywords_list",
            "cast_list", "director", "vote_average", "vote_count",
            "popularity", "year"]
    df = df[keep].reset_index(drop=True)
    print(f"Cleaning: {start} -> {len(df)} movies kept")
    return df


def save_clean(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Lists are stored with "|" separators so the CSV stays readable.
    out = df.copy()
    for col in ["genres_list", "keywords_list", "cast_list"]:
        out[col] = out[col].apply("|".join)
    out.to_csv(CLEAN_CSV, index=False)
    print("Saved", CLEAN_CSV)


def load_clean():
    """Read the cleaned CSV back, restoring the list columns."""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError("Run `python -m src.data_preprocessing` first.")
    df = pd.read_csv(CLEAN_CSV)
    for col in ["genres_list", "keywords_list", "cast_list"]:
        df[col] = df[col].fillna("").apply(lambda s: s.split("|") if s else [])
    df["overview"] = df["overview"].fillna("")
    df["director"] = df["director"].fillna("")
    return df


if __name__ == "__main__":
    _, _, merged = load_raw_data()
    understand_data(merged)
    save_clean(clean_data(merged))
