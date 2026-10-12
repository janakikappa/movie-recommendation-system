"""Stage 6: Feature engineering - build one text "document" per movie.

Run:  python -m src.feature_engineering
"""
import re

import pandas as pd

from src.data_preprocessing import load_clean
from src.utils import FEATURES_CSV


def clean_text(text):
    """Lowercase, drop punctuation/odd characters, collapse spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def squash(names):
    """'Sam Worthington' -> 'samworthington'.
    Keeps a person's full name as ONE token, so 'Sam Worthington' and
    'Sam Neill' are not treated as sharing the word 'sam'."""
    return [re.sub(r"[^a-z0-9]", "", n.lower()) for n in names if n]


def build_combined_features(row):
    """genres + keywords + cast + director + overview -> one string.
    Genres and director are repeated so they carry a bit more weight; the
    overview supplies descriptive words."""
    genres = squash(row["genres_list"])
    keywords = squash(row["keywords_list"])
    cast = squash(row["cast_list"])
    director = squash([row["director"]])
    parts = genres * 2 + keywords + cast + director * 2
    return " ".join(parts) + " " + clean_text(row["overview"])


def add_features(df):
    df = df.copy()
    df["combined_features"] = df.apply(build_combined_features, axis=1)
    # Safety: a movie whose combined text ended up empty cannot be compared
    df = df[df["combined_features"].str.strip() != ""].reset_index(drop=True)
    return df


def save_features(df):
    out = df.copy()
    for col in ["genres_list", "keywords_list", "cast_list"]:
        out[col] = out[col].apply("|".join)
    out.to_csv(FEATURES_CSV, index=False)
    print("Saved", FEATURES_CSV)


def load_features():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError("Run `python -m src.feature_engineering` first.")
    df = pd.read_csv(FEATURES_CSV)
    for col in ["genres_list", "keywords_list", "cast_list"]:
        df[col] = df[col].fillna("").apply(lambda s: s.split("|") if s else [])
    df["director"] = df["director"].fillna("")
    df["combined_features"] = df["combined_features"].fillna("")
    return df


if __name__ == "__main__":
    data = add_features(load_clean())
    print(data[["title", "combined_features"]].head(3).to_string())
    save_features(data)
