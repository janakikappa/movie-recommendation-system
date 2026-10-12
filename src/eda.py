"""Stage 5: Exploratory Data Analysis. Saves charts to assets/.

Run:  python -m src.eda
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # save to files without opening windows
import matplotlib.pyplot as plt

from src.data_preprocessing import load_clean
from src.utils import ASSETS_DIR


def _save(fig, name):
    ASSETS_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / name, dpi=120)
    plt.close(fig)


def overview(df):
    genres = Counter(g for lst in df["genres_list"] for g in lst)
    print("Number of movies:", len(df))
    print("Number of genres:", len(genres))
    print("Missing overviews:", (df["overview"].str.strip() == "").sum())
    print("Duplicate titles:", df["title"].duplicated().sum())
    return genres


def run_eda(df):
    genres = overview(df)

    # Top 10 most popular movies
    top = df.nlargest(10, "popularity").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["title"], top["popularity"], color="#1f4e79")
    ax.set(title="Top 10 Most Popular Movies", xlabel="Popularity score", ylabel="Movie")
    _save(fig, "top_popular.png")

    # Top 10 highest rated (require many votes so 1-vote movies don't win)
    min_votes = 500 if (df["vote_count"] >= 500).sum() >= 10 else 0
    rated = df[df["vote_count"] >= min_votes].nlargest(10, "vote_average").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rated["title"], rated["vote_average"], color="#2e8b57")
    ax.set(title=f"Top 10 Highest-Rated Movies ({min_votes}+ votes)",
           xlabel="Average rating", ylabel="Movie")
    _save(fig, "top_rated.png")

    # Rating distribution (ignore 0 = unrated)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.loc[df["vote_average"] > 0, "vote_average"], bins=20, color="#c0504d")
    ax.set(title="Distribution of Ratings", xlabel="Average rating", ylabel="Number of movies")
    _save(fig, "rating_distribution.png")

    # Popularity is heavily skewed, so clip the top 1% for a readable histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    pop = df["popularity"]
    ax.hist(pop[pop <= pop.quantile(0.99)], bins=30, color="#8064a2")
    ax.set(title="Distribution of Popularity (top 1% clipped)",
           xlabel="Popularity score", ylabel="Number of movies")
    _save(fig, "popularity_distribution.png")

    # Movies per genre
    common = genres.most_common(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([g for g, _ in common], [c for _, c in common], color="#e69f00")
    ax.set(title="Number of Movies by Genre (Top 10)", xlabel="Genre", ylabel="Number of movies")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "genre_counts.png")

    # Pie chart: genre share
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([c for _, c in common], labels=[g for g, _ in common], autopct="%1.0f%%")
    ax.set_title("Share of Top 10 Genres")
    _save(fig, "genre_share.png")

    # Box plot: rating by genre
    names = {g for g, _ in common}
    by_genre = {}
    for glist, rating in zip(df["genres_list"], df["vote_average"]):
        if rating > 0:
            for g in glist:
                if g in names:
                    by_genre.setdefault(g, []).append(rating)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(list(by_genre.values()), tick_labels=list(by_genre.keys()))
    ax.set(title="Rating Spread by Genre", xlabel="Genre", ylabel="Average rating")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "rating_by_genre_box.png")
    print("EDA charts saved in", ASSETS_DIR)


if __name__ == "__main__":
    run_eda(load_clean())
