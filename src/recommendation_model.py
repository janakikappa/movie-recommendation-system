"""Stages 7-9: TF-IDF feature extraction, cosine similarity, recommendations,
and evaluation.

Build the model:  python -m src.recommendation_model
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineering import load_features
from src.utils import MODELS_DIR, SIMILARITY_PKL, VECTORIZER_PKL


# ---------------------------------------------------------------- training
def build_model(df):
    """Fit TF-IDF on combined_features and compute the similarity matrix.

    TF-IDF: each word gets a weight = (how often it appears in THIS movie)
    x (how rare it is across ALL movies). Common words get low weight, rare
    and distinctive words (a director name, a keyword) get high weight.

    Cosine similarity: the angle between two movie vectors. 1 = same
    direction (very similar), 0 = nothing in common.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    vectors = vectorizer.fit_transform(df["combined_features"])
    # float32 halves memory (~90 MB instead of ~180 MB for 4800 movies)
    similarity = cosine_similarity(vectors).astype(np.float32)
    return vectorizer, similarity


def save_model(vectorizer, similarity):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PKL)
    joblib.dump(similarity, SIMILARITY_PKL, compress=3)
    print("Saved", VECTORIZER_PKL, "and", SIMILARITY_PKL)


def load_model():
    """Load saved model files; raise a clear error if missing or corrupted."""
    try:
        return joblib.load(VECTORIZER_PKL), joblib.load(SIMILARITY_PKL)
    except FileNotFoundError:
        raise FileNotFoundError("Model files not found. Run `python -m src.recommendation_model`.")
    except Exception as exc:  # corrupted pickle, version mismatch, ...
        raise RuntimeError(f"Model files could not be loaded ({exc}). Rebuild them.") from exc


# ---------------------------------------------------------- recommendation
class Recommender:
    """Holds the movie table + similarity matrix and answers queries."""

    def __init__(self, df, similarity):
        if df.empty:
            raise ValueError("The movie dataset is empty.")
        if similarity.shape != (len(df), len(df)):
            raise ValueError("Similarity matrix does not match the dataset. Rebuild the model.")
        self.df = df.reset_index(drop=True)
        self.similarity = similarity
        # lowercase title -> row index, so lookup ignores capitalization/spaces
        self.index_by_title = {t.strip().lower(): i for i, t in enumerate(self.df["title"])}

    def find_movie(self, title):
        """Return the row index for a title, or None if not found."""
        if not isinstance(title, str):
            return None
        return self.index_by_title.get(title.strip().lower())

    def recommend(self, movie_title, number_of_recommendations=5,
                  genre=None, min_rating=0.0):
        """Return a DataFrame of similar movies (empty if none).
        Raises ValueError with a friendly message for invalid input."""
        idx = self.find_movie(movie_title)
        if idx is None:
            raise ValueError(f"Movie '{movie_title}' was not found in the dataset.")
        if number_of_recommendations < 1:
            raise ValueError("Number of recommendations must be at least 1.")

        scores = self.similarity[idx]
        order = np.argsort(-scores)               # highest similarity first
        order = order[order != idx]               # remove the movie itself

        result = self.df.iloc[order].copy()
        result["similarity"] = scores[order]
        # Filters are applied BEFORE taking the top N, so you still get N results
        if genre and genre != "All":
            result = result[result["genres_list"].apply(lambda g: genre in g)]
        result = result[result["vote_average"] >= min_rating]
        return result.head(number_of_recommendations).reset_index(drop=True)


def recommend_movies(movie_title, number_of_recommendations=5, recommender=None):
    """Convenience function: returns list of dicts (title + metadata)."""
    recommender = recommender or get_default_recommender()
    result = recommender.recommend(movie_title, number_of_recommendations)
    return result[["title", "genres_list", "vote_average", "popularity", "similarity"]] \
        .to_dict("records")


def get_default_recommender():
    _, similarity = load_model()
    return Recommender(load_features(), similarity)


# -------------------------------------------------------------- evaluation
def evaluate(recommender, n=5, sample_size=300, seed=42):
    """Offline sanity metrics (NOT accuracy - there is no ground truth):
    - avg_similarity : mean similarity score of the top-n results
    - genre_overlap  : mean Jaccard overlap between the query's genres and each
                       recommendation's genres (a rough relevance proxy)
    - diversity      : 1 - mean similarity between recommended movies
                       (higher = less repetitive lists)
    - coverage       : share of all movies that appear in at least one list
    """
    rng = np.random.default_rng(seed)
    df = recommender.df
    sample = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sims, overlaps, diversities, seen = [], [], [], set()
    for i in sample:
        rec = recommender.recommend(df.loc[i, "title"], n)
        if rec.empty:
            continue
        sims.append(rec["similarity"].mean())
        query_genres = set(df.loc[i, "genres_list"])
        for g in rec["genres_list"]:
            union = query_genres | set(g)
            overlaps.append(len(query_genres & set(g)) / len(union) if union else 0)
        ids = [recommender.find_movie(t) for t in rec["title"]]
        seen.update(ids)
        if len(ids) > 1:
            sub = recommender.similarity[np.ix_(ids, ids)]
            off_diag = sub[~np.eye(len(ids), dtype=bool)]
            diversities.append(1 - off_diag.mean())
    return {
        "movies_evaluated": len(sims),
        "avg_similarity": float(np.mean(sims)),
        "genre_overlap": float(np.mean(overlaps)),
        "diversity": float(np.mean(diversities)),
        "coverage": len(seen) / len(df),
    }


if __name__ == "__main__":
    data = load_features()
    vec, sim = build_model(data)
    save_model(vec, sim)
    rec = Recommender(data, sim)
    for k, v in evaluate(rec).items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
