"""Streamlit web app.   Run:  streamlit run app/app.py"""
import os
import sys
from pathlib import Path

# Make "src" importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from src.feature_engineering import load_features
from src.recommendation_model import Recommender, load_model

try:  # optional: load TMDB_API_KEY from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")


# ------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading movie data and model...")
def get_recommender():
    """Load once and reuse across reruns (the similarity matrix is large)."""
    _, similarity = load_model()
    return Recommender(load_features(), similarity)


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(tmdb_id, api_key):
    """Fetch a poster URL from TMDB. Returns None on any problem."""
    if not api_key:
        return None
    try:
        resp = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
                            params={"api_key": api_key}, timeout=5)
        resp.raise_for_status()
        path = resp.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except (requests.RequestException, ValueError):
        return None


def show_movie_card(container, row, api_key, show_score=True):
    with container:
        poster = get_poster_url(row["id"], api_key)
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.caption("🎞️ No poster available")
        st.markdown(f"**{row['title']}**")
        st.caption(", ".join(row["genres_list"]) or "Genre unknown")
        st.write(f"⭐ {row['vote_average']:.1f}  ·  🔥 {row['popularity']:.0f}")
        if show_score:
            st.progress(float(max(0.0, min(1.0, row["similarity"]))),
                        text=f"Similarity {row['similarity']:.2f}")


# ---------------------------------------------------------------------- UI
st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favorite movies using "
         "Data Science and Machine Learning.")

try:
    recommender = get_recommender()
except FileNotFoundError as err:
    st.error("The dataset or model files are missing.")
    st.info("Follow the README: download the dataset, then run the three build "
            "commands (preprocessing, features, model).")
    st.caption(str(err))
    st.stop()
except (RuntimeError, ValueError) as err:
    st.error("The model could not be loaded. Please rebuild it "
             "(`python -m src.recommendation_model`).")
    st.caption(str(err))
    st.stop()

api_key = os.getenv("TMDB_API_KEY", "").strip()
if api_key in ("", "your_api_key_here"):
    api_key = ""

all_genres = sorted({g for lst in recommender.df["genres_list"] for g in lst})

with st.sidebar:
    st.header("Filters")
    genre = st.selectbox("Genre", ["All"] + all_genres)
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)
    top_n = st.selectbox("Recommendations", [5, 10, 15], index=0)
    st.divider()
    st.caption("Posters: " + ("enabled (TMDB)" if api_key else
                              "off - set TMDB_API_KEY in .env to enable"))
    st.caption("Content-based filtering: TF-IDF + cosine similarity")

titles = sorted(recommender.df["title"].tolist())  # selectbox is searchable
selected = st.selectbox("Search or select a movie:", titles, index=None,
                        placeholder="Start typing a movie name...")

if st.button("Recommend Movies", type="primary"):
    if not selected:
        st.warning("Please choose a movie first.")
    else:
        try:
            with st.spinner("Finding similar movies..."):
                results = recommender.recommend(selected, top_n, genre, min_rating)
        except ValueError as err:
            st.error(str(err))
        else:
            chosen = recommender.df.iloc[recommender.find_movie(selected)].copy()
            chosen["similarity"] = 1.0
            st.subheader("Selected Movie")
            left, _ = st.columns([1, 3])
            show_movie_card(left, chosen, api_key, show_score=False)

            st.subheader("Recommended Movies")
            if results.empty:
                st.info("No movies match these filters. Try lowering the minimum "
                        "rating or choosing Genre = All.")
            else:
                if len(results) < top_n:
                    st.caption(f"Only {len(results)} movies matched your filters.")
                for start in range(0, len(results), 5):  # 5 cards per row
                    cols = st.columns(5)
                    for col, (_, row) in zip(cols, results.iloc[start:start + 5].iterrows()):
                        show_movie_card(col, row, api_key)
