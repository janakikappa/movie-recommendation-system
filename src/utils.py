"""Shared helpers: file paths and small parsing utilities."""
import ast
from pathlib import Path

# Project root = parent of the "src" folder
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

MOVIES_CSV = RAW_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = RAW_DIR / "tmdb_5000_credits.csv"
CLEAN_CSV = PROCESSED_DIR / "movies_clean.csv"
FEATURES_CSV = PROCESSED_DIR / "movies_features.csv"
VECTORIZER_PKL = MODELS_DIR / "vectorizer.pkl"
SIMILARITY_PKL = MODELS_DIR / "similarity.pkl"


def parse_json_names(value, limit=None):
    """Turn a TMDB JSON-like string such as
    '[{"id": 28, "name": "Action"}, ...]' into ['Action', ...].
    Returns [] if the value is missing or malformed."""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else []
    except (ValueError, SyntaxError):
        return []
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [n for n in names if n]
    return names[:limit] if limit else names


def get_director(crew_value):
    """Find the director's name inside the 'crew' JSON string."""
    try:
        crew = ast.literal_eval(crew_value) if isinstance(crew_value, str) else []
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""
