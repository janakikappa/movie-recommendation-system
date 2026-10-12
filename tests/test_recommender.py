"""Run with:  python -m pytest tests   (or: python tests/test_recommender.py)
Requires the model to be built first."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommendation_model import get_default_recommender


def test_flow():
    rec = get_default_recommender()
    title = rec.df.loc[0, "title"]
    out = rec.recommend(f"  {title.upper()}  ", 5)       # case/space tolerant
    assert 0 < len(out) <= 5
    assert title not in out["title"].values              # never recommends itself
    assert out["similarity"].is_monotonic_decreasing     # sorted best-first
    try:
        rec.recommend("this movie does not exist 12345")
        assert False, "should raise"
    except ValueError:
        pass


if __name__ == "__main__":
    test_flow()
    print("All tests passed")
