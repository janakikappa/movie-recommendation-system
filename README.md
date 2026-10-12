# 🎬 Movie Recommendation System Using Data Science and Machine Learning

## 1. Problem Statement
Users face thousands of movies and struggle to choose. This project recommends movies similar to one the user already likes, using the movie's own data.

## 2. Objectives
- Apply the full Data Science lifecycle (collect → clean → EDA → features → model → evaluate → app).
- Build a content-based recommender (TF-IDF + cosine similarity).
- Deliver an interactive Streamlit web app with filters and optional posters.

## 3. Features
Searchable movie selector (case-insensitive) · Top 5/10/15 results · Genre and minimum-rating filters · Similarity score, rating, popularity per movie · Optional TMDB posters · Friendly error messages · Cached model loading.

## 4. Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn (TF-IDF, cosine similarity), Joblib, Streamlit, Requests.

## 5. Dataset
**TMDB 5000 Movie Dataset** (not included in this repo): https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in `data/raw/`.

## 6. Data Science Methodology
| Stage | Where |
|---|---|
| Collection & understanding | `src/data_preprocessing.py` |
| Cleaning | `src/data_preprocessing.py` |
| EDA | `src/eda.py` (charts → `assets/`) |
| Feature engineering | `src/feature_engineering.py` |
| TF-IDF, similarity, evaluation | `src/recommendation_model.py` |
| Application | `app/app.py` |
| Testing | `tests/test_recommender.py` |

## 7. System Architecture
```
CSV files → Cleaning → EDA → combined_features → TF-IDF vectors
→ Cosine similarity matrix (saved in models/) → Recommender → Streamlit UI
```

## 8. Algorithm
1. Each movie becomes one text: genres + keywords + cast + director + overview (names joined into single tokens, e.g. `samworthington`).
2. **TF-IDF** converts text to vectors; distinctive words get higher weight than common ones.
3. **Cosine similarity** compares every pair of vectors (0 = unrelated, 1 = identical direction).
4. For a chosen movie: look up its row → sort scores → drop itself → apply filters → return top N.

## 9. Project Structure
```
data/raw, data/processed   notebooks/movie_analysis.ipynb
src/ (utils, data_preprocessing, eda, feature_engineering, recommendation_model)
models/ (vectorizer.pkl, similarity.pkl)   app/app.py   assets/   tests/
requirements.txt  .env.example  .gitignore  README.md
```

## 10. Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add a TMDB key **only if** you want posters.

## 11. How to Run (from the project root)
```bash
python -m src.data_preprocessing     # clean data
python -m src.eda                    # create charts
python -m src.feature_engineering    # build combined_features
python -m src.recommendation_model   # TF-IDF, similarity, save models, print evaluation
python tests/test_recommender.py     # test
streamlit run app/app.py             # launch the app
```

## 12. Screenshots
Add your own after running: `assets/screenshot_home.png`, `assets/screenshot_results.png`. EDA charts are generated in `assets/`.

## 13. Results
Run the model script and record the printed numbers here: average similarity, genre overlap, diversity, coverage. *(Not filled in: these depend on your actual dataset run.)*

**Evaluation note:** there is no ground truth (no user watch data), so accuracy/precision/recall are NOT reported. The metrics are sanity checks: genre overlap is a rough relevance proxy, diversity shows how repetitive lists are, coverage shows how much of the catalogue gets recommended. Final judgment needs user feedback.

## 14. Memory vs Computation Trade-off
The 4800×4800 float32 matrix is ~90 MB in RAM. Precomputing it makes each recommendation a fast row lookup, but costs memory and grows quadratically with catalogue size. Computing on demand would save memory but be slower for each click.

## 15. Limitations
- Only content is used: no user tastes, so no personalization.
- Sparse overviews give weak matches; sequels/franchises dominate results.
- Popular-movie bias is not corrected; the dataset is old (up to ~2017).
- Quadratic similarity matrix does not scale to millions of movies.

## 16. Future Enhancements
Collaborative filtering, hybrid recommender, user login/ratings/watch history, personalization, deep learning and semantic (embedding-based) NLP, recommendation explanations, mobile app, cloud deployment.

## 17. Conclusion
The project shows the complete Data Science workflow, from raw data to a working ML-powered application, using explainable techniques.

---
## Viva Q&A
**What is the project?** A content-based recommender that suggests movies similar to one the user selects.
**Why Data Science?** It involves collection, cleaning, analysis, feature engineering and ML to extract recommendations from data.
**Why TF-IDF?** Converts text to numbers and weights meaningful words higher than common ones.
**Why cosine similarity?** Measures similarity of two vectors by angle, so text length matters less.
**Type of recommender?** Content-based filtering.
**Input / output?** A movie title / a ranked list of similar movies.
**Why not report accuracy?** No labelled ground truth exists; we use similarity, diversity, coverage and qualitative checks instead.
**Cold-start?** New movies can be recommended from metadata alone, which is an advantage over collaborative filtering.
**Why combine cast names into one token?** So "Sam Worthington" and "Sam Neill" don't match on the word "sam".
