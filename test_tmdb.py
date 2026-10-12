import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TMDB_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    response = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": api_key, "query": "Inception"},
        timeout=15,
    )

    print("Status code:", response.status_code)

    if response.ok:
        movies = response.json().get("results", [])
        for movie in movies[:3]:
            print(movie["title"], "-", movie.get("poster_path"))
    else:
        print("Request failed:", response.text)