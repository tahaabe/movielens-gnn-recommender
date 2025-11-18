import pandas as pd

def load_movielens():
    ratings_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    movies_url  = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"

    ratings = pd.read_csv(
        ratings_url,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    movies_cols = [
        "movieId", "title", "release_date", "video_release", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = pd.read_csv(
        movies_url,
        sep="|",
        names=movies_cols,
        encoding="latin-1"
    )

    return ratings, movies
