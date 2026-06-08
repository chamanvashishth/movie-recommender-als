import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


class ColdStartRecommender:

    def __init__(self):

        self.movies = None
        self.genre_matrix = None
        self.mlb = MultiLabelBinarizer()

    def fit(
        self,
        movies_df,
        ratings_df
    ):

        self.movies = movies_df.copy()

        genre_lists = (
            self.movies.genres
            .str.split("|")
            .tolist()
        )

        self.genre_matrix = (
            self.mlb.fit_transform(
                genre_lists
            )
        )

        popularity = (
            ratings_df.groupby("movieId")
            .agg(
                avg_rating=("rating", "mean"),
                rating_count=("rating", "count")
            )
            .reset_index()
        )

        self.movies = self.movies.merge(
            popularity,
            on="movieId",
            how="left"
        )

        self.movies["avg_rating"] = (
            self.movies["avg_rating"]
            .fillna(0)
        )

        self.movies["rating_count"] = (
            self.movies["rating_count"]
            .fillna(0)
        )

    def recommend(
        self,
        selected_genres,
        top_k=10
    ):

        profile = np.zeros(
            len(self.mlb.classes_)
        )

        for genre in selected_genres:

            if genre in self.mlb.classes_:

                idx = list(
                    self.mlb.classes_
                ).index(genre)

                profile[idx] = 1

        sims = cosine_similarity(
            profile.reshape(1, -1),
            self.genre_matrix
        )[0]

        df = self.movies.copy()

        df["similarity"] = sims

        df["score"] = (
            0.7 * df["similarity"]
            + 0.3 * (
                df["avg_rating"] / 5.0
            )
        )

        return (
            df.sort_values(
                "score",
                ascending=False
            )
            .head(top_k)
        )
