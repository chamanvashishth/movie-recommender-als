import numpy as np


class RecommendationExplainer:

    def __init__(
        self,
        model,
        movies_df,
        train_matrix
    ):

        self.model = model
        self.movies_df = movies_df
        self.train_matrix = train_matrix

    def get_highly_rated_movies(
        self,
        user_idx,
        top_n=5
    ):

        row = self.train_matrix[user_idx]

        rated_items = row.indices
        ratings = row.data

        if len(ratings) == 0:
            return []

        order = np.argsort(
            ratings
        )[::-1]

        return rated_items[
            order[:top_n]
        ]

    def explain(
        self,
        user_idx,
        recommended_item
    ):

        favorites = (
            self.get_highly_rated_movies(
                user_idx
            )
        )

        if len(favorites) == 0:
            return (
                "Popular among users "
                "with similar tastes."
            )

        rec_vector = (
            self.model.V[
                recommended_item
            ]
        )

        best_movie = None
        best_similarity = -1

        for movie_idx in favorites:

            movie_vector = (
                self.model.V[movie_idx]
            )

            similarity = (
                np.dot(
                    rec_vector,
                    movie_vector
                )
                /
                (
                    np.linalg.norm(
                        rec_vector
                    )
                    *
                    np.linalg.norm(
                        movie_vector
                    )
                    + 1e-8
                )
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_movie = movie_idx

        title = (
            self.movies_df[
                self.movies_df.movieId
                == best_movie + 1
            ]
            .iloc[0]
            .title
        )

        return (
            f"Because you rated "
            f"{title} highly"
        )

    def confidence(
        self,
        item_idx
    ):

        norm = np.linalg.norm(
            self.model.V[item_idx]
        )

        return min(
            1.0,
            norm / 5
        )
