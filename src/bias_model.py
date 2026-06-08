import numpy as np


class BiasModel:

    def __init__(self):
        self.global_mean = 0.0
        self.user_bias = None
        self.item_bias = None

    def fit(self, ratings_df):

        self.global_mean = ratings_df.rating.mean()

        n_users = ratings_df.userId.max()
        n_items = ratings_df.movieId.max()

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        for user_id, grp in ratings_df.groupby("userId"):

            self.user_bias[user_id - 1] = (
                grp.rating.mean()
                - self.global_mean
            )

        for item_id, grp in ratings_df.groupby("movieId"):

            self.item_bias[item_id - 1] = (
                grp.rating.mean()
                - self.global_mean
            )

        return self

    def predict(self, user_idx, item_idx):

        return (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
        )
