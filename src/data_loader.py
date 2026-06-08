import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
)


class MovieLensLoader:
    """
    Downloads MovieLens-1M dataset and creates
    sparse user-item matrices.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.dataset_dir = os.path.join(data_dir, "ml-1m")

        os.makedirs(self.data_dir, exist_ok=True)

    def download(self):

        zip_path = os.path.join(self.data_dir, "ml-1m.zip")

        if os.path.exists(self.dataset_dir):
            print("Dataset already exists.")
            return

        print("Downloading MovieLens 1M...")

        urllib.request.urlretrieve(
            MOVIELENS_URL,
            zip_path
        )

        print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_dir)

        print("Done.")

    def load_ratings(self):

        path = os.path.join(
            self.dataset_dir,
            "ratings.dat"
        )

        ratings = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=[
                "userId",
                "movieId",
                "rating",
                "timestamp"
            ]
        )

        return ratings

    def load_movies(self):

        path = os.path.join(
            self.dataset_dir,
            "movies.dat"
        )

        movies = pd.read_csv(
            path,
            sep="::",
            engine="python",
            encoding="latin1",
            names=[
                "movieId",
                "title",
                "genres"
            ]
        )

        return movies

    def build_sparse_matrix(self, ratings):

        n_users = ratings.userId.max()
        n_items = ratings.movieId.max()

        rows = ratings.userId.values - 1
        cols = ratings.movieId.values - 1

        values = ratings.rating.values.astype(np.float32)

        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        return matrix

    def train_test_split(self, ratings):

        train_parts = []
        test_parts = []

        for _, group in ratings.groupby("userId"):

            group = group.sort_values("timestamp")

            n_test = max(1, int(len(group) * 0.2))

            test = group.tail(n_test)
            train = group.iloc[:-n_test]

            train_parts.append(train)
            test_parts.append(test)

        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)

        return train_df, test_df
