import numpy as np

from scipy.sparse.linalg import svds

from src.als_optimizer import ALSOptimizer
from src.bias_model import BiasModel


class MatrixFactorizationALS:

    def __init__(
        self,
        factors=20,
        reg=0.01,
        iterations=20
    ):

        self.factors = factors
        self.reg = reg
        self.iterations = iterations

        self.U = None
        self.V = None

        self.bias_model = BiasModel()

        self.history = {
            "train_rmse": []
        }

    def initialize_svd(self, R):

        U, S, Vt = svds(
            R,
            k=self.factors
        )

        sqrt_s = np.sqrt(S)

        self.U = U @ np.diag(sqrt_s)

        self.V = (
            np.diag(sqrt_s)
            @ Vt
        ).T

    def fit(
        self,
        R,
        ratings_df
    ):

        self.bias_model.fit(ratings_df)

        self.initialize_svd(R)

        optimizer = ALSOptimizer(
            factors=self.factors,
            reg=self.reg
        )

        for itr in range(self.iterations):

            self.U = optimizer.update_users(
                R,
                self.U,
                self.V
            )

            self.V = optimizer.update_items(
                R,
                self.U,
                self.V
            )

            rmse = self.compute_train_rmse(R)

            self.history["train_rmse"].append(
                rmse
            )

            print(
                f"ALS Iter {itr+1}/{self.iterations} "
                f"| RMSE={rmse:.4f}"
            )

    def predict(
        self,
        user_idx,
        item_idx
    ):

        score = (
            self.U[user_idx]
            @ self.V[item_idx]
        )

        score += (
            self.bias_model.global_mean
            + self.bias_model.user_bias[user_idx]
            + self.bias_model.item_bias[item_idx]
        )

        return float(np.clip(score, 1, 5))

    def compute_train_rmse(
        self,
        R
    ):

        rows, cols = R.nonzero()

        preds = np.sum(
            self.U[rows] * self.V[cols],
            axis=1
        )

        truth = np.array(
            R[rows, cols]
        ).flatten()

        return np.sqrt(
            np.mean(
                (truth - preds) ** 2
            )
        )

    def recommend(
        self,
        user_idx,
        R,
        top_k=10
    ):

        scores = (
            self.U[user_idx]
            @ self.V.T
        )

        scores += (
            self.bias_model.global_mean
            + self.bias_model.user_bias[user_idx]
            + self.bias_model.item_bias
        )

        rated = R[user_idx].indices

        scores[rated] = -999

        top = np.argsort(scores)[::-1]

        return top[:top_k]

    def save(self, path="models"):

        np.save(
            f"{path}/U.npy",
            self.U
        )

        np.save(
            f"{path}/V.npy",
            self.V
        )

        np.save(
            f"{path}/user_bias.npy",
            self.bias_model.user_bias
        )

        np.save(
            f"{path}/item_bias.npy",
            self.bias_model.item_bias
        )

        np.save(
            f"{path}/global_mean.npy",
            np.array(
                [self.bias_model.global_mean]
            )
      )
