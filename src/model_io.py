import os
import numpy as np


class ModelIO:

    @staticmethod
    def save(model, path):

        os.makedirs(
            path,
            exist_ok=True
        )

        np.save(
            f"{path}/U.npy",
            model.U
        )

        np.save(
            f"{path}/V.npy",
            model.V
        )

        np.save(
            f"{path}/user_bias.npy",
            model.bias_model.user_bias
        )

        np.save(
            f"{path}/item_bias.npy",
            model.bias_model.item_bias
        )

        np.save(
            f"{path}/global_mean.npy",
            np.array(
                [
                    model.bias_model.global_mean
                ]
            )
        )

    @staticmethod
    def load(model, path):

        model.U = np.load(
            f"{path}/U.npy"
        )

        model.V = np.load(
            f"{path}/V.npy"
        )

        model.bias_model.user_bias = (
            np.load(
                f"{path}/user_bias.npy"
            )
        )

        model.bias_model.item_bias = (
            np.load(
                f"{path}/item_bias.npy"
            )
        )

        model.bias_model.global_mean = (
            np.load(
                f"{path}/global_mean.npy"
            )[0]
        )

        return model
