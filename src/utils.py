import os
import random
import numpy as np


def seed_everything(
    seed=42
):

    random.seed(seed)

    np.random.seed(seed)


def ensure_dirs():

    directories = [

        "models",

        "results",

        "logs",

        "data"

    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )
