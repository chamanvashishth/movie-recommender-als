import numpy as np

from scipy.sparse import csr_matrix

from src.matrix_factorization import (
    MatrixFactorizationALS
)


def test_training():

    ratings = csr_matrix(
        np.array(
            [
                [5, 4, 0],
                [4, 0, 3],
                [0, 5, 4]
            ]
        )
    )

    model = MatrixFactorizationALS(
        factors=2,
        iterations=2
    )

    import pandas as pd

    df = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 2, 3],
            "rating": [5, 4, 4, 3, 5, 4]
        }
    )

    model.fit(
        ratings,
        df
    )

    assert model.U is not None
