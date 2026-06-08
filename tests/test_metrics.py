from src.metrics import (
    rmse,
    mae
)


def test_rmse():

    truth = [1, 2, 3]

    pred = [1, 2, 3]

    assert rmse(
        truth,
        pred
    ) == 0


def test_mae():

    truth = [1, 2, 3]

    pred = [1, 2, 3]

    assert mae(
        truth,
        pred
    ) == 0
