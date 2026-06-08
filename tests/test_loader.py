from src.data_loader import (
    MovieLensLoader
)


def test_loader_init():

    loader = MovieLensLoader()

    assert loader is not None
