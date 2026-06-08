import os

from src.data_loader import (
    MovieLensLoader
)

from src.matrix_factorization import (
    MatrixFactorizationALS
)


def main():

    os.makedirs(
        "models",
        exist_ok=True
    )

    loader = MovieLensLoader()

    loader.download()

    ratings = loader.load_ratings()

    movies = loader.load_movies()

    train_df, test_df = (
        loader.train_test_split(
            ratings
        )
    )

    train_matrix = (
        loader.build_sparse_matrix(
            train_df
        )
    )

    print(
        f"Train ratings: "
        f"{len(train_df):,}"
    )

    print(
        f"Test ratings: "
        f"{len(test_df):,}"
    )

    model = MatrixFactorizationALS(
        factors=20,
        reg=0.01,
        iterations=20
    )

    model.fit(
        train_matrix,
        train_df
    )

    model.save("models")

    train_df.to_csv(
        "models/train.csv",
        index=False
    )

    test_df.to_csv(
        "models/test.csv",
        index=False
    )

    movies.to_csv(
        "models/movies.csv",
        index=False
    )

    print(
        "\nTraining complete."
    )


if __name__ == "__main__":
    main()
