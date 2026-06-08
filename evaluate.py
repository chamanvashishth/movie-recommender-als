import numpy as np
import pandas as pd

from src.metrics import (
    rmse,
    mae,
    compute_ndcg,
    hit_rate_at_k,
    catalog_coverage
)

from src.matrix_factorization import (
    MatrixFactorizationALS
)


def main():

    train_df = pd.read_csv(
        "models/train.csv"
    )

    test_df = pd.read_csv(
        "models/test.csv"
    )

    n_users = (
        train_df.userId.max()
    )

    n_items = (
        train_df.movieId.max()
    )

    model = MatrixFactorizationALS()

    model.U = np.load(
        "models/U.npy"
    )

    model.V = np.load(
        "models/V.npy"
    )

    user_bias = np.load(
        "models/user_bias.npy"
    )

    item_bias = np.load(
        "models/item_bias.npy"
    )

    global_mean = np.load(
        "models/global_mean.npy"
    )[0]

    model.bias_model.user_bias = (
        user_bias
    )

    model.bias_model.item_bias = (
        item_bias
    )

    model.bias_model.global_mean = (
        global_mean
    )

    from scipy.sparse import csr_matrix

    rows = (
        train_df.userId.values - 1
    )

    cols = (
        train_df.movieId.values - 1
    )

    values = (
        train_df.rating.values
    )

    train_matrix = csr_matrix(
        (
            values,
            (rows, cols)
        ),
        shape=(n_users, n_items)
    )

    predictions = []
    targets = []

    for row in test_df.itertuples():

        pred = model.predict(
            row.userId - 1,
            row.movieId - 1
        )

        predictions.append(pred)
        targets.append(row.rating)

    rmse_score = rmse(
        targets,
        predictions
    )

    mae_score = mae(
        targets,
        predictions
    )

    ndcg_score = compute_ndcg(
        model,
        test_df,
        train_matrix,
        k=10
    )

    hit_rate = hit_rate_at_k(
        model,
        test_df,
        train_matrix,
        k=10
    )

    coverage = catalog_coverage(
        model,
        train_matrix,
        n_items,
        k=10
    )

    print("\nRESULTS")
    print("=" * 40)

    print(
        f"RMSE      : "
        f"{rmse_score:.4f}"
    )

    print(
        f"MAE       : "
        f"{mae_score:.4f}"
    )

    print(
        f"NDCG@10   : "
        f"{ndcg_score:.4f}"
    )

    print(
        f"HitRate10 : "
        f"{hit_rate:.4f}"
    )

    print(
        f"Coverage  : "
        f"{coverage:.4f}"
    )


if __name__ == "__main__":
    main()
