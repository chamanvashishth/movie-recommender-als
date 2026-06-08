import numpy as np
from collections import defaultdict


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def dcg_at_k(relevance, k=10):
    relevance = np.asarray(relevance)[:k]

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevance) + 2))

    return np.sum(relevance / discounts)


def ndcg_at_k(relevance, k=10):
    actual = dcg_at_k(relevance, k)

    ideal = dcg_at_k(
        sorted(relevance, reverse=True),
        k
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def compute_ndcg(
    model,
    test_df,
    train_matrix,
    k=10,
    relevance_threshold=4.0
):

    user_groups = test_df.groupby("userId")

    ndcgs = []

    for user_id, group in user_groups:

        user_idx = user_id - 1

        recommendations = model.recommend(
            user_idx=user_idx,
            R=train_matrix,
            top_k=k
        )

        test_items = {
            row.movieId - 1: row.rating
            for row in group.itertuples()
        }

        relevance = []

        for item in recommendations:

            if item in test_items:
                relevance.append(
                    int(
                        test_items[item]
                        >= relevance_threshold
                    )
                )
            else:
                relevance.append(0)

        ndcgs.append(
            ndcg_at_k(relevance, k)
        )

    return float(np.mean(ndcgs))


def hit_rate_at_k(
    model,
    test_df,
    train_matrix,
    k=10,
    relevance_threshold=4.0
):

    hits = 0
    total = 0

    user_groups = test_df.groupby("userId")

    for user_id, group in user_groups:

        total += 1

        user_idx = user_id - 1

        recommendations = set(
            model.recommend(
                user_idx,
                train_matrix,
                top_k=k
            )
        )

        relevant_items = set(
            (
                group[
                    group.rating >= relevance_threshold
                ].movieId.values - 1
            )
        )

        if len(
            recommendations.intersection(
                relevant_items
            )
        ) > 0:
            hits += 1

    return hits / total


def catalog_coverage(
    model,
    train_matrix,
    num_items,
    k=10
):

    recommended_items = set()

    for user_idx in range(
        train_matrix.shape[0]
    ):

        recs = model.recommend(
            user_idx,
            train_matrix,
            top_k=k
        )

        recommended_items.update(recs)

    return (
        len(recommended_items)
        / num_items
    )
