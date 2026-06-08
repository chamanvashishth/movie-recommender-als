import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix

os.makedirs(
    "results",
    exist_ok=True
)


def plot_rating_distribution(ratings):

    plt.figure(figsize=(8, 5))

    ratings["rating"].value_counts() \
        .sort_index() \
        .plot(kind="bar")

    plt.title(
        "Rating Distribution"
    )

    plt.xlabel("Rating")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        "results/rating_distribution.png"
    )

    plt.close()


def plot_sparsity(matrix):

    sample = matrix[
        :500,
        :500
    ].toarray()

    sparsity = (
        1 -
        np.count_nonzero(sample)
        / sample.size
    ) * 100

    plt.figure(figsize=(8, 8))

    plt.imshow(
        sample > 0,
        aspect="auto"
    )

    plt.title(
        f"Sparsity {sparsity:.2f}%"
    )

    plt.tight_layout()

    plt.savefig(
        "results/user_item_sparsity.png"
    )

    plt.close()


def plot_als_convergence(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_rmse"],
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("RMSE")

    plt.title(
        "ALS Convergence"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "results/als_convergence.png"
    )

    plt.close()


def plot_latent_space(
    V,
    movies
):

    pca = PCA(
        n_components=2,
        random_state=42
    )

    coords = pca.fit_transform(V)

    plt.figure(
        figsize=(12, 10)
    )

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        alpha=0.5
    )

    plt.title(
        "Movie Latent Space"
    )

    plt.tight_layout()

    plt.savefig(
        "results/latent_factor_visualization.png"
    )

    plt.close()
