import os
import streamlit as st
import numpy as np
import pandas as pd

from src.cold_start import ColdStartRecommender
from src.data_loader import MovieLensLoader
from src.matrix_factorization import MatrixFactorizationALS

st.set_page_config(
    page_title="MovieLens Recommender — Matrix Factorization",
    layout="wide",
)

MODEL_DIR = "models"
REQUIRED_MODEL_FILES = [
    "U.npy",
    "V.npy",
    "user_bias.npy",
    "item_bias.npy",
    "global_mean.npy",
]


def model_files_available():
    return all(
        os.path.exists(os.path.join(MODEL_DIR, name))
        for name in REQUIRED_MODEL_FILES
    )


@st.cache_data(show_spinner="Preparing MovieLens data...")
def load_data():
    loader = MovieLensLoader()
    loader.download()

    ratings = loader.load_ratings()
    movies = loader.load_movies()
    train_df, _ = loader.train_test_split(ratings)

    return train_df, movies


@st.cache_resource
def load_model():
    if not model_files_available():
        return None

    model = MatrixFactorizationALS()
    model.U = np.load(os.path.join(MODEL_DIR, "U.npy"))
    model.V = np.load(os.path.join(MODEL_DIR, "V.npy"))
    model.bias_model.user_bias = np.load(
        os.path.join(MODEL_DIR, "user_bias.npy")
    )
    model.bias_model.item_bias = np.load(
        os.path.join(MODEL_DIR, "item_bias.npy")
    )
    model.bias_model.global_mean = np.load(
        os.path.join(MODEL_DIR, "global_mean.npy")
    )[0]

    return model


train_df, movies = load_data()
model = load_model()

st.title("MovieLens Recommender")
st.caption(
    "Movie recommendations using ALS matrix factorization when trained "
    "artifacts are available, with a cold-start recommender fallback."
)

if model is None:
    st.info(
        "Trained ALS artifacts are not included in this deployment, so the "
        "app is running in cold-start mode using MovieLens ratings and genres."
    )

st.sidebar.header("Controls")
new_user = st.sidebar.checkbox("I'm a new user", value=True)

if model is not None and not new_user:
    max_user = min(len(model.U), 6040)
    user_id = st.sidebar.slider("User ID", 1, max_user, 1)
else:
    new_user = True
    genres = sorted(
        {
            genre
            for row in movies["genres"].dropna()
            for genre in row.split("|")
        }
    )
    selected = st.sidebar.multiselect("Favorite Genres", genres)

if st.sidebar.button("Get Recommendations"):
    if not new_user and model is not None:
        loader = MovieLensLoader()
        train_matrix = loader.build_sparse_matrix(train_df)
        user_idx = user_id - 1
        recs = model.recommend(user_idx, top_k=10, R=train_matrix)

        st.subheader(f"Top Recommendations for User #{user_id}")
        for movie_idx in recs:
            movie_id = movie_idx + 1
            match = movies[movies.movieId == movie_id]
            if match.empty:
                continue

            movie = match.iloc[0]
            rating = model.predict(user_idx, movie_idx)
            stars = "★" * max(1, round(rating))
            st.markdown(
                f"### {movie.title}\n"
                f"**Genres:** {movie.genres}\n\n"
                f"**Predicted rating:** {stars} ({rating:.2f}/5)"
            )
    else:
        cold = ColdStartRecommender()
        cold.fit(movies, train_df)

        if not selected:
            st.warning("Select at least one favorite genre.")
        else:
            recs = cold.recommend(selected, top_k=10)
            st.subheader("Recommendations")
            st.dataframe(
                recs[["title", "genres", "avg_rating", "rating_count"]],
                use_container_width=True,
                hide_index=True,
            )
