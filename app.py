import streamlit as st
import pandas as pd
import numpy as np

from src.cold_start import (
    ColdStartRecommender
)

from src.explainability import (
    RecommendationExplainer
)

from src.matrix_factorization import (
    MatrixFactorizationALS
)


st.set_page_config(
    page_title=
    "MovieLens Recommender — Matrix Factorization",
    layout="wide"
)


@st.cache_resource
def load_model():

    model = MatrixFactorizationALS()

    model.U = np.load(
        "models/U.npy"
    )

    model.V = np.load(
        "models/V.npy"
    )

    model.bias_model.user_bias = np.load(
        "models/user_bias.npy"
    )

    model.bias_model.item_bias = np.load(
        "models/item_bias.npy"
    )

    model.bias_model.global_mean = np.load(
        "models/global_mean.npy"
    )[0]

    return model


@st.cache_data
def load_data():

    train_df = pd.read_csv(
        "models/train.csv"
    )

    movies = pd.read_csv(
        "models/movies.csv"
    )

    return train_df, movies


model = load_model()

train_df, movies = load_data()

st.title(
    "MovieLens Recommender — Matrix Factorization"
)

st.sidebar.header(
    "Controls"
)

new_user = st.sidebar.checkbox(
    "I'm a new user"
)

if not new_user:

    user_id = st.sidebar.slider(
        "User ID",
        1,
        6040,
        1
    )

else:

    genres = sorted(
        list(
            {
                g
                for row in movies.genres
                for g in row.split("|")
            }
        )
    )

    selected = st.sidebar.multiselect(
        "Favorite Genres",
        genres
    )

if st.sidebar.button(
    "Get Recommendations"
):

    if not new_user:

        user_idx = user_id - 1

        recs = model.recommend(
            user_idx,
            top_k=10,
            R=None
        )

        st.subheader(
            f"Top Recommendations for User #{user_id}"
        )

        for movie_idx in recs:

            movie_id = movie_idx + 1

            movie = movies[
                movies.movieId == movie_id
            ]

            if len(movie) == 0:
                continue

            movie = movie.iloc[0]

            rating = model.predict(
                user_idx,
                movie_idx
            )

            stars = (
                "★" *
                round(rating)
            )

            st.markdown(
                f"""
                ### {movie.title}
                Genres: {movie.genres}

                Rating:
                {stars} ({rating:.2f}/5)
                """
            )

    else:

        cold = ColdStartRecommender()

        cold.fit(
            movies,
            train_df
        )

        recs = cold.recommend(
            selected,
            top_k=10
        )

        st.subheader(
            "Recommendations"
        )

        st.dataframe(
            recs[
                [
                    "title",
                    "genres",
                    "avg_rating"
                ]
            ]
      )
