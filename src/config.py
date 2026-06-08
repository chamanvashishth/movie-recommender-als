from dataclasses import dataclass


@dataclass
class Config:

    DATA_DIR = "data"

    MODEL_DIR = "models"

    RESULTS_DIR = "results"

    RANDOM_STATE = 42

    FACTORS = 20

    ALS_ITERATIONS = 20

    REGULARIZATION = 0.01

    TOP_K = 10

    TEST_SIZE = 0.2
