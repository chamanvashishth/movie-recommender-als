import logging
import os


def get_logger(name):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        )

        file_handler = logging.FileHandler(
            "logs/training.log"
        )

        file_handler.setFormatter(
            formatter
        )

        stream_handler = (
            logging.StreamHandler()
        )

        stream_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

        logger.addHandler(
            stream_handler
        )

    return logger
