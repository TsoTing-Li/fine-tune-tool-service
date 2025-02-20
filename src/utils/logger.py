import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_LOGGER = "uvicorn.error"


def setup_logger(folder: str, name: str, limit: int, count: int):
    # Get uvicorn logger
    logger = logging.getLogger(BASE_LOGGER)

    # Create Logs Folder
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(exist_ok=True, parents=True)

    # Combine Path
    file_handler = RotatingFileHandler(
        folder / name,
        mode="a",
        encoding="utf-8",
        maxBytes=int(limit),
        backupCount=int(count),
        delay=0,
    )

    # Define Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-.4s] %(message)s (%(filename)s:%(lineno)s)",
        "%y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)
    return logger


accel_logger = setup_logger(
    folder=os.environ["ACCELTUNE_LOG_FOLDER"],
    name=os.environ["ACCELTUNE_LOG_NAME"],
    limit=os.environ["ACCELTUNE_LOG_LIMIT"],
    count=os.environ["ACCELTUNE_LOG_COUNT"],
)
