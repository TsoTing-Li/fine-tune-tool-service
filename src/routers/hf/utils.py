import os
from typing import Union


def add_token(token: str) -> str:
    os.environ["HF_TOKEN"] = token
    return token


def get_token() -> Union[str, None]:
    return os.getenv("HF_TOKEN", None)
