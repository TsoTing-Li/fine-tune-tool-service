import os


def add_token(token: str) -> str:
    os.environ["HF_TOKEN"] = token
    return token
