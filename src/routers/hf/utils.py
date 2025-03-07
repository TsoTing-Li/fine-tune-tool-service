import os
from typing import Union

import httpx
from fastapi import status


def add_token(token: str) -> str:
    os.environ["HF_TOKEN"] = token
    return token


def get_token() -> Union[str, None]:
    return os.getenv("HF_TOKEN", None)


async def call_hf_whoami(hf_token: str) -> dict:
    async with httpx.AsyncClient() as aclient:
        response = await aclient.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {hf_token}"},
        )

        if response.status_code == status.HTTP_200_OK:
            return response.json()
        elif response.status_code == status.HTTP_401_UNAUTHORIZED:
            raise ValueError(response.json()["error"])
        else:
            raise RuntimeError(response.json()["error"])
