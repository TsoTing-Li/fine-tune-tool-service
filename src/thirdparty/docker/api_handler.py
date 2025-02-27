from collections.abc import AsyncGenerator
from typing import Literal, Union

import httpx
from fastapi import status

from src.utils.utils import generate_uuid


async def create_container(aclient: httpx.AsyncClient, name: str, data: dict) -> str:
    container_name_or_id = f"{name}-{generate_uuid()}"
    response = await aclient.post(
        "http://docker/containers/create",
        json=data,
        params={"name": container_name_or_id},
    )

    if response.status_code == status.HTTP_201_CREATED:
        return container_name_or_id
    else:
        raise RuntimeError(
            f"Create container error: {response.status_code}, {response.text}"
        )


async def start_container(aclient: httpx.AsyncClient, container_name_or_id: str) -> str:
    response = await aclient.post(
        f"http://docker/containers/{container_name_or_id}/start"
    )

    if response.status_code == status.HTTP_204_NO_CONTENT:
        return container_name_or_id
    else:
        raise RuntimeError(
            f"Startup container error: {response.status_code}, {response.text}"
        )


async def stop_container(
    aclient: httpx.AsyncClient,
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    response = await aclient.post(
        f"http://docker/containers/{container_name_or_id}/stop",
        params={"signal": signal, "t": wait_sec},
    )

    if response.status_code == status.HTTP_204_NO_CONTENT:
        return container_name_or_id
    elif response.status_code == status.HTTP_304_NOT_MODIFIED:
        return f"{container_name_or_id}, already stopped"
    else:
        raise RuntimeError(
            f"Stop container error: {response.status_code}, {response.text}"
        ) from None


async def get_container_log(
    aclient: httpx.AsyncClient, container_name_or_id: str, tail: Union[str, int] = "all"
) -> AsyncGenerator[str, None]:
    params = {"follow": True, "stdout": True, "stderr": True, "tail": str(tail)}
    async with aclient.stream(
        "GET", f"http://docker/containers/{container_name_or_id}/logs", params=params
    ) as response:
        if response.status_code == status.HTTP_200_OK:
            async for chunk in response.aiter_text():
                yield chunk
        elif response.status_code == status.HTTP_404_NOT_FOUND:
            raise ValueError(response.json()["message"])
        else:
            raise RuntimeError(response.json()["message"])
