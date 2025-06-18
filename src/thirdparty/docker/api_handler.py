import json
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
    elif response.status_code == status.HTTP_400_BAD_REQUEST:
        raise KeyError(response.json()["message"])
    elif response.status_code in {status.HTTP_404_NOT_FOUND, status.HTTP_409_CONFLICT}:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


async def start_container(aclient: httpx.AsyncClient, container_name_or_id: str) -> str:
    response = await aclient.post(
        f"http://docker/containers/{container_name_or_id}/start"
    )

    if response.status_code == status.HTTP_204_NO_CONTENT:
        return container_name_or_id
    elif response.status_code == status.HTTP_304_NOT_MODIFIED:
        return f"{container_name_or_id}, already started"
    elif response.status_code == status.HTTP_404_NOT_FOUND:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


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
    elif response.status_code == status.HTTP_404_NOT_FOUND:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


async def get_container_log(
    aclient: httpx.AsyncClient,
    container_name_or_id: str,
    follow: bool = True,
    stdout: bool = True,
    stderr: bool = True,
    tail: Union[str, int] = "all",
) -> AsyncGenerator[str, None]:
    params = {"follow": follow, "stdout": stdout, "stderr": stderr, "tail": str(tail)}
    async with aclient.stream(
        "GET", f"http://docker/containers/{container_name_or_id}/logs", params=params
    ) as response:
        if response.status_code == status.HTTP_200_OK:
            async for chunk in response.aiter_text():
                yield chunk
        elif response.status_code == status.HTTP_404_NOT_FOUND:
            async for chunk in response.aiter_lines():
                error_msg = json.loads(chunk)
            raise ValueError(error_msg["message"])
        else:
            async for chunk in response.aiter_lines():
                error_msg = json.loads(chunk)
            raise RuntimeError(error_msg["message"])


async def get_container_info(aclient: httpx.AsyncClient, container_name: str) -> dict:
    params = {"all": True, "filters": json.dumps({"name": [container_name]})}

    response = await aclient.get("http://docker/containers/json", params=params)

    if response.status_code == status.HTTP_200_OK:
        return response.json()[0] if response.json() else dict()
    elif response.status_code == status.HTTP_400_BAD_REQUEST:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


async def wait_for_container(aclient: httpx.AsyncClient, container_name: str) -> dict:
    response = await aclient.post(f"http://docker/containers/{container_name}/wait")

    if response.status_code == status.HTTP_200_OK:
        return response.json()
    elif response.status_code in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


async def remove_container(
    aclient: httpx.AsyncClient, container_name_or_id: str
) -> None:
    response = await aclient.delete(f"http://docker/containers/{container_name_or_id}")

    if response.status_code == status.HTTP_204_NO_CONTENT:
        return
    elif response.status_code == status.HTTP_400_BAD_REQUEST:
        raise ValueError(response.json()["message"])
    elif response.status_code == status.HTTP_404_NOT_FOUND:
        raise ValueError(response.json()["message"])
    elif response.status_code == status.HTTP_409_CONFLICT:
        raise ValueError(response.json()["message"])
    else:
        raise RuntimeError(response.json()["message"])


async def attach_container(
    aclient: httpx.AsyncClient,
    container_name_or_id: str,
    stream: bool = True,
    stdout: bool = True,
    stderr: bool = True,
    logs: bool = True,
) -> AsyncGenerator[str, None]:
    params = {"stream": stream, "stdout": stdout, "stderr": stderr, "logs": logs}
    async with aclient.stream(
        "POST", f"http://docker/containers/{container_name_or_id}/attach", params=params
    ) as response:
        if response.status_code == status.HTTP_200_OK:
            async for chunk in response.aiter_text():
                yield chunk
        elif response.status_code == status.HTTP_400_BAD_REQUEST:
            async for chunk in response.aiter_lines():
                error_msg = json.loads(chunk)
            raise ValueError(error_msg["message"])
        elif response.status_code == status.HTTP_404_NOT_FOUND:
            async for chunk in response.aiter_lines():
                error_msg = json.loads(chunk)
            raise ValueError(error_msg["message"])
        else:
            async for chunk in response.aiter_lines():
                error_msg = json.loads(chunk)
            raise RuntimeError(error_msg["message"])
