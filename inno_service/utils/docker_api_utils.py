from typing import Literal

import httpx
from utils import generate_uuid


async def create_container(aclient: httpx.AsyncClient, name: str, data: dict) -> str:
    container_name_or_id = f"{name}-{generate_uuid()}"
    response = await aclient.post(
        "http://docker/containers/create",
        json=data,
        params={"name": container_name_or_id},
    )

    if response.status_code == 201:
        return container_name_or_id
    else:
        raise RuntimeError(f"Error: {response.status_code}, {response.text}") from None


async def start_container(aclient: httpx.AsyncClient, container_name_or_id: str) -> str:
    response = await aclient.post(
        f"http://docker/containers/{container_name_or_id}/start"
    )

    if response.status_code == 204:
        return container_name_or_id
    else:
        raise RuntimeError(f"Error: {response.status_code}, {response.text}") from None


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

    if response.status_code == 204:
        return container_name_or_id
    else:
        raise RuntimeError(f"Error: {response.status_code}, {response.text}") from None
