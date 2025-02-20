import json
import os
from typing import Literal

import httpx

from src.thirdparty.docker.api_handler import (
    create_container,
    start_container,
    stop_container,
)

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


async def start_ollama_container(image_name: str, model_name: str) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    root_path = os.environ["ROOT_PATH"]
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{root_path}/ollama:/root/.ollama:rw",
                f"{root_path}/saves:{SAVE_PATH}:rw",
            ],
            "NetworkMode": "host",
            "AutoRemove": True,
        },
        "Tty": True,
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        container_name_or_id = await create_container(
            aclient=aclient, name=f"ollama-{model_name}", data=data
        )

        started_container = await start_container(
            aclient=aclient, container_name_or_id=container_name_or_id
        )

        return started_container


async def run_ollama_model(ollama_url: str, model_name: str) -> None:
    async with httpx.AsyncClient() as aclient:
        async with aclient.stream(
            "POST",
            f"{ollama_url}/api/create",
            json={
                "model": model_name,
                "modelfile": f"FROM {SAVE_PATH}/{model_name}/quantize/{model_name}.gguf",
            },
        ) as response:
            if response.status_code == 200:
                async for chunk in response.aiter_lines():
                    if chunk:
                        data_chunk = json.loads(chunk)

                        if data_chunk.get("error"):
                            raise RuntimeError(f"{data_chunk['error']}") from None

                        if data_chunk["status"] == "success":
                            break
            else:
                raise RuntimeError(f"{response.text}") from None

        response = await aclient.post(
            f"{ollama_url}/api/generate", json={"model": model_name, "keep_alive": -1}
        )
        if response.status_code != 200:
            raise RuntimeError(f"{response.text}")


async def stop_ollama_container(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport) as aclient:
        stopped_container = await stop_container(
            aclient=aclient,
            container_name_or_id=container_name_or_id,
            signal=signal,
            wait_sec=wait_sec,
        )

        return stopped_container
