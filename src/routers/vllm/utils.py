import asyncio
import time
from typing import Literal

import httpx

from src.thirdparty.docker.api_handler import (
    create_container,
    start_container,
    stop_container,
)


async def start_vllm_container(
    image_name: str,
    service_port: int,
    docker_network_name: str,
    cmd: list,
    model_name: str,
    local_safetensors_path: str,
    hf_home: str,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{hf_home}:{hf_home}:rw",
                f"{local_safetensors_path}:{local_safetensors_path}:rw",
            ],
            "PortBindings": {f"{service_port}/tcp": [{"HostPort": f"{service_port}"}]},
            "AutoRemove": True,
            "NetworkMode": docker_network_name,
        },
        "Cmd": cmd,
        "Env": [f"HF_HOME={hf_home}"],
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        container_name_or_id = await create_container(
            aclient=aclient, name=f"vllm-{model_name}", data=data
        )

        started_container = await start_container(
            aclient=aclient, container_name_or_id=container_name_or_id
        )

        return started_container


async def health_check(
    aclient: httpx.AsyncClient,
    health_check_url: str,
    timeout: int = 30,
    interval: int = 2,
) -> bool:
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = await aclient.get(health_check_url)
            if response.status_code == 200:
                return True

        except httpx.RequestError:
            pass

        await asyncio.sleep(interval)

    return False


async def run_vllm_model(vllm_url: str) -> None:
    async with httpx.AsyncClient() as aclient:
        is_loaded = await health_check(
            aclient=aclient, health_check_url=f"{vllm_url}/health"
        )

        if is_loaded:
            return
        else:
            raise RuntimeError("model loading failed")


async def stop_vllm_container(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        stopped_container = await stop_container(
            aclient=aclient,
            container_name_or_id=container_name_or_id,
            signal=signal,
            wait_sec=wait_sec,
        )

        return stopped_container
