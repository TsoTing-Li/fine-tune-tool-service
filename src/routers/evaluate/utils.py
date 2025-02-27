from typing import Literal

import httpx

from src.config import params
from src.thirdparty.docker.api_handler import (
    create_container,
    start_container,
    stop_container,
)


async def run_lm_eval(image_name: str, cmd: list, eval_name: str) -> str:
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
                f"{params.COMMON_CONFIG.hf_home}:{params.COMMON_CONFIG.hf_home}:rw",
                f"{params.COMMON_CONFIG.root_path}/saves:{params.COMMON_CONFIG.save_path}:rw",
            ],
            "NetworkMode": "host",
        },
        "Cmd": cmd,
        "Env": [f"HF_HOME={params.COMMON_CONFIG.hf_home}"],
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        container_name_or_id = await create_container(
            aclient=aclient, name=f"eval-{eval_name}", data=data
        )

        started_container = await start_container(
            aclient=aclient, container_name_or_id=container_name_or_id
        )

        return started_container


async def stop_eval(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        stopped_eval_container = await stop_container(
            aclient=aclient,
            container_name_or_id=container_name_or_id,
            signal=signal,
            wait_sec=wait_sec,
        )

    return stopped_eval_container
