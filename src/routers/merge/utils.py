from typing import Literal

import httpx

from src.config.params import (
    COMMON_CONFIG,
)
from src.thirdparty.docker.api_handler import (
    create_container,
    remove_container,
    start_container,
    stop_container,
)
from src.utils.logger import accel_logger


async def run_merge(
    image_name: str,
    cmd: list,
    docker_network_name: str,
    merge_name: str,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    env_var = [f"HF_HOME={COMMON_CONFIG.hf_home}"]
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{COMMON_CONFIG.hf_home}:{COMMON_CONFIG.hf_home}:rw",
                f"{COMMON_CONFIG.root_path}/saves/{merge_name}:{COMMON_CONFIG.save_path}/{merge_name}:rw",
            ],
            "NetworkMode": docker_network_name,
            "AutoRemove": False,
        },
        "Cmd": cmd,
        "Env": env_var,
    }

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_name_or_id = await create_container(
                aclient=aclient, name=f"merge-{merge_name}", data=data
            )
            started_container = await start_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

        return started_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None


async def stop_merge(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            stopped_container = await stop_container(
                aclient=aclient,
                container_name_or_id=container_name_or_id,
                signal=signal,
                wait_sec=wait_sec,
            )

            await remove_container(
                aclient=aclient, container_name_or_id=stopped_container
            )

        return stopped_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None
