import os
from typing import Literal

import aiofiles
import httpx
import yaml

from inno_service.utils.docker_api_utils import (
    create_container,
    start_container,
    stop_container,
)


async def get_model_params(path: str) -> dict:
    try:
        async with aiofiles.open(path) as af:
            content = await af.read()

        return yaml.safe_load(content)

    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exists") from None


async def run_lm_eval(image_name: str, cmd: list, eval_name: str) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    hf_home = os.environ["HF_HOME"]
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
                f"{hf_home}:{hf_home}:rw",
                f"{root_path}/saves:{os.environ['WS']}/saves:rw",
            ],
            "NetworkMode": "host",
        },
        "Cmd": cmd,
        "Env": [f"HF_HOME={hf_home}"],
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
