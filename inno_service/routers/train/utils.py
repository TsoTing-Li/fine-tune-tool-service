import os
from typing import Literal

import aiofiles
import httpx
import yaml

from inno_service.utils.logger import accel_logger
from inno_service.utils.utils import generate_uuid

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


def basemodel2dict(data) -> dict:
    train_args = {
        "model_name_or_path": data.model_name_or_path,
        **data.method.model_dump(),
        **data.dataset.model_dump(),
        **data.output.model_dump(),
        **data.params.model_dump(),
        **data.val.model_dump(),
    }

    return train_args


def add_train_path(path: str) -> str:
    os.makedirs(path, exist_ok=False)
    return path


async def write_yaml(path: str, data: dict) -> None:
    try:
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except Exception as e:
        raise OSError(f"Unexpected error: {e}") from None


async def run_train(image_name: str, cmd: list, train_name: str) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    hf_home = os.environ["HF_HOME"]
    root_path = os.environ["ROOT_PATH"]
    container_name = f"{train_name}-{generate_uuid()}"
    params = {"name": container_name}
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
                f"{root_path}/data:/app/data:rw",
                f"{root_path}/saves/{train_name}:{SAVE_PATH}/{train_name}:rw",
            ],
        },
        "Cmd": cmd,
        "Env": [f"HF_HOME={hf_home}"],
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(
            "http://docker/containers/create", json=data, params=params
        )

        if response.status_code == 201:
            accel_logger.info(f"Fine-tune created, container: {container_name}")

            response = await aclient.post(
                f"http://docker/containers/{container_name}/start"
            )

            if response.status_code == 204:
                accel_logger.info(f"Fine-tune started, container: {container_name}")
                return container_name

            else:
                accel_logger.info(
                    f"Fine-tune startup failed, container: {container_name}"
                )
                accel_logger.info(f"Error: {response.status_code}, {response.text}")
                raise RuntimeError(
                    f"Error: {response.status_code}, {response.text}"
                ) from None

        else:
            accel_logger.info(f"Fine-tune creation failed, container: {container_name}")
            accel_logger.info(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None


async def stop_train(
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
) -> str:
    params = {"signal": signal, "t": wait_sec}

    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(
            f"http://docker/containers/{container_name_or_id}/stop", params=params
        )

        if response.status_code == 204:
            accel_logger.info(f"Fine-tune stopped, container: {container_name_or_id}")
            return container_name_or_id
        else:
            accel_logger.info(
                f"Fine-tune stop failed, container: {container_name_or_id}"
            )
            accel_logger.info(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None
