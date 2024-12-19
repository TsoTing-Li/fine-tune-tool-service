import os
from typing import Literal

import aiofiles
import httpx
import yaml

from inno_service.utils.utils import generate_uuid


def basemodel2dict(data) -> dict:
    train_args = {
        "model_name_or_path": data.model_name_or_path,
        **data.method.model_dump(),
        **data.dataset.model_dump(),
        **data.output.model_dump(),
        **data.params.model_dump(),
        **data.eval.model_dump(),
    }

    return train_args


async def write_train_yaml_to_two_path(train_config_path: str, path: str, data: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=False)
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except FileExistsError:
        raise FileExistsError(f"'{path}' is already exists") from None

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
                f"{root_path}/saves/{train_name}:/app/saves/{train_name}:rw",
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
            response = await aclient.post(
                f"http://docker/containers/{container_name}/start"
            )

            if response.status_code == 204:
                print("Container started")
                return container_name

            else:
                print(f"Container started failed: {response.status_code}")
                raise RuntimeError(
                    f"Error: {response.status_code}, {response.text}"
                ) from None

        else:
            print(f"{response.status_code}\n{response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None


async def stop_train(
    base_url: str,
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
):
    params = {"signal": signal, "t": wait_sec}

    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(
            f"{base_url}/{container_name_or_id}/stop", params=params
        )

        if response.status_code == 204:
            print("Container stop")
        else:
            print(f"Container stop failed: {response.status_code}\n{response.text}")
