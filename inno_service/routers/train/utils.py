import asyncio
import os
import shutil
from typing import Literal, Union

import aiofiles
import aiofiles.os
import httpx
import yaml
from fastapi import HTTPException, UploadFile

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


async def call_ds_api(
    name: str, ds_args: dict, ds_file: Union[UploadFile, None] = None
) -> str:
    base_url = f"http://127.0.0.1:{os.getenv('MAIN_SERVICE_PORT')}/acceltune/deepspeed"
    async with httpx.AsyncClient(timeout=None) as aclient:
        if ds_args["src"] == "default":
            payload = {
                "json": {
                    "name": name,
                    "stage": ds_args["stage"],
                    "enable_offload": ds_args["enable_offload"],
                    "offload_device": ds_args["offload_device"],
                }
            }
        elif ds_args["src"] == "file":
            payload = {
                "files": {
                    "ds_file": (
                        ds_file.filename,
                        await ds_file.read(),
                        ds_file.content_type,
                    ),
                    "name": (None, name),
                }
            }

        response = await aclient.post(f"{base_url}/{ds_args['src']}/", **payload)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()


async def write_yaml(path: str, data: dict) -> None:
    try:
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except Exception as e:
        raise OSError(f"Unexpected error: {e}") from None


async def get_yaml_content(path: str) -> dict:
    async with aiofiles.open(path) as af:
        content = await af.read()

    return yaml.safe_load(content)


def find_all_yaml(base_path: str) -> list:
    yaml_files = list()

    for task in os.listdir(base_path):
        yaml_path = os.path.join(base_path, task, f"{task}.yaml")
        if os.path.isfile(yaml_path):
            yaml_files.append(yaml_path)

    return yaml_files


async def get_train_args(train_name: str) -> list:
    if not train_name:
        yaml_files = find_all_yaml(base_path=SAVE_PATH)
    else:
        yaml_files = [os.path.join(SAVE_PATH, train_name, f"{train_name}.yaml")]

    get_schedule = [get_yaml_content(path=yaml_path) for yaml_path in yaml_files]
    file_contents = await asyncio.gather(*get_schedule)

    train_args_info = {
        os.path.splitext(os.path.basename(yaml_path))[0]: content
        for yaml_path, content in zip(yaml_files, file_contents)
    }

    return train_args_info


def del_train(path: str) -> str:
    shutil.rmtree(path)
    return path


async def async_clear_exists_path(train_name: str) -> None:
    train_args = await get_yaml_content(f"{SAVE_PATH}/{train_name}/{train_name}.yaml")
    is_exists = await aiofiles.os.path.exists(train_args["output_dir"])
    if is_exists:
        await asyncio.to_thread(shutil.rmtree, train_args["output_dir"])


async def run_train(image_name: str, cmd: list, train_name: str) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    hf_home = os.environ["HF_HOME"]
    root_path = os.environ["ROOT_PATH"]
    container_name = f"train-{train_name}-{generate_uuid()}"
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
