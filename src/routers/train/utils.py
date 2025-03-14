import asyncio
import os
import re
import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import List, Literal, Union

import aiofiles
import aiofiles.os
import httpx
import orjson
import yaml
from fastapi import HTTPException, UploadFile, status

from src.config.params import (
    COMMON_CONFIG,
    MAINSERVICE_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
from src.thirdparty.docker.api_handler import (
    create_container,
    get_container_log,
    remove_container,
    start_container,
    stop_container,
    wait_for_container,
)
from src.thirdparty.redis.handler import redis_async
from src.utils.logger import accel_logger


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
    base_url = f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/deepspeed"
    async with httpx.AsyncClient(timeout=None) as aclient:
        if ds_args["src"] == "default":
            payload = {
                "json": {
                    "train_name": name,
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
                    "train_name": (None, name),
                }
            }

        response = await aclient.post(f"{base_url}/{ds_args['src']}/", **payload)

        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )

        return response.json()


async def write_yaml(path: str, data: dict) -> None:
    try:
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except Exception as e:
        raise OSError(f"Unexpected error: {e}") from None


async def del_train(path: str) -> None:
    await asyncio.to_thread(shutil.rmtree, path)


async def async_clear_last_checkpoint(train_path: str) -> None:
    for method in {"full", "lora"}:
        checkpoint_path = os.path.join(train_path, method)
        is_exists = await aiofiles.os.path.exists(checkpoint_path)

        if is_exists:
            if not await aiofiles.os.path.isdir(checkpoint_path):
                return

            for item in await asyncio.to_thread(os.listdir, checkpoint_path):
                item_path = os.path.join(checkpoint_path, item)

                if await aiofiles.os.path.isfile(item_path):
                    await aiofiles.os.remove(item_path)
                elif await aiofiles.os.path.isdir(item_path):
                    await asyncio.to_thread(shutil.rmtree, item_path)


async def async_clear_file(paths: List[str]) -> None:
    async def delete_file(file_path: str) -> None:
        is_exists = await aiofiles.os.path.exists(file_path)
        if is_exists:
            await aiofiles.os.remove(file_path)

    await asyncio.gather(*(delete_file(path) for path in paths))


@asynccontextmanager
async def record_train_log(
    log_path: str,
) -> AsyncGenerator[aiofiles.threadpool.text.AsyncTextIndirectIOWrapper, None]:
    file = await aiofiles.open(log_path, "w")
    try:
        yield file
    finally:
        await file.close()


async def monitor_train_status(train_name: str, container_name_or_id: str):
    ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    try:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            async with record_train_log(
                log_path=os.path.join(COMMON_CONFIG.save_path, train_name, "train.log")
            ) as log_file:  # write all training log into file
                async for log in get_container_log(
                    aclient=aclient, container_name_or_id=container_name_or_id
                ):
                    for log_split in log.splitlines():
                        if log_split == "":
                            break
                        elif log_split[0] in ("\x01", "\x02"):
                            log_split = log_split[8:]

                        log_file: aiofiles.threadpool.text.AsyncTextIndirectIOWrapper
                        await log_file.write(f"{ANSI_ESCAPE.sub('', log_split)}\n")

            container_info = await wait_for_container(
                aclient=aclient, container_name=container_name_or_id
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                train_status = STATUS_CONFIG.finish
            elif exit_status in {137, 143}:
                train_status = STATUS_CONFIG.stopped
            elif exit_status == 1:
                train_status = STATUS_CONFIG.failed

            await remove_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

    except ValueError as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except RuntimeError as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"Docker error: {e}")

    except Exception as e:
        train_status = STATUS_CONFIG.failed
        accel_logger.error(f"{e}")

    finally:
        try:
            info = await redis_async.client.hget(TASK_CONFIG.train, train_name)
            info = orjson.loads(info)
            info["container"]["train"]["status"] = train_status
            info["container"]["train"]["id"] = None
            await redis_async.client.hset(
                TASK_CONFIG.train, train_name, orjson.dumps(info)
            )
        except Exception as e:
            accel_logger.error(f"Database error: {e}")


async def run_train(
    image_name: str,
    cmd: list,
    docker_network_name: str,
    train_name: str,
    is_deepspeed: bool,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    env_var = [f"HF_HOME={COMMON_CONFIG.hf_home}"]
    if is_deepspeed:
        env_var.append("FORCE_TORCHRUN=1")
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
                f"{COMMON_CONFIG.root_path}/data:{COMMON_CONFIG.data_path}:rw",
                f"{COMMON_CONFIG.root_path}/saves/{train_name}:{COMMON_CONFIG.save_path}/{train_name}:rw",
            ],
            "NetworkMode": docker_network_name,
        },
        "Cmd": cmd,
        "Env": env_var,
    }

    try:
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_name_or_id = await create_container(
                aclient=aclient, name=f"train-{train_name}", data=data
            )
            started_container = await start_container(
                aclient=aclient, container_name_or_id=container_name_or_id
            )

        return started_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None


async def stop_train(
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

        return stopped_container

    except Exception as e:
        accel_logger.error(f"{e}")
        raise RuntimeError(f"{e}") from None
