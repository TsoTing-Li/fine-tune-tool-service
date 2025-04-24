import asyncio
import os
import shutil
from typing import Literal

import aiofiles
import aiofiles.os
import httpx
import orjson
from fastapi import status

from src.config.params import COMMON_CONFIG, TASK_CONFIG
from src.thirdparty.docker.api_handler import (
    remove_container,
    stop_container,
    wait_for_container,
)
from src.thirdparty.redis.handler import redis_async


async def quantize_as_gguf(
    quantize_service_url: str,
    quantize_name: str,
    checkpoint_path: str,
    output_path: str,
) -> str:
    data = {
        "quantize_name": quantize_name,
        "checkpoint_path": f"{os.path.join(COMMON_CONFIG.root_path, os.path.relpath(checkpoint_path, COMMON_CONFIG.workspace_path))}",
        "output_path": f"{os.path.join(COMMON_CONFIG.root_path, os.path.relpath(output_path, COMMON_CONFIG.workspace_path))}",
        "hf_ori": False,
    }
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(quantize_service_url, json=data)

        if response.status_code != status.HTTP_200_OK:
            raise RuntimeError(
                f"Error: {response.status_code}, {response.json()['detail'][0]['msg']}"
            ) from None

    return response.json()["container_name"]


async def update_quantize_info(quantize_name: str, container_name: str) -> None:
    info = await redis_async.client.hget(TASK_CONFIG.train, quantize_name)
    info = orjson.loads(info)
    info["container"]["quantize"]["status"] = "active"
    info["container"]["quantize"]["id"] = container_name
    await redis_async.client.hset(TASK_CONFIG.train, quantize_name, orjson.dumps(info))


async def check_quantize_status(container_name_or_id: str) -> None:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        container_info = await wait_for_container(
            aclient=aclient, container_name=container_name_or_id
        )
        exit_status = container_info["StatusCode"]
        if exit_status == 0:
            return
        elif exit_status in {137, 143}:
            raise asyncio.CancelledError("received stop signal")
        elif exit_status == 1:
            raise RuntimeError("quantize")


async def remove_finish_container(container_name: str) -> None:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        await remove_container(aclient=aclient, container_name_or_id=container_name)


async def merge_async_tasks(quantize_name: str, container_name: str):
    tasks = [
        asyncio.create_task(
            update_quantize_info(
                quantize_name=quantize_name, container_name=container_name
            )
        ),
        asyncio.create_task(check_quantize_status(container_name_or_id=container_name)),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in pending:
        task.cancel()

    await asyncio.gather(*pending, return_exceptions=True)

    for task in done:
        try:
            task.result()
        except asyncio.CancelledError as e:
            raise asyncio.CancelledError(f"{e}") from None
        except RuntimeError as e:
            raise RuntimeError(f"{e}") from None
        except Exception as e:
            raise Exception(f"{e}") from None


async def stop_quantize(
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


async def del_quantize_folder(qunatize_folder: str) -> None:
    is_exists = await aiofiles.os.path.exists(qunatize_folder)

    if is_exists:
        if not await aiofiles.os.path.isdir(qunatize_folder):
            return

        await asyncio.to_thread(shutil.rmtree, qunatize_folder)
