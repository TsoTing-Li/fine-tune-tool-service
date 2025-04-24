import asyncio
import hashlib
import os
import re
import zipfile
from collections.abc import AsyncGenerator
from typing import Any, Literal, Tuple, Union

import aiofiles
import aiofiles.os
import httpx
import orjson
from fastapi import HTTPException, status

from src.config.params import (
    MAINSERVICE_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
from src.routers.accelbrain.error import AccelBrainError, AccelTuneError
from src.routers.train.utils import export_data_process, write_yaml
from src.thirdparty.docker.api_handler import remove_container, wait_for_container
from src.thirdparty.redis.handler import redis_async


async def call_internal_merge_api(merge_name: str) -> str:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/merge/start/",
            json={"merge_name": merge_name},
        )
        container_name = response.json()["container_name"]

    if response.status_code == status.HTTP_200_OK:
        transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
        async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
            container_info = await wait_for_container(
                aclient=aclient, container_name=container_name
            )
            exit_status = container_info["StatusCode"]
            if exit_status == 0:
                merge_status = STATUS_CONFIG.finish
            elif exit_status in {137, 143}:
                merge_status = STATUS_CONFIG.stopped
            elif exit_status == 1:
                merge_status = STATUS_CONFIG.failed
            else:
                merge_status = STATUS_CONFIG.failed

            await remove_container(aclient=aclient, container_name_or_id=container_name)

        return merge_status
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())


def check_file_complete(path: str) -> bool:
    shard_pattern = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
    found_shards = {}
    required_files = {
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    found_files = set()
    is_shard_model = False

    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name

            if name in required_files:
                found_files.add(name)

            if name == "model.safetensors":
                is_shard_model = True

            match = shard_pattern.fullmatch(name)
            if match:
                idx = int(match.group(1))
                total = int(match.group(2))
                found_shards[idx] = total

            if name == "model.safetensors.index.json":
                found_files.add(name)

    if not required_files.issubset(found_files):
        return False

    if is_shard_model:
        return True

    if found_shards:
        shard_total = next(iter(found_shards.values()))
        if (
            len(found_shards) == shard_total
            and "model.safetensors.index.json" in found_files
        ):
            return True

    return False


async def update_last_model_path(name: str, last_model_path: str):
    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, name)
        info = orjson.loads(info)
        info["last_model_path"] = last_model_path

        await redis_async.client.hset(TASK_CONFIG.train, name, orjson.dumps(info))

    except Exception:
        raise RuntimeError("Database error") from None


async def check_merge_status(
    name: str, train_args: dict, last_model_path: Union[str, None]
) -> None:
    finetuning_type: Literal["full", "lora"] = train_args["finetuning_type"]
    template = train_args["template"]
    base_model = train_args["base_model"]
    root_output_dir = os.path.dirname(train_args["output_dir"])

    if finetuning_type == "lora":
        if last_model_path is not None:
            if not os.path.exists(last_model_path) or not check_file_complete(
                path=last_model_path
            ):
                merge_path = os.path.join(root_output_dir, "merge")
                export_data = export_data_process(
                    adapter_name_or_path=last_model_path,
                    export_dir=merge_path,
                    model_name_or_path=base_model,
                    template=template,
                    finetuning_type=finetuning_type,
                )
                await write_yaml(
                    path=os.path.join(root_output_dir, "export.yaml"), data=export_data
                )
                try:
                    merge_status = await call_internal_merge_api(merge_name=name)
                    if merge_status != STATUS_CONFIG.finish:
                        raise RuntimeError(f"merge {merge_status}")
                    await update_last_model_path(name=name, last_model_path=merge_path)
                except HTTPException as e:
                    raise AccelTuneError(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        action="Check file",
                        progress=-1,
                        detail={"error": e.detail[0]["msg"]},
                    ) from None
                except Exception as e:
                    raise AccelTuneError(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        action="Check file",
                        progress=-1,
                        detail={"error": f"{e}"},
                    ) from None
        else:
            raise AccelTuneError(
                status_code=status.HTTP_404_NOT_FOUND,
                action="Check file",
                progress=-1,
                detail={"error": "can not found model file"},
            ) from None

    elif finetuning_type == "full":
        if not os.path.exists(last_model_path) or last_model_path is None:
            raise AccelTuneError(
                status_code=status.HTTP_404_NOT_FOUND,
                action="Check file",
                progress=-1,
                detail={"error": "can not found model file"},
            ) from None


async def call_internal_quantize_api(quantize_name: str) -> None:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/quantize/start/",
            json={"quantize_name": quantize_name},
        )

        if response.status_code == status.HTTP_200_OK:
            return
        else:
            raise AccelTuneError(
                status_code=response.status_code,
                action="Internal quantize",
                progress=-1,
                detail={"error": response.json()["detail"][0]["msg"]},
            )


async def check_quantize_status(quantize_name: str):
    try:
        while True:
            info = await redis_async.client.hget(TASK_CONFIG.train, quantize_name)
            info = orjson.loads(info)

            if info["container"]["quantize"]["status"] == STATUS_CONFIG.finish:
                break
            elif info["container"]["quantize"]["status"] == STATUS_CONFIG.active:
                await asyncio.sleep(3)
            elif info["container"]["quantize"]["status"] in {
                STATUS_CONFIG.setup,
                STATUS_CONFIG.failed,
            }:
                await call_internal_quantize_api(quantize_name=quantize_name)
    except AccelTuneError:
        raise
    except Exception as e:
        raise AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="Internal quantize",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None


async def check_accelbrain_url(accelbrain_url: str) -> Tuple[str, int]:
    try:
        async with httpx.AsyncClient() as aclient:
            response = await aclient.get(f"http://{accelbrain_url}/model_handler/")

            if (
                response.status_code == status.HTTP_200_OK
                and response.json()["status"] == "alive"
            ):
                return response.json()["status"], response.status_code

    except httpx.ConnectError:
        raise ConnectionError("AccelBrain Service is unavailable") from None

    except httpx.TimeoutException:
        raise TimeoutError("Request timeout") from None  # default is set 5 seconds


def calc_sha256(file_path: str, chunk_size: int = 65536) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()

    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} does not exists") from None

    except Exception as e:
        raise RuntimeError(e) from None


def zip_folder_and_get_hash(path: str, zip_path: str) -> dict:
    file_hashes = dict()

    try:
        with zipfile.ZipFile(
            file=zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)

                file_hash = calc_sha256(file_path=file_path)
                file_hashes[file] = file_hash

                zipf.write(file_path, arcname=os.path.relpath(file_path, path))

        zip_hash = calc_sha256(file_path=zip_path)
        file_hashes[os.path.basename(zip_path)] = zip_hash

        return file_hashes

    except FileNotFoundError as e:
        raise AccelTuneError(
            status_code=status.HTTP_404_NOT_FOUND,
            action="Zip folder",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None

    except Exception as e:
        raise AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="Zip folder",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None


async def async_file_generator(
    file_path: str, deploy_unique_key: str, chunk_size: int = 50 * 1024 * 1024
) -> AsyncGenerator[bytes, None]:
    try:
        async with aiofiles.open(file_path, "rb") as af:
            await af.seek(0, 2)
            file_size = await af.tell()
            await af.seek(0)

            uploaded_size = 0
            while data := await af.read(chunk_size):
                uploaded_size += len(data)
                upload_progress = round((uploaded_size / file_size), 2)

                try:
                    await redis_async.client.rpush(
                        f"{deploy_unique_key}-upload_progress", upload_progress
                    )
                except Exception as e:
                    raise RuntimeError(f"Database error: {e}") from None

                yield data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}") from None

    except Exception as e:
        raise RuntimeError(f"{e}") from None


async def generate_multi_part(
    file_path: str, deploy_unique_key: str, model_name: str, boundary: bytes
) -> AsyncGenerator[bytes, None]:
    file_name = os.path.basename(file_path)

    try:
        yield b"--" + boundary + b"\r\n"
        yield b'Content-Disposition: form-data; name="model_name_on_ollama"\r\n'
        yield b"Content-Type: text/plain\r\n\r\n"
        yield f"{model_name}\r\n".encode()

        yield b"--" + boundary + b"\r\n"
        yield f'Content-Disposition: form-data; name="model"; filename="{file_name}"\r\n'.encode()
        yield b"Content-Type: application/zip\r\n\r\n"

        async for chunk in async_file_generator(
            file_path=file_path, deploy_unique_key=deploy_unique_key
        ):
            yield chunk

        yield b"\r\n--" + boundary + b"--\r\n"

    except FileNotFoundError as e:
        raise AccelTuneError(
            status_code=status.HTTP_404_NOT_FOUND,
            action="Generate multi part",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None

    except Exception as e:
        raise AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="Generate multi part",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None


async def call_accelbrain_deploy(
    file_path: str,
    model_name: str,
    deploy_path: str,
    deploy_unique_key: str,
    accelbrain_url: str,
    boundary: bytes,
) -> AsyncGenerator[bytes, None, None]:
    yield (
        orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_200_OK,
                    "message": {
                        "action": "Start compress file",
                        "progress": 0.0,
                        "detail": {"model_name": model_name},
                    },
                }
            }
        )
    )
    os.makedirs(os.path.dirname(deploy_path), exist_ok=True)
    await asyncio.to_thread(zip_folder_and_get_hash, file_path, deploy_path)

    async with httpx.AsyncClient(timeout=None) as aclient:
        async with aclient.stream(
            "POST",
            f"http://{accelbrain_url}/model_handler/deploy/",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary.decode('ascii')}"
            },
            content=generate_multi_part(
                file_path=deploy_path,
                deploy_unique_key=deploy_unique_key,
                model_name=model_name,
                boundary=boundary,
            ),
        ) as response:
            if response.status_code == status.HTTP_200_OK:
                async for chunk in response.aiter_lines():
                    if chunk:
                        receive_content = orjson.loads(chunk.strip())

                        if receive_content["status"] != status.HTTP_200_OK:
                            raise AccelBrainError(
                                status_code=receive_content["status"],
                                action=receive_content["message"]["action"],
                                progress=receive_content["message"]["progress"],
                                detail=receive_content["message"]["detail"],
                            )

                        accelbrain_info = {"AccelBrain": receive_content}
                        yield orjson.dumps(accelbrain_info)
            else:
                error_content = await response.aread()
                raise AccelBrainError(
                    status_code=response.status_code,
                    action="AccelBrain process",
                    progress=-1,
                    detail={"error": error_content.decode()},
                )


async def monitor_progress(
    deploy_unique_key: str, model_name: str
) -> AsyncGenerator[bytes, None, None]:
    try:
        while True:
            (
                _,
                upload_progress,
            ) = await redis_async.client.blpop(f"{deploy_unique_key}-upload_progress")
            yield (
                orjson.dumps(
                    {
                        "AccelTune": {
                            "status": status.HTTP_200_OK,
                            "message": {
                                "action": "Upload file",
                                "progress": float(upload_progress),
                                "detail": {"model_name": model_name},
                            },
                        }
                    }
                )
            )

            if upload_progress == "1.0":
                break

    except asyncio.CancelledError as e:
        raise AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Async task cancelled",
            action="Monitor progress",
            progress=-1,
            detail={"error": f"{e}"},
        ) from None


async def merge_async_generators(
    *gens: AsyncGenerator[Any, None],
) -> AsyncGenerator[Any, None, None]:
    tasks = {asyncio.create_task(gen.__anext__()): gen for gen in gens}

    while tasks:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            gen = tasks.pop(task)
            try:
                item = task.result()
            except StopAsyncIteration:
                continue
            else:
                yield item
                tasks[asyncio.create_task(gen.__anext__())] = gen


async def update_deploy_status(key: str, new_status: str) -> Union[bytes, None]:
    try:
        info = await redis_async.client.hget(TASK_CONFIG.deploy, key)
        info = orjson.loads(info)
        info["status"] = new_status
        await redis_async.client.hset(TASK_CONFIG.deploy, key, orjson.dumps(info))
        return
    except Exception as e:
        acceltune_error = AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="Database error",
            progress=-1,
            detail={"error": f"{e}"},
        )
        return orjson.dumps(acceltune_error.error_data)


async def remove_zip_file(file_path: str) -> None:
    is_exists = await aiofiles.os.path.exists(file_path)
    if is_exists:
        await aiofiles.os.remove(file_path)


async def deploy_to_accelbrain_service(
    file_path: str,
    model_name: str,
    train_args: dict,
    last_model_path: str,
    deploy_path: str,
    deploy_unique_key: str,
    accelbrain_url: str,
) -> AsyncGenerator[str, None, None]:
    try:
        yield (
            orjson.dumps(
                {
                    "AccelTune": {
                        "status": status.HTTP_200_OK,
                        "message": {
                            "action": "Check file",
                            "progress": 0.0,
                            "detail": {"model_name": model_name},
                        },
                    }
                }
            )
        ) + b"\n"
        await check_merge_status(
            name=model_name, train_args=train_args, last_model_path=last_model_path
        )

        yield (
            orjson.dumps(
                {
                    "AccelTune": {
                        "status": status.HTTP_200_OK,
                        "message": {
                            "action": "Start quantize",
                            "progress": 0.0,
                            "detail": {"model_name": model_name},
                        },
                    }
                }
            )
        ) + b"\n"
        await check_quantize_status(quantize_name=model_name)

        monitor_progress_generator = monitor_progress(
            deploy_unique_key=deploy_unique_key, model_name=model_name
        )
        accelbrain_deploy_generator = call_accelbrain_deploy(
            file_path=file_path,
            model_name=model_name,
            deploy_path=deploy_path,
            deploy_unique_key=deploy_unique_key,
            accelbrain_url=accelbrain_url,
            boundary=os.urandom(16).hex().encode("ascii"),
        )

        async for item in merge_async_generators(
            monitor_progress_generator, accelbrain_deploy_generator
        ):
            if isinstance(item, bytes):
                yield item + b"\n"
            elif isinstance(item, str):
                yield item + "\n"

        target_model_status = STATUS_CONFIG.finish

    except AccelTuneError as e:
        target_model_status = STATUS_CONFIG.failed
        yield str(e) + "\n"

    except AccelBrainError as e:
        target_model_status = STATUS_CONFIG.failed
        yield str(e) + "\n"

    except httpx.ConnectError as e:
        target_model_status = STATUS_CONFIG.failed
        acceltune_error = AccelTuneError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            action="Connected to AccelBrain error",
            progress=-1,
            detail={"error": f"{e}"},
        )
        yield orjson.dumps(acceltune_error.error_data) + b"\n"

    except httpx.TimeoutException as e:
        target_model_status = STATUS_CONFIG.failed
        acceltune_error = AccelTuneError(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            action="Request timeout",
            progress=-1,
            detail={"error": f"{e}"},
        )
        yield orjson.dumps(acceltune_error.error_data) + b"\n"

    except (KeyboardInterrupt, SystemExit) as e:
        target_model_status = STATUS_CONFIG.failed
        acceltune_error = AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="KeyboardInterrupt or SystemExit",
            progress=-1,
            detail={"error": f"{e}"},
        )
        yield orjson.dumps(acceltune_error.error_data) + b"\n"

    except Exception as e:
        target_model_status = STATUS_CONFIG.failed
        acceltune_error = AccelTuneError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            action="Unexpected error",
            progress=-1,
            detail={"error": f"{e}"},
        )
        yield orjson.dumps(acceltune_error.error_data) + b"\n"

    finally:
        await remove_zip_file(file_path=deploy_path)
        result = await update_deploy_status(
            key=deploy_unique_key,
            new_status=target_model_status,
        )
        if result and isinstance(result, bytes):
            yield result + b"\n"
