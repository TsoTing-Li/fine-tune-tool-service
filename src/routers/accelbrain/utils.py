import asyncio
import hashlib
import os
import threading
import zipfile
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any, Tuple, Union

import aiofiles
import httpx
import orjson
from fastapi import status

import redis.asyncio as async_redis
from src.config.params import MAINSERVICE_CONFIG, STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_async

zip_locks = defaultdict(threading.Lock)


async def update_status_safely(name: str, key: str, new_status: dict):
    while True:
        try:
            await redis_async.client.watch(key)

            current_value = await redis_async.client.hget(name, key)
            current_value = orjson.loads(current_value)

            current_value["deploy_status"].update(new_status)

            pipe = await redis_async.client.pipeline()
            pipe.multi()
            await pipe.hset(name, key, orjson.dumps(current_value))
            await pipe.execute()
            break

        except async_redis.WatchError:
            continue


async def call_internal_quantize_api(quantize_name: str) -> None:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{MAINSERVICE_CONFIG.port}/acceltune/quantize/start/",
            json={"quantize_name": quantize_name},
        )

        if response.status_code == status.HTTP_200_OK:
            return
        elif response.status_code == status.HTTP_404_NOT_FOUND:
            raise KeyError(f"{response.json()['detail'][0]['msg']}")
        elif response.status_code == status.HTTP_409_CONFLICT:
            raise ValueError(f"{response.json()['detail'][0]['msg']}")
        elif response.status_code == 499:
            raise ConnectionResetError(f"{response.json()['detail'][0]['msg']}")
        elif response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise RuntimeError(f"{response.json()['detail'][0]['msg']}")


async def check_accelbrain_url(accelbrain_url: str) -> Tuple[str, int]:
    try:
        async with httpx.AsyncClient() as aclient:
            response = await aclient.get(f"http://{accelbrain_url}/model_handler/")

            if (
                response.status_code == status.HTTP_200_OK
                and response.json()["status"] == "alive"
            ):
                return response.json()["status"], response.status_code
            else:
                raise ValueError(f"AccelBrain Url: {accelbrain_url} is not alive")

    except httpx.ConnectError:
        raise ValueError(f"AccelBrain Url: {accelbrain_url} is not alive") from None

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

    with zip_locks[zip_path]:
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
            raise FileNotFoundError(e) from None

        except Exception as e:
            raise RuntimeError(e) from None


async def async_file_generator(
    file_path: str, model_name: str, chunk_size: int = 50 * 1024 * 1024
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
                        f"{model_name}-upload_progress", upload_progress
                    )
                except Exception as e:
                    raise RuntimeError(f"Database error: {e}") from None

                yield data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}") from None

    except Exception as e:
        raise RuntimeError(f"{e}") from None


async def generate_multi_part(
    file_path: str, model_name: str, boundary: bytes
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
            file_path=file_path, model_name=model_name
        ):
            yield chunk

        yield b"\r\n--" + boundary + b"--\r\n"

    except Exception as e:
        raise RuntimeError(f"{e}") from None


async def call_accelbrain_deploy(
    file_path: str,
    model_name: str,
    deploy_path: str,
    accelbrain_url: str,
    boundary: bytes,
):
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
                file_path=deploy_path, model_name=model_name, boundary=boundary
            ),
        ) as response:
            if response.status_code == status.HTTP_200_OK:
                async for chunk in response.aiter_lines():
                    if chunk:
                        accelbrain_info = {"AccelBrain": orjson.loads(chunk)}
                        yield orjson.dumps(accelbrain_info)
            else:
                error_content = await response.aread()
                accelbrain_error = {"AccelBrain": error_content.decode()}
                yield orjson.dumps(accelbrain_error)


async def monitor_progress(model_name: str):
    try:
        while True:
            (
                _,
                upload_progress,
            ) = await redis_async.client.blpop(f"{model_name}-upload_progress")
            yield orjson.dumps(
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

            if upload_progress == "1.0":
                break

    except asyncio.CancelledError as e:
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": {
                        "action": "Async task cancelled",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )


async def merge_async_generators(
    *gens: AsyncGenerator[Any, None],
) -> AsyncGenerator[Any, None]:
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


async def update_deploy_status(
    name: str, key: str, new_status: dict
) -> Union[str, None]:
    try:
        await update_status_safely(name=name, key=key, new_status=new_status)
        return
    except Exception as e:
        return orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": {
                        "action": "Database error",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )


async def deploy_to_accelbrain_service(
    file_path: str,
    model_name: str,
    deploy_path: str,
    accelbrain_device: str,
    accelbrain_url: str,
) -> AsyncGenerator[str, None]:
    try:
        yield orjson.dumps(
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
        await call_internal_quantize_api(quantize_name=model_name)

        monitor_progress_generator = monitor_progress(model_name=model_name)
        accelbrain_deploy_generator = call_accelbrain_deploy(
            file_path=file_path,
            model_name=model_name,
            deploy_path=deploy_path,
            accelbrain_url=accelbrain_url,
            boundary=os.urandom(16).hex().encode("ascii"),
        )

        async for item in merge_async_generators(
            monitor_progress_generator, accelbrain_deploy_generator
        ):
            yield item

        target_model_status = STATUS_CONFIG.finish

    except KeyError as e:
        target_model_status = STATUS_CONFIG.failed
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_404_NOT_FOUND,
                    "message": {
                        "action": "deploy_name not found",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    except ValueError as e:
        target_model_status = STATUS_CONFIG.failed
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_409_CONFLICT,
                    "message": {
                        "action": "process in launch",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    except ConnectionResetError as e:
        target_model_status = STATUS_CONFIG.stopped
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": 499,
                    "message": {
                        "action": "Client close the request",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    except httpx.ConnectError as e:
        target_model_status = STATUS_CONFIG.failed
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_502_BAD_GATEWAY,
                    "message": {
                        "action": "connected error",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    except httpx.TimeoutException as e:
        target_model_status = STATUS_CONFIG.failed
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_408_REQUEST_TIMEOUT,
                    "message": {
                        "action": "request timeout",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    except Exception as e:
        target_model_status = STATUS_CONFIG.failed
        yield orjson.dumps(
            {
                "AccelTune": {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": {
                        "action": "unexpected error",
                        "progress": -1,
                        "detail": {"error": f"{e}"},
                    },
                }
            }
        )

    finally:
        result = await update_deploy_status(
            name=TASK_CONFIG.accelbrain_device,
            key=accelbrain_device,
            new_status={model_name: target_model_status},
        )
        if result:
            yield result
