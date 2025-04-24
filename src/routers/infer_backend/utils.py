import os
import re
from typing import Literal, Union

import httpx
import orjson
from fastapi import HTTPException, status

from src.config.params import (
    COMMON_CONFIG,
    DOCKERNETWORK_CONFIG,
    MAINSERVICE_CONFIG,
    OLLAMA_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
    VLLM_CONFIG,
)
from src.routers.train.utils import export_data_process, write_yaml
from src.thirdparty.docker.api_handler import remove_container, wait_for_container
from src.thirdparty.redis.handler import redis_async
from src.utils.utils import assemble_image_name


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

                await remove_container(
                    aclient=aclient, container_name_or_id=container_name
                )

            return merge_status
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.json()
            )


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


async def check_merge_status_and_reload(
    name: str, train_args: dict, last_model_path: Union[str, None]
) -> dict:
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
                    raise RuntimeError(e.detail[0]["msg"]) from None
        else:
            raise FileNotFoundError("can not found model file")

    elif finetuning_type == "full":
        if not os.path.exists(last_model_path) or last_model_path is None:
            raise FileNotFoundError("can not found model file")

    info = await redis_async.client.hget(TASK_CONFIG.train, name)
    info = orjson.loads(info)
    return info


async def startup_vllm_service(
    model_name: str,
    local_safetensors_path: str,
    base_model: str,
    hf_home: str,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    cpu_offload_gb: int = 110,
    tensor_parallel_size: int = 1,
) -> dict:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://{MAINSERVICE_CONFIG.container_name}:{MAINSERVICE_CONFIG.port}/acceltune/vllm/start/safetensors/",
            json={
                "image_name": assemble_image_name(
                    username=COMMON_CONFIG.username,
                    repository=VLLM_CONFIG.name,
                    tag=VLLM_CONFIG.tag,
                ),
                "service_port": VLLM_CONFIG.port,
                "docker_network_name": DOCKERNETWORK_CONFIG.network_name,
                "model_name": model_name,
                "local_safetensors_path": local_safetensors_path,
                "base_model": base_model,
                "hf_home": hf_home,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "cpu_offload_gb": cpu_offload_gb,
                "tensor_parallel_size": tensor_parallel_size,
            },
        )

        if response.status_code == status.HTTP_200_OK:
            return response.json()
        else:
            raise RuntimeError(f"{response.text}") from None


async def startup_ollama_service(local_gguf_path: str, model_name: str) -> dict:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://{MAINSERVICE_CONFIG.container_name}:{MAINSERVICE_CONFIG.port}/acceltune/ollama/start/",
            json={
                "image_name": assemble_image_name(
                    username=COMMON_CONFIG.username,
                    repository=OLLAMA_CONFIG.name,
                    tag=OLLAMA_CONFIG.tag,
                ),
                "docker_network_name": DOCKERNETWORK_CONFIG.network_name,
                "local_gguf_path": local_gguf_path,
                "model_name": model_name,
            },
        )

        if response.status_code == status.HTTP_200_OK:
            return response.json()
        else:
            raise RuntimeError(f"{response.json()}") from None


async def stop_model_service(
    container_name: str, infer_backend_type: Literal["vllm", "ollama"]
) -> str:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://{MAINSERVICE_CONFIG.container_name}:{MAINSERVICE_CONFIG.port}/acceltune/{infer_backend_type}/stop/",
            json={f"{infer_backend_type}_container": container_name},
        )

        if response.status_code != status.HTTP_200_OK:
            raise RuntimeError(f"{response.text}")

        return response.json()[f"{infer_backend_type}_container"]
