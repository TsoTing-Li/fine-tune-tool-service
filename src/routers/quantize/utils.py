import os
from typing import Dict, Literal, Tuple, Union

import aiofiles
import httpx
import orjson
import yaml
from fastapi import status

from src.config.params import COMMON_CONFIG
from src.thirdparty.docker.api_handler import get_container_log


async def get_quantize_args(yaml_path: str) -> Tuple[str, str]:
    try:
        async with aiofiles.open(yaml_path) as af:
            content = await af.read()

        yaml_content = yaml.safe_load(content)

        return (
            yaml_content["output_dir"],
            yaml_content["finetuning_type"],
        )

    except FileNotFoundError:
        raise FileNotFoundError(f"{yaml_path} does not exists") from None


async def get_lora_base_model(train_name: str) -> str:
    file_path = f"/app/saves/{train_name}/lora/adapter_config.json"

    try:
        async with aiofiles.open(file_path, "rb") as af:
            adapter_config_content = await af.read()
            return orjson.loads(adapter_config_content)["base_model_name_or_path"]

    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} does not exists") from None

    except orjson.JSONDecodeError:
        raise TypeError("Invalid JSON format") from None


def get_model_snapshot(model_name: Union[str, None] = None) -> Dict[str, str]:
    from huggingface_hub import scan_cache_dir

    model_snapshot_path = dict()

    for cache_info in scan_cache_dir().repos:
        if model_name:
            if model_name == cache_info.repo_id:
                model_snapshot_path[model_name] = next(
                    (str(rev.snapshot_path) for rev in cache_info.revisions), ""
                )
                break
        else:
            if cache_info.repo_id not in model_snapshot_path:
                model_snapshot_path[cache_info.repo_id] = next(
                    (str(rev.snapshot_path) for rev in cache_info.revisions), ""
                )

    return model_snapshot_path


async def quantize_as_gguf(
    quantize_service_url: str,
    quantize_name: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    data = {
        "quantize_name": quantize_name,
        "checkpoint_path": f"{os.path.join(COMMON_CONFIG.root_path, os.path.relpath(checkpoint_path, COMMON_CONFIG.workspace_path))}",
        "output_path": f"{os.path.join(COMMON_CONFIG.root_path, os.path.relpath(output_path, COMMON_CONFIG.workspace_path))}",
        "hf_ori": False,
    }
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(quantize_service_url, json=data)

        if response.status_code == status.HTTP_200_OK:
            print(f"Full Container: {quantize_name} success")
        else:
            print(f"Full Container: {quantize_name} failed")
            print(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None

    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        async for log in get_container_log(
            aclient=aclient, container_name_or_id=response.json()["container_name"]
        ):
            if not log:
                return


async def stop_quantize(
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
            print(f"Fine-tune stopped, container: {container_name_or_id}")
            return container_name_or_id
        else:
            print(f"Fine-tune stop failed, container: {container_name_or_id}")
            print(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None
