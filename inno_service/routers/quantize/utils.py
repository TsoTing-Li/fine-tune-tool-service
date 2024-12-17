import os
from typing import Dict, Literal, Tuple, Union

import aiofiles
import httpx
import orjson
import yaml


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
    finetune_type: Literal["full", "lora"],
) -> dict:
    host_path = os.environ["ROOT_PATH"]
    base_path = os.environ["WS"]
    container_ids = dict()

    if finetune_type == "full":
        full_data = {
            "quantize_name": quantize_name,
            "checkpoint_path": f"{os.path.join(host_path, os.path.relpath(checkpoint_path, base_path))}",
            "output_path": f"{os.path.join(host_path, os.path.relpath(output_path, base_path))}",
            "hf_ori": False,
        }
        async with httpx.AsyncClient(timeout=None) as aclient:
            response = await aclient.post(
                f"{quantize_service_url}/{finetune_type}/", json=full_data
            )

            if response.status_code == 200:
                print(f"Full Container: {quantize_name} success")
                container_ids["full_container"] = response.json()["container_name"]
            else:
                print(f"Full Container: {quantize_name} failed")
                print(f"Error: {response.status_code}, {response.text}")
                raise RuntimeError(
                    f"Error: {response.status_code}, {response.text}"
                ) from None

    elif finetune_type == "lora":
        base_model = await get_lora_base_model(train_name=quantize_name)
        model_snapshot_path = get_model_snapshot(model_name=base_model)[base_model]

        full_data = {
            "quantize_name": quantize_name,
            "checkpoint_path": f"{model_snapshot_path}",
            "output_path": f"{os.path.join(host_path, os.path.relpath(output_path, base_path))}",
            "hf_ori": True,
        }

        lora_data = {
            "quantize_name": quantize_name,
            "base_model": model_snapshot_path,
            "lora_path": f"{os.path.join(host_path, os.path.relpath(checkpoint_path, base_path))}",
            "output_path": f"{os.path.join(host_path, os.path.relpath(output_path, base_path))}",
            "hf_ori": True,
        }

        async with httpx.AsyncClient(timeout=None) as aclient:
            response = await aclient.post(
                f"{quantize_service_url}/full/", json=full_data
            )

            if response.status_code == 200:
                container_ids["full_container"] = response.json()["container_name"]
                print(f"Full Container: {quantize_name} success")

                response = await aclient.post(
                    f"{quantize_service_url}/{finetune_type}/", json=lora_data
                )

                if response.status_code == 200:
                    container_ids["lora_container"] = response.json()["container_name"]
                    print(f"Lora Container: {quantize_name} success")

                else:
                    print(f"Lora Container: {quantize_name} failed")
                    print(f"Error: {response.status_code}, {response.text}")
                    raise RuntimeError(
                        f"Error: {response.status_code}, {response.text}"
                    ) from None
            else:
                print(f"Full Container: {quantize_name} failed")
                print(f"Error: {response.status_code}, {response.text}")
                raise RuntimeError(
                    f"Error: {response.status_code}, {response.text}"
                ) from None

    return container_ids
