import json
import os
from typing import Literal

import aiofiles
import httpx
import yaml

from inno_service.utils.utils import generate_uuid


async def get_model_params(path: str) -> dict:
    try:
        async with aiofiles.open(path) as af:
            content = await af.read()

        return yaml.safe_load(content)

    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exists") from None


async def start_vllm_container(
    image_name: str,
    cmd: list,
    service_port: int,
    model_name: str,
    base_model: str,
    finetune_type: Literal["full", "lora"],
    cpu_offload_gb: int = 0,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    hf_home = os.environ["HF_HOME"]
    custom_model_path = f"{os.environ['SAVE_PATH']}/{model_name}/{finetune_type}"
    ws_path = os.environ["WS"]
    container_name = f"vllm-{model_name}-{generate_uuid()}"

    if finetune_type == "full":
        cmd.extend(["--cpu-offload-gb", f"{cpu_offload_gb}"])
    elif finetune_type == "lora":
        lora_data = {
            "name": model_name,
            "path": f"{ws_path}/{custom_model_path}",
            "base_model_name": base_model,
        }
        cmd.extend(["--enable-lora", "--lora-modules", json.dumps(lora_data)])

    data = {
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{hf_home}:{hf_home}:rw",
                f"{os.environ['ROOT_PATH']}/{custom_model_path}:{ws_path}/{custom_model_path}:rw",
            ],
            "PortBindings": {"8000/tcp": [{"HostPort": f"{service_port}"}]},
            "AutoRemove": True,
            "NetworkMode": "host",
        },
        # "NetworkingConfig": {"EndpointsConfig": {acceltune_network: {}}},
        "Cmd": cmd,
        "Env": [f"HF_HOME={hf_home}"],
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(
            "http://docker/containers/create",
            json=data,
            params={"name": container_name},
        )

        if response.status_code == 201:
            print("created")
            response = await aclient.post(
                f"http://docker/containers/{container_name}/start"
            )

            if response.status_code == 204:
                print("started")
                return container_name

            else:
                print("startup fail")
                raise RuntimeError(
                    f"Error: {response.status_code}, {response.text}"
                ) from None

        else:
            print("created fail")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None


async def stop_vllm_container(
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
            print(f"VLLM stopped, container: {container_name_or_id}")
            return container_name_or_id
        else:
            print(f"VLLM failed, container: {container_name_or_id}")
            print(f"Error: {response.status_code}, {response.text}")
            raise RuntimeError(
                f"Error: {response.status_code}, {response.text}"
            ) from None
