from typing import Literal

import httpx

from src.thirdparty.docker.api_handler import create_container, start_container


async def start_vllm_container(
    image_name: str,
    service_port: int,
    docker_network_name: str,
    cmd: list,
    model_name: str,
    local_safetensors_path: str,
    hf_home: str,
) -> str:
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
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
                f"{local_safetensors_path}:{local_safetensors_path}:rw",
            ],
            "PortBindings": {f"{service_port}/tcp": [{"HostPort": f"{service_port}"}]},
            "AutoRemove": True,
            "NetworkMode": docker_network_name,
        },
        "Cmd": cmd,
        "Env": [f"HF_HOME={hf_home}"],
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        container_name_or_id = await create_container(
            aclient=aclient, name=f"vllm-{model_name}", data=data
        )

        started_container = await start_container(
            aclient=aclient, container_name_or_id=container_name_or_id
        )

        return started_container


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
