from typing import Literal

import httpx
from fastapi import status

from src.config.params import (
    COMMON_CONFIG,
    DOCKERNETWORK_CONFIG,
    MAINSERVICE_CONFIG,
    OLLAMA_CONFIG,
    VLLM_CONFIG,
)
from src.utils.utils import assemble_image_name


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
            raise RuntimeError(f"{response.text}") from None


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
