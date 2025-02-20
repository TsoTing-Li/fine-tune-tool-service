import os

import httpx

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


async def startup_model_service(model_name: str) -> dict:
    if os.path.exists(f"{SAVE_PATH}/{model_name}/quantize"):
        print("get 'gguf' format")
        service_func = startup_ollama_service
        service_type = "ollama_service"
    else:
        print("not found 'gguf' format")
        print("get 'safetensors' format")
        service_func = startup_vllm_service
        service_type = "vllm_service"

    service_info = await service_func(model_name=model_name)

    return {
        "model_service_url": service_info[service_type],
        "container_name": service_info["container_name"],
        "model_name": service_info["model_name"],
    }


async def startup_vllm_service(model_name: str) -> dict:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{os.environ['MAIN_SERVICE_PORT']}/acceltune/vllm/start/safetensors/",
            json={"model_name": model_name},
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"{response.text}") from None


async def startup_ollama_service(model_name: str) -> dict:
    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{os.environ['MAIN_SERVICE_PORT']}/acceltune/ollama/start/",
            json={"model_name": model_name},
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"{response.text}") from None


async def stop_model_service(container_name: str) -> str:
    if container_name.startswith("ollama"):
        service_type = "ollama"
    elif container_name.startswith("vllm"):
        service_type = "vllm"

    async with httpx.AsyncClient(timeout=None) as aclient:
        response = await aclient.post(
            f"http://127.0.0.1:{os.environ['MAIN_SERVICE_PORT']}/acceltune/{service_type}/stop/",
            json={f"{service_type}_container": container_name},
        )

        if response.status_code != 200:
            raise RuntimeError(f"{response.text}")

        return response.json()[f"{service_type}_container"]
