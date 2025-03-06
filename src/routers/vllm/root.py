import json

from fastapi import APIRouter, HTTPException, Response, status

from src.config.params import VLLM_CONFIG
from src.routers.vllm import schema, utils
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/vllm", tags=["VLLM"])


@router.post("/start/safetensors/")
async def start_vllm(request_data: schema.PostStartVLLM):
    error_handler = ResponseErrorHandler()

    try:
        container_name = await utils.start_vllm_container(
            image_name=request_data.image_name,
            service_port=request_data.service_port,
            docker_network_name=request_data.docker_network_name,
            cmd=[
                "--model",
                request_data.local_safetensors_path,
                "--gpu_memory_utilization",
                f"{request_data.gpu_memory_utilization}",
                "--max_model_len",
                f"{request_data.max_model_len}",
                "--tensor-parallel-size",
                f"{request_data.tensor_parallel_size}",
                "--enforce-eager",
                "--tokenizer",
                request_data.base_model,
                "--cpu-offload-gb",
                f"{request_data.cpu_offload_gb}",
                "--served-model-name",
                request_data.model_name,
                "--port",
                f"{request_data.service_port}",
            ],
            model_name=request_data.model_name,
            local_safetensors_path=request_data.local_safetensors_path,
            hf_home=request_data.hf_home,
        )

        await utils.run_vllm_model(
            vllm_url=f"http://{container_name}:{VLLM_CONFIG.port}"
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(
            {
                "vllm_service": f"http://{container_name}:{VLLM_CONFIG.port}",
                "container_name": container_name,
                "model_name": request_data.model_name,
            }
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_vllm(request_data: schema.PostStopVLLM):
    error_handler = ResponseErrorHandler()

    try:
        vllm_container = await utils.stop_vllm_container(
            container_name_or_id=request_data.vllm_container
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"vllm_container": request_data.vllm_container},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"vllm_container": vllm_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
