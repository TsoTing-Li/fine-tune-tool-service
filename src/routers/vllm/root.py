import json
import os

from fastapi import APIRouter, HTTPException, Response, status

from src.routers.vllm import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/vllm", tags=["VLLM"])

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")
VLLM_SERVICE_PORT = os.getenv("VLLM_SERVICE_PORT", 8003)


@router.post("/start/safetensors/")
async def start_vllm(request_data: schema.PostStartVLLM):
    validator.PostStartVLLM(model_name=request_data.model_name)
    error_handler = ResponseErrorHandler()

    try:
        model_params = await utils.get_model_params(
            path=os.path.join(
                SAVE_PATH,
                f"{request_data.model_name}/{request_data.model_name}.yaml",
            )
        )

        container_name = await utils.start_vllm_container(
            image_name=f"{os.environ['USER_NAME']}/{os.environ['VLLM_SERVICE_NAME']}:{os.environ['VLLM_SERVICE_TAG']}",
            cmd=[
                "--model",
                model_params["model_name_or_path"]
                if model_params["finetuning_type"] == "lora"
                else model_params["output_dir"],
                "--gpu_memory_utilization",
                f"{request_data.gpu_memory_utilization}",
                "--max_model_len",
                f"{request_data.max_model_len}",
                "--tensor-parallel-size",
                f"{request_data.tensor_parallel_size}",
                "--enforce-eager",
                "--tokenizer",
                model_params["model_name_or_path"],
            ],
            service_port=VLLM_SERVICE_PORT,
            model_name=request_data.model_name,
            base_model=model_params["model_name_or_path"],
            finetune_type=model_params["finetuning_type"],
            cpu_offload_gb=request_data.cpu_offload_gb,
        )
        service_model_name = (
            request_data.model_name
            if model_params["finetuning_type"] == "lora"
            else model_params["output_dir"]
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
                "vllm_service": f"http://{container_name}:{VLLM_SERVICE_PORT}",
                "container_name": container_name,
                "model_name": service_model_name,
            }
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_vllm(request_data: schema.PostStopVLLM):
    validator.PostStopVLLM(vllm_container=request_data.vllm_container)
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
