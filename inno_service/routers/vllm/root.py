import json
import os

from fastapi import APIRouter, Response, status

from inno_service.routers.vllm import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler

router = APIRouter(prefix="/vllm", tags=["VLLM"])


@router.post("/start/safetensors/")
async def start_vllm(request_data: schema.PostStartVLLM):
    validator.PostStartVLLM(model_name=request_data.model_name)
    error_handler = ResponseErrorHandler()

    try:
        model_params = await utils.get_model_params(
            path=os.path.join(
                os.environ["WS"],
                os.environ["SAVE_PATH"],
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
            model_name=request_data.model_name,
            base_model=model_params["model_name_or_path"],
            finetune_type=model_params["finetuning_type"],
            cpu_offload_gb=request_data.cpu_offload_gb,
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input=request_data.model_dump(),
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"vllm_service": container_name}),
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
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"vllm_container": request_data.vllm_container},
        )
        return Response(
            content=json.dumps(error_handler.errors),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"vllm_container": vllm_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
