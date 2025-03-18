import json
import os
from typing import Annotated, Union

import orjson
from fastapi import APIRouter, HTTPException, Query, Response, status

from src.config.params import COMMON_CONFIG, STATUS_CONFIG, TASK_CONFIG
from src.routers.infer_backend import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/infer-backend", tags=["Infer-Backend"])


@router.post("/start/")
async def start_infer_backend(
    request_data: schema.PostInferBackendStart,
):
    validator.PostInferBackendStart(model_name=request_data.model_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.model_name)
        info = orjson.loads(info)

        service_type = "vllm"
        model_service_info = await utils.startup_vllm_service(
            model_name=request_data.model_name,
            local_safetensors_path=os.path.join(
                COMMON_CONFIG.root_path,
                "saves",
                request_data.model_name,
                "merge" if info["train_args"]["finetuning_type"] == "lora" else "full",
            ),
            base_model=info["train_args"]["model_name_or_path"],
            hf_home=COMMON_CONFIG.hf_home,
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

    try:
        info["container"]["infer_backend"]["status"] = STATUS_CONFIG.active
        info["container"]["infer_backend"]["id"] = model_service_info["container_name"]
        info["container"]["infer_backend"]["url"] = model_service_info[
            f"{service_type}_service"
        ]
        info["container"]["infer_backend"]["type"] = service_type
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.model_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(
            {
                "model_service_url": model_service_info[f"{service_type}_service"],
                "container_name": model_service_info["container_name"],
                "model_name": model_service_info["model_name"],
            }
        ),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_infer_backend(
    request_data: schema.PostInferBackendStop,
):
    validator.PostInferBackendStop(model_name=request_data.model_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(TASK_CONFIG.train, request_data.model_name)
        info = orjson.loads(info)
        container_id = info["container"]["infer_backend"]["id"]
        info["container"]["infer_backend"]["status"] = "stopped"
        info["container"]["infer_backend"]["url"] = None
        info["container"]["infer_backend"]["id"] = None
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.model_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        stopped_container = await utils.stop_model_service(
            container_name=container_id,
            infer_backend_type=info["container"]["infer_backend"]["type"],
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
        content=json.dumps({"stopped_container": stopped_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/")
async def get_infer_backend(
    model_name: Annotated[Union[str, None], Query()] = None,
):
    query_data = schema.GetInferBackend(model_name=model_name)
    validator.GetInferBackend(model_name=query_data.model_name)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.model_name:
            info = await redis_async.client.hget(
                TASK_CONFIG.train, query_data.model_name
            )
            info = orjson.loads(info)
            infer_backend_info = {
                "name": info["name"],
                "loaded": True
                if info["container"]["infer_backend"]["status"] == STATUS_CONFIG.active
                else False,
                "model_service_url": info["container"]["infer_backend"]["url"],
            }
        else:
            info = await redis_async.client.hgetall(TASK_CONFIG.train)
            infer_backend_info = [
                {
                    "name": (value := orjson.loads(v))["name"],
                    "loaded": value["container"]["infer_backend"]["status"]
                    == STATUS_CONFIG.active,
                    "model_service_url": value["container"]["infer_backend"]["url"],
                }
                for v in info.values()
            ]

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(infer_backend_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
