import json
import os

import orjson
from fastapi import APIRouter, HTTPException, Response, status

from src.config.params import COMMON_CONFIG, QUANTIZESERVICE_CONFIG, TASK_CONFIG
from src.routers.quantize import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")

router = APIRouter(prefix="/quantize", tags=["Quantize"])


@router.post("/start/")
async def start_quantize(request_data: schema.PostStartQuantize):
    validator.PostStartQuantize(
        checkpoint_path=os.path.join(SAVE_PATH, f"{request_data.quantize_name}")
    )
    error_handler = ResponseErrorHandler()

    try:
        await utils.quantize_as_gguf(
            quantize_service_url=f"http://{QUANTIZESERVICE_CONFIG.container_name}:{QUANTIZESERVICE_CONFIG.port}/gguf/full/",
            quantize_name=request_data.quantize_name,
            checkpoint_path=os.path.join(
                COMMON_CONFIG.save_path, request_data.quantize_name, "merge"
            ),
            output_path=os.path.join(
                COMMON_CONFIG.save_path, request_data.quantize_name, "quantize"
            ),
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
        info = await redis_async.client.hget(
            TASK_CONFIG.train, request_data.quantize_name
        )
        info = orjson.loads(info)
        info["is_quantize"] = True
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.quantize_name, orjson.dumps(info)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg=f"Database error: {e}",
            input=request_data.model_dump(),
        )

    return Response(
        content=json.dumps({"quantize_name": request_data.quantize_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.post("/stop/")
async def stop_quantize(request_data: schema.PostStopQuantize):
    quantize_container = validator.PostStopQuantize(
        quantize_container=request_data.quantize_container
    ).quantize_container
    error_handler = ResponseErrorHandler()

    try:
        quantize_container = await utils.stop_quantize(
            container_name_or_id=quantize_container
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
        content=json.dumps({"quantize_container": quantize_container}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
