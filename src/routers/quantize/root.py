import asyncio
import json
import os

import orjson
from fastapi import APIRouter, HTTPException, Response, status

from src.config.params import (
    COMMON_CONFIG,
    QUANTIZESERVICE_CONFIG,
    STATUS_CONFIG,
    TASK_CONFIG,
)
from src.routers.quantize import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/quantize", tags=["Quantize"])


@router.post("/start/")
async def start_quantize(request_data: schema.PostStartQuantize):
    validator.PostStartQuantize(quantize_name=request_data.quantize_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(
            TASK_CONFIG.train, request_data.quantize_name
        )
        info = orjson.loads(info)

        container_name = await utils.quantize_as_gguf(
            quantize_service_url=f"http://{QUANTIZESERVICE_CONFIG.container_name}:{QUANTIZESERVICE_CONFIG.port}/gguf/full/",
            quantize_name=request_data.quantize_name,
            checkpoint_path=info["last_model_path"],
            output_path=os.path.join(
                COMMON_CONFIG.save_path, request_data.quantize_name, "quantize"
            ),
        )

        await utils.merge_async_tasks(
            quantize_name=request_data.quantize_name, container_name=container_name
        )
        quantize_status = STATUS_CONFIG.finish

    except asyncio.CancelledError:
        accel_logger.info("asyncio cancelled")
        quantize_status = STATUS_CONFIG.stopped

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        quantize_status = STATUS_CONFIG.failed

    finally:
        try:
            if quantize_status != STATUS_CONFIG.finish:
                accel_logger.info("quantize is not finish, delete exist folder")
                await utils.del_quantize_folder(
                    qunatize_folder=os.path.join(
                        COMMON_CONFIG.save_path, request_data.quantize_name, "quantize"
                    )
                )
            await utils.remove_finish_container(container_name=container_name)
        except Exception as e:
            accel_logger.error(f"Failed to remove container, {e}")

    try:
        info = await redis_async.client.hget(
            TASK_CONFIG.train, request_data.quantize_name
        )
        info = orjson.loads(info)
        info["container"]["quantize"]["status"] = quantize_status
        info["container"]["quantize"]["id"] = None
        await redis_async.client.hset(
            TASK_CONFIG.train, request_data.quantize_name, orjson.dumps(info)
        )
        database_done = True

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg="Database error",
            input=request_data.model_dump(),
        )
        database_done = False
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    finally:
        if not database_done:
            await utils.del_quantize_folder(
                qunatize_folder=os.path.join(
                    COMMON_CONFIG.save_path, request_data.quantize_name, "quantize"
                )
            )

    if quantize_status == STATUS_CONFIG.finish:
        return Response(
            content=json.dumps({"quantize_name": request_data.quantize_name}),
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )
    elif quantize_status == STATUS_CONFIG.stopped:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Client Closed Request",
            input=request_data.model_dump(),
        )
        raise HTTPException(status_code=499, detail=error_handler.errors)
    elif quantize_status == STATUS_CONFIG.failed:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        )


@router.post("/stop/")
async def stop_quantize(request_data: schema.PostStopQuantize):
    validator.PostStopQuantize(quantize_name=request_data.quantize_name)
    error_handler = ResponseErrorHandler()

    try:
        info = await redis_async.client.hget(
            TASK_CONFIG.train, request_data.quantize_name
        )
        info = orjson.loads(info)

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg="Database error",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    try:
        await utils.stop_quantize(
            container_name_or_id=info["container"]["quantize"]["id"]
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
        content=json.dumps({"quantize_name": request_data.quantize_name}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
