import json
import os
from typing import Annotated

import orjson
from fastapi import APIRouter, Query, Response, status
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from src import thirdparty
from src.config import params
from src.routers.accelbrain import schema, utils, validator
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import get_current_time

router = APIRouter(prefix="/accelbrain", tags=["Accelbrain"])


@router.post("/deploy/")
async def deploy_accelbrain(request_data: schema.PostDeploy):
    validator.PostDeploy(deploy_name=request_data.deploy_name)
    error_handler = ResponseErrorHandler()

    try:
        return StreamingResponse(
            content=utils.deploy_to_accelbrain_service(
                file_path=os.path.join(
                    params.COMMON_CONFIG.save_path, request_data.deploy_name, "quantize"
                ),
                model_name=request_data.deploy_name,
                deploy_path=os.path.join(
                    params.COMMON_CONFIG.save_path,
                    request_data.deploy_name,
                    "quantize",
                    f"{request_data.deploy_name}.zip",
                ),
                accelbrain_url=request_data.accelbrain_url,
            ),
            status_code=status.HTTP_200_OK,
            media_type="text/event-stream",
        )

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"deploy_name": request_data.deploy_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None


@router.get("/health/")
async def check_accelbrain(accelbrain_url: Annotated[str, Query(...)]):
    query_data = schema.GetHealthcheck(accelbrain_url=accelbrain_url)
    error_handler = ResponseErrorHandler()

    try:
        accelbrain_status, accelbrain_status_code = await utils.check_accelbrain_url(
            accelbrain_url=query_data.accelbrain_url
        )

    except ValueError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"accelbrain_url": query_data.accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_QUERY],
            msg=f"Unexpected error: {e}",
            input={"accelbrain_url": query_data.accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"status": accelbrain_status}),
        status_code=accelbrain_status_code,
        media_type="application/json",
    )


@router.post("/device/")
async def set_device(request_data: schema.PostDevice):
    current_time = get_current_time(use_unix=True)
    validator.PostDevice(
        accelbrain_device=request_data.accelbrain_device,
        accelbrain_url=request_data.accelbrain_url,
    )
    error_handler = ResponseErrorHandler()

    try:
        device_info = {
            "device": request_data.accelbrain_device,
            "url": request_data.accelbrain_url,
            "created_time": current_time,
            "modified_time": None,
        }
        await thirdparty.redis.handler.redis_async.client.hset(
            params.TASK_CONFIG.accelbrain_device,
            request_data.accelbrain_device,
            orjson.dumps(device_info),
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
        content=json.dumps([device_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/device/")
async def get_device(accelbrain_device: Annotated[str, Query(...)] = None):
    query_data = schema.GetDevice(accelbrain_device=accelbrain_device)
    validator.GetDevice(accelbrain_device=query_data.accelbrain_device)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.accelbrain_device:
            accelbrain_device_info = (
                await thirdparty.redis.handler.redis_async.client.hget(
                    params.TASK_CONFIG.accelbrain_device, query_data.accelbrain_device
                )
            )
            device_info = [orjson.loads(accelbrain_device_info)]
        else:
            info = await thirdparty.redis.handler.redis_async.client.hgetall(
                params.TASK_CONFIG.accelbrain_device
            )
            device_info = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(device_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.put("/device/")
async def modify_device(request_data: schema.PutDevice):
    modified_time = get_current_time(use_unix=True)
    validator.PutDevice(
        accelbrain_device=request_data.accelbrain_device,
        accelbrain_url=request_data.accelbrain_url,
    )
    error_handler = ResponseErrorHandler()

    try:
        device_info = await thirdparty.redis.handler.redis_async.client.hget(
            params.TASK_CONFIG.accelbrain_device,
            request_data.accelbrain_device,
        )
        device_info = orjson.loads(device_info)
        device_info["url"] = request_data.accelbrain_url
        device_info["modified_time"] = modified_time
        await thirdparty.redis.handler.redis_async.client.hset(
            params.TASK_CONFIG.accelbrain_device,
            request_data.accelbrain_device,
            orjson.dumps(device_info),
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
        content=json.dumps([device_info]),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.delete("/device/")
async def delete_device(accelbrain_device: Annotated[str, Query(...)]):
    query_data = schema.DelDevice(accelbrain_device=accelbrain_device)
    validator.DelDevice(accelbrain_device=query_data.accelbrain_device)
    error_handler = ResponseErrorHandler()

    try:
        device_info = await thirdparty.redis.handler.redis_async.client.hget(
            params.TASK_CONFIG.accelbrain_device, query_data.accelbrain_device
        )
        device_info = orjson.loads(device_info)
        await thirdparty.redis.handler.redis_async.client.hdel(
            params.TASK_CONFIG.accelbrain_device, query_data.accelbrain_device
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps([device_info]),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        media_type="application/json",
    )
