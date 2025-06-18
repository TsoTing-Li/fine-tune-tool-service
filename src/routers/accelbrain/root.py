import json
import os
from typing import Annotated, Union
from uuid import UUID

import orjson
from fastapi import APIRouter, Query, Response, status
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from src.config.params import STATUS_CONFIG, TASK_CONFIG
from src.routers.accelbrain import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import generate_uuid, get_current_time

router = APIRouter(prefix="/accelbrain", tags=["AccelBrain"])


@router.post("/deploy/start/")
async def start_deploy_accelbrain(request_data: schema.PostDeploy):
    validator.PostDeploy(
        deploy_name=request_data.deploy_name,
        device_uuid=request_data.device_uuid,
    )
    error_handler = ResponseErrorHandler()

    try:
        accelbrain_device_info = await redis_async.client.hget(
            TASK_CONFIG.accelbrain_device, str(request_data.device_uuid)
        )
        accelbrain_device_info = orjson.loads(accelbrain_device_info)

        info = await redis_async.client.hget(
            TASK_CONFIG.train, request_data.deploy_name
        )
        info = orjson.loads(info)

        deploy_unique_key = f"{request_data.deploy_name}-{request_data.device_uuid}"
        await redis_async.client.hset(
            TASK_CONFIG.deploy,
            deploy_unique_key,
            orjson.dumps(
                {
                    "deploy_model": request_data.deploy_name,
                    "deploy_device": accelbrain_device_info["name"],
                    "deploy_url": accelbrain_device_info["url"],
                    "status": STATUS_CONFIG.active,
                }
            ),
        )

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
        return StreamingResponse(
            content=utils.deploy_to_accelbrain_service(
                file_path=os.path.join(
                    os.path.dirname(info["train_args"]["output_dir"]), "quantize"
                ),
                model_name=request_data.deploy_name,
                train_args=info["train_args"],
                last_model_path=info["last_model_path"],
                deploy_path=os.path.join(
                    os.path.dirname(info["train_args"]["output_dir"]),
                    "deploy",
                    f"{deploy_unique_key}.zip",
                ),
                deploy_unique_key=deploy_unique_key,
                accelbrain_url=accelbrain_device_info["url"],
            ),
            status_code=status.HTTP_200_OK,
            media_type="text/event-stream",
        )

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input={
                "deploy_name": request_data.deploy_name,
                "device_uuid": str(request_data.device_uuid),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None


@router.get("/deploy/")
async def get_deploy_status(
    deploy_name: Annotated[Union[str, None], Query()] = None,
    device_uuid: Annotated[Union[UUID, None], Query()] = None,
):
    query_data = schema.GetDeploy(deploy_name=deploy_name, device_uuid=device_uuid)
    validator.GetDeploy(
        deploy_name=query_data.deploy_name, device_uuid=query_data.device_uuid
    )
    error_handler = ResponseErrorHandler()

    try:
        if query_data.deploy_name is not None and query_data.device_uuid is not None:
            deploy_status = await redis_async.client.hget(
                TASK_CONFIG.deploy, f"{query_data.deploy_name}-{query_data.device_uuid}"
            )
            deploy_status = [orjson.loads(deploy_status)]
        else:
            info = await redis_async.client.hgetall(TASK_CONFIG.deploy)
            deploy_status = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception:
        error_handler.add(
            type=error_handler.ERR_REDIS,
            loc=[error_handler.LOC_DATABASE],
            msg="Database error",
            input=query_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(deploy_status),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/health/")
async def check_accelbrain(
    url: Annotated[str, Query(..., min_length=9, max_length=21)],
):
    query_data = schema.GetHealthCheck(url=url)
    error_handler = ResponseErrorHandler()

    try:
        accelbrain_status, accelbrain_status_code = await utils.check_accelbrain_url(
            accelbrain_url=query_data.url
        )

    except ConnectionError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_CONNECTION,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"url": query_data.url},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_handler.errors
        ) from None

    except TimeoutError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_TIMEOUT,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"url": query_data.url},
        )
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=error_handler.errors,
        ) from None

    except RuntimeError as e:
        accel_logger.error(f"Runtime error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"url": query_data.url},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_QUERY],
            msg="Unexpected error",
            input={"url": query_data.url},
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
    unix_time, _ = get_current_time()
    accelbrain_device_uuid = generate_uuid()
    validator.PostDevice(
        name=request_data.name,
        url=request_data.url,
    )
    error_handler = ResponseErrorHandler()

    try:
        device_info = {
            "uuid": accelbrain_device_uuid,
            "name": request_data.name,
            "url": request_data.url,
            "created_time": unix_time,
            "modified_time": None,
        }
        await redis_async.client.hset(
            TASK_CONFIG.accelbrain_device,
            accelbrain_device_uuid,
            orjson.dumps(device_info),
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Database error",
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
async def get_device(uuid: Annotated[Union[UUID, None], Query()] = None):
    query_data = schema.GetDevice(uuid=uuid)
    validator.GetDevice(uuid=query_data.uuid)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.uuid:
            accelbrain_device_info = await redis_async.client.hget(
                TASK_CONFIG.accelbrain_device, str(query_data.uuid)
            )
            device_info = [orjson.loads(accelbrain_device_info)]
        else:
            info = await redis_async.client.hgetall(TASK_CONFIG.accelbrain_device)
            device_info = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Database error",
            input={"uuid": str(query_data.uuid)},
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
    unix_time, _ = get_current_time()
    validator.PutDevice(
        uuid=request_data.uuid, name=request_data.name, url=request_data.url
    )
    error_handler = ResponseErrorHandler()

    try:
        device_info = await redis_async.client.hget(
            TASK_CONFIG.accelbrain_device,
            str(request_data.uuid),
        )
        device_info = orjson.loads(device_info)
        device_info["name"] = request_data.name or device_info["name"]
        device_info["url"] = request_data.url or device_info["url"]
        device_info["modified_time"] = unix_time
        await redis_async.client.hset(
            TASK_CONFIG.accelbrain_device,
            str(request_data.uuid),
            orjson.dumps(device_info),
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Database error",
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
async def delete_device(uuid: Annotated[UUID, Query(...)]):
    query_data = schema.DelDevice(uuid=uuid)
    validator.DelDevice(uuid=query_data.uuid)
    error_handler = ResponseErrorHandler()

    try:
        device_info = await redis_async.client.hget(
            TASK_CONFIG.accelbrain_device, str(query_data.uuid)
        )
        device_info = orjson.loads(device_info)
        await redis_async.client.hdel(
            TASK_CONFIG.accelbrain_device, str(query_data.uuid)
        )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Database error",
            input=query_data.model_dump(),
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
