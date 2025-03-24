import json
from typing import Annotated, Union

import orjson
from fastapi import APIRouter, HTTPException, Query, Response, status

from src.config.params import TASK_CONFIG
from src.routers.info import schema, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/info", tags=["Info"])


@router.get("/support-model/")
async def get_support_model(base_model: Annotated[Union[str, None], Query()] = None):
    query_data = schema.GetSupportModel(base_model=base_model)
    validator.GetSupportModel(base_model=query_data.base_model)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.base_model is not None:
            info = await redis_async.client.hget(
                TASK_CONFIG.support_model, query_data.base_model
            )
            support_model_info = [orjson.loads(info)]
        else:
            info = await redis_async.client.hgetall(TASK_CONFIG.support_model)
            support_model_info = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
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
        content=json.dumps(support_model_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/eval-task/")
async def get_eval_task(eval_task: Annotated[Union[str, None], Query()] = None):
    query_data = schema.GetEvalTask(eval_task=eval_task)
    validator.GetEvalTask(eval_task=query_data.eval_task)
    error_handler = ResponseErrorHandler()

    try:
        if query_data.eval_task is not None:
            info = await redis_async.client.hget(
                TASK_CONFIG.eval_tasks, query_data.eval_task
            )
            eval_task_info = [orjson.loads(info)]
        else:
            info = await redis_async.client.hgetall(TASK_CONFIG.eval_tasks)
            eval_task_info = (
                [orjson.loads(value) for value in info.values()]
                if len(info) != 0
                else list()
            )

    except Exception as e:
        accel_logger.error(f"Database error: {e}")
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
        content=json.dumps(eval_task_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
